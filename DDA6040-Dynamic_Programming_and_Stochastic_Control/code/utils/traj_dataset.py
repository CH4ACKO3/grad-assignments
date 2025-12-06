from torch.utils.data import Dataset
import torch
import numpy as np
import random
import heapq
from collections import deque
from pipelines.utils.tokenizer import Tokenizer
import logging
from time import time
import sys
from collections import defaultdict
from typing import Optional
import h5py
from math import ceil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ShortestPathDataset(Dataset):
    def __init__(self, evaluate=False, train_classes=1000):
        self.data = []
        self.evaluate = evaluate
        tokenizer = Tokenizer(train_classes=train_classes)
        with h5py.File(f"data/dijkstra_8.h5", "r") as f:
            for i in range(len(f)):
                seq = f[f"seq_{i}"]
                if not self.evaluate:
                    tokenizer.shuffle()

                self.data.append({
                    "prompt": np.array(tokenizer.encode(seq["prompt"][:])),
                    "events": np.array(tokenizer.encode(seq["events"][:])),
                })
        if not self.evaluate:
            self.data = self.data[:int(len(self.data)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        decoder_input_ids = np.concatenate([item['prompt'], item['events']], axis=0)
        decoder_labels = np.concatenate([np.full((item['prompt'].shape[0]+1,), -100), item['events'][1:]], axis=0)
        attention_mask = np.concatenate([np.ones((item['prompt'].shape[0],)), np.ones((item['events'].shape[0],))], axis=0)

        return {
            'decoder_input_ids': torch.from_numpy(decoder_input_ids).to(dtype=torch.long),
            'decoder_labels': torch.from_numpy(decoder_labels).to(dtype=torch.long),
            'attention_mask': torch.from_numpy(attention_mask).to(dtype=torch.bool),
        }

class CustomCollator:
    def __init__(self, pad_token_id=0, max_length=-1):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        # Extract lengths more efficiently
        lengths_list = [item['decoder_input_ids'].shape[0] for item in batch]
        lengths = torch.tensor(lengths_list, dtype=torch.long).unsqueeze(1)
        max_dec_len = lengths.max().item() if self.max_length == -1 else min(self.max_length, lengths.max().item())

        # Pre-allocate tensors
        decoder_input_ids_list = torch.zeros((len(batch), max_dec_len), dtype=torch.long)
        decoder_labels_list = torch.full((len(batch), max_dec_len), -100, dtype=torch.long)
        attention_mask_list = torch.zeros((len(batch), max_dec_len), dtype=torch.bool)
        
        # Vectorized assignment
        for i, item in enumerate(batch):
            seq_len = lengths_list[i]
            decoder_input_ids_list[i, :seq_len] = item['decoder_input_ids']
            decoder_labels_list[i, :seq_len] = item['decoder_labels']
            attention_mask_list[i, :seq_len] = item['attention_mask']
        
        return {
            'decoder_input_ids': decoder_input_ids_list,
            'decoder_labels': decoder_labels_list,
            'attention_mask': attention_mask_list,
            'lengths': lengths
        }

class DijkstraGenerator:
    def __init__(self, graph_size=8, max_edge_prob=1.0):
        self.graph_size = graph_size
        self.max_edge_prob = max_edge_prob

    def generate_sequence(self):
        """generate a random sequence"""

        # Sample path probability for this graph
        edge_prob = np.random.uniform(0.0, self.max_edge_prob)
        
        adj_mat = np.full((self.graph_size, self.graph_size), -1)
        
        edge_probs = np.random.rand(self.graph_size, self.graph_size)
        
        edge_mask = edge_probs < edge_prob
        
        weights = np.random.randint(0, 10, (self.graph_size, self.graph_size))
        
        adj_mat[edge_mask] = weights[edge_mask]

        start, goal = np.random.choice(range(self.graph_size), size=2, replace=False)

        # Generate prompt
        prompt = ["start", f"node_{start}", "goal", f"node_{goal}"]
        for i in range(self.graph_size):
            for j in range(self.graph_size):
                if adj_mat[i, j] != -1:
                    prompt.extend(["edge", f"node_{i}", f"node_{j}", str(adj_mat[i, j])])
        
        # Dijkstra
        events = ["bos"]
        visited = set()
        queue = [(0, start)]
        dis = defaultdict(lambda: float('inf'))
        dis[start] = 0
        prev = defaultdict(lambda: None)

        while queue:
            dist, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            events.extend(["close", f"node_{node}", str(dis[node])])
            if node == goal:
                break
            for neighbor in range(self.graph_size):
                if adj_mat[node, neighbor] != -1 and neighbor not in visited:
                    if dis[neighbor] > dis[node] + adj_mat[node, neighbor]:
                        dis[neighbor] = dis[node] + adj_mat[node, neighbor]
                        prev[neighbor] = node
                        heapq.heappush(queue, (dist + adj_mat[node, neighbor], neighbor))
                        events.extend(["open", f"node_{neighbor}", str(dis[neighbor])])
        
        if dis[goal] < float('inf'):
            # backtrack
            path = []
            node = goal
            while node is not None:
                path.append(node)
                events.extend(["path", f"node_{node}", str(dis[node])])
                node = prev[node]
        
        events.extend(["eos"])
        return prompt, events
    
    def generate_dataset(self, num_seq:int = 100_000):
        sequences = []
        lengths = []
        while (len(sequences) < num_seq):
            prompt, events = self.generate_sequence()
            sequences.append({
                "prompt": prompt,
                "events": events,
            })
            lengths.append(len(prompt)+len(events))
        logger.info(f"Generated {len(sequences)} sequences with length avg={np.mean(lengths)}, max={np.max(lengths)}, min={np.min(lengths)}")
        return sequences

if __name__ == "__main__":
    import sys
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.info("Starting DijkstraGenerator")

    generator = DijkstraGenerator(grid_size=8, wall_prob=0.3)
    trajectories = generator.generate_dataset(lengths=np.array([[200, 300]]), num_each=100)