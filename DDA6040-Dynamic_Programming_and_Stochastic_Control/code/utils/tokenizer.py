import numpy as np
from typing import List, Union

class Tokenizer:
    def __init__(self, train_classes=1000):
        self.train_classes = train_classes
        self.dict = {
            "pad": 0,
            "bos": 1,
            "eos": 2,
            "start": 3,
            "goal": 4,
            "edge": 5,
            "open": 6,
            "close": 7,
            "path": 8,
        }

        self.dict.update({str(i): len(self.dict)+i for i in range(100-len(self.dict))})

        self.dict.update({f"node_{i}": 100+i for i in range(10)})

        self.inv_dict = {v: k for k, v in self.dict.items()}
    
    def shuffle(self):
        idx = np.random.choice(range(110, 110 + self.train_classes), size=10, replace=False)
        self.dict.update({f"node_{i}": int(idx[i]) for i in range(10)})
        self.inv_dict = {v: k for k, v in self.dict.items()}
    
    def encode(self, tokens:Union[np.ndarray, List[str], str]):
        if isinstance(tokens, np.ndarray):
            return [self.dict[token.decode('utf-8')] for token in tokens]
        elif isinstance(tokens, str):
            return self.dict[tokens]
        elif isinstance(tokens, list):
            return [self.dict[token] for token in tokens]
        else:
            raise ValueError(f"Unsupported type: {type(tokens)}")
    
    def decode(self, ids:Union[np.ndarray, List[int], int, np.integer]):
        if isinstance(ids, np.integer):
            return self.inv_dict[ids.item()]
        elif isinstance(ids, int):
            return self.inv_dict[ids]
        elif isinstance(ids, np.ndarray):
            if ids.ndim == 0:
                return self.inv_dict[ids.item()]
            else:
                # return a high-dimensional array of strings
                return np.vectorize(lambda x: self.inv_dict[int(x)])(ids)
        elif isinstance(ids, list):
            return [self.inv_dict[id] for id in ids]
        else:
            raise ValueError(f"Unsupported type: {type(ids)}")
    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokens = ["start", "node_0", "goal", "node_2", "edge", "node_0", "node_1", "1", "edge", "node_1", "node_2", "2",
        "bos", "close", "node_0", "0", "open", "node_1", "1", "close", "node_1", "1", "open", "node_2", "3", "close", "node_2", "3", "path", "node_2", "3", "path", "node_1", "1", "path", "node_0", "0", "eos"]
    print(f"Tokens: {tokens}")
    tokenizer.shuffle()
    encoded = tokenizer.encode(tokens)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")