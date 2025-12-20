from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import json
import sys
import logging
import uuid
import gc
import time
from datetime import datetime
import random
from pipelines.utils.traj_dataset import ShortestPathDataset, CustomCollator
from pipelines.utils.storage import MetadataStorage
from pipelines.utils.model import print_params
from transformers import LlamaConfig, get_scheduler, LlamaForCausalLM

torch.set_float32_matmul_precision("high")

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for faster training on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

exp_uuid = str(uuid.uuid4())

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.INFO)
root_logger.addHandler(stream_handler)

file_handler = logging.FileHandler(f"logs/exp/{exp_uuid}.log")
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--b_size", type=int, default=1024)
    parser.add_argument("--mb_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--size", type=str, default="medium")
    parser.add_argument("--train_classes", type=int, default=1000)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for model optimization")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of data loading workers (default: cpu_count//4)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch")
    parser.add_argument("--seed", type=int, default=42)
    
    args, unknown = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_logger.info(f"{exp_uuid} - Training started with configuration: {args} and {unknown}")

    model_storage = MetadataStorage(storage_dir="models", fix_uuid=exp_uuid)
    model_storage_config = {
        "b_size": args.b_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "train_classes": args.train_classes,
        "seed": args.seed,
        "size": args.size,
        **{k.lstrip('-'): v for k, v in zip(unknown[::2], unknown[1::2])},
    }
    root_logger.info(f"Model storage configuration: {model_storage_config}")

    train_dataset = ShortestPathDataset(train_classes=args.train_classes)
    collator = CustomCollator()
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() // 8)
    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.mb_size, 
        shuffle=True, 
        collate_fn=collator, 
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        drop_last=True
    )
    root_logger.info(f"Dataset initialized with {len(train_dataset)} samples, num_workers={num_workers}")

    # Initialize model
    with open("size.json", "r") as f:
        config_dict = json.load(f)
    model_config = LlamaConfig(**config_dict[args.size])
    model_config.vocab_size = args.train_classes + 110
    model = LlamaForCausalLM(model_config).to(device)
    model.train()

    if args.compile:
        model = torch.compile(model, mode="default")
        root_logger.info("Model compiled successfully with dynamic shape support")

    print_params(model)

    # Initialize optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.num_epochs * 0.05),
        num_training_steps=args.num_epochs
    )
    scaler = torch.amp.GradScaler()

    root_logger.info(f"Train loop started{', may take a while to compile the model...' if args.compile else ''}")
    try:
        start_time = time.time()
        for epoch in range(args.num_epochs):
            total_loss = 0
            b_count = 0
            optimizer.zero_grad()
            
            for i, mini_batch in enumerate(dataloader):
                # Non-blocking transfer for better overlap with computation
                decoder_input_ids = mini_batch['decoder_input_ids'].to(device, non_blocking=True)
                decoder_labels = mini_batch['decoder_labels'].to(device, non_blocking=True)
                attention_mask = mini_batch['attention_mask'].to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(decoder_input_ids, attention_mask=attention_mask, labels=decoder_labels)

                # Single backward call with combined loss
                scaler.scale(outputs.loss).backward()
                
                total_loss += outputs.loss.item()
                b_count += decoder_input_ids.shape[0]
                
                # Update weights when accumulated batch size is reached
                if b_count >= args.b_size or i == len(dataloader) - 1:
                    # Gradient clipping for stability (optional but recommended)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    b_count = 0

            lr_scheduler.step()
            avg_loss = total_loss / len(dataloader)
            # Format loss in scientific notation as 3 significant digits, e.g. 1.234x10^-3
            if avg_loss == 0:
                formatted_loss = "0.000x10^+0"
            else:
                exponent = int(np.floor(np.log10(abs(avg_loss))))
                mantissa = avg_loss / 10**exponent
                formatted_loss = f"{mantissa:.3f}x10^{exponent:+d}"
            current_time = time.time()
            estimated_time = (current_time - start_time) / (epoch + 1) * args.num_epochs
            eta_time = start_time + estimated_time
            eta_time = datetime.fromtimestamp(eta_time).strftime("%d-%H:%M:%S")
            root_logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {formatted_loss}, Time: {(current_time - start_time)/3600:.2f}h/{(estimated_time)/3600:.2f}h, ETA: {eta_time}")
        root_logger.info(f"Training finished successfully")
        # Save model state dict - handle both compiled and non-compiled models
        if hasattr(model, '_orig_mod'):
            # Model was compiled with torch.compile
            torch.save(model._orig_mod.state_dict(), f"tmp/{exp_uuid}.pt")
        else:
            # Model was not compiled
            torch.save(model.state_dict(), f"tmp/{exp_uuid}.pt")
        model_uuid = model_storage.store_file(file_path=f"tmp/{exp_uuid}.pt", metadata=model_storage_config)
        root_logger.info(f"Model saved successfully with UUID: {model_uuid}")
    finally:
        # Explicitly cleanup DataLoader workers to prevent hanging
        if num_workers > 0:
            root_logger.info("Cleaning up DataLoader workers...")
            del dataloader
            gc.collect()
            root_logger.info("DataLoader workers cleaned up")
        
        # Close log handlers to ensure all logs are flushed
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)