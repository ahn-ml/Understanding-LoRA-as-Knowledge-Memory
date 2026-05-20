# src/trainers/ntp_trainer.py

"""
Next Token Prediction (NTP) Trainer.
Trains each sample independently to memorize data (similar to memorizing a phonebook).
"""

import os
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler

# Import internal project modules
from src.utils import log_to_wandb
from src.models.model_loader import apply_chat_template


class SimpleDictDataset(Dataset):
    """PyTorch Dataset class for a simple list of dictionaries."""
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class NTPTrainer:
    def __init__(self, config, model, tokenizer, dataset):
        self.config = config
        self.training_config = config['training']
        self.model = model
        self.tokenizer = tokenizer

        # Load output directory directly from config
        self.output_dir = config['output_dir']

        self.device = next(model.parameters()).device
        
        # Create Data Loader
        train_dataset = SimpleDictDataset(dataset)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate']
        )
        
        # Learning Rate Scheduler setup
        warmup_ratio = self.training_config.get("warmup_ratio", 0.0)
        num_train_steps = self.training_config['num_train_steps']
        num_warmup_steps = int(num_train_steps * warmup_ratio)

        logging.info(f"Scheduler: Cosine with warmup. Total steps: {num_train_steps}, Warmup steps: {num_warmup_steps}")

        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
        
        logging.info(f"NTPTrainer initialized (Batch size: {self.training_config['batch_size']})")
    
    def _collate_fn(self, batch):
        """
        Tokenizes batch data and applies Chat Template if configured.
        """
        use_template = self.config.get('dataset', {}).get('use_chat_template', False)
        
        texts = []
        for item in batch:
            raw_text = item['text']
            # Check if Chat Template should be applied (Universal for Llama, Qwen, etc.)
            if use_template:
                messages = [{"role": "user", "content": raw_text}]
                # Apply template using tokenizer's internal function
                texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                # If no template, combine raw text with EOS token
                texts.append(raw_text + self.tokenizer.eos_token)
        
        # Tokenization
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048, # Set generously considering context length
            return_tensors="pt"
        )
        
        # Generate Labels for Causal LM (same as input_ids, mask padding with -100)
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
    
    def _save_checkpoint(self, step_or_name: str):
        """Saves LoRA adapter (checkpoint)."""
        checkpoint_dir = os.path.join(self.output_dir, str(step_or_name))
        os.makedirs(checkpoint_dir, exist_ok=True)
        # PEFT model saves only LoRA weights via save_pretrained
        self.model.save_pretrained(checkpoint_dir)
        logging.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def train(self):
        """Main training loop."""
        self.model.train()
        num_steps = self.training_config['num_train_steps']
        
        # Create an infinite iterator from the dataloader
        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(range(num_steps), desc="NTP Training")
        
        for step in progress_bar:
            try:
                batch = next(data_iterator)
            except StopIteration:
                # Restart from the beginning if dataloader finishes an epoch
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)
            
            # Move batch to GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            if (step + 1) % 10 == 0: # Log every 10 steps
                wandb_metrics = {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                }
                log_to_wandb(wandb_metrics, step=step + 1)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.7f}"
            })
            
            # Checkpoint strategy
            strategy_cfg = self.training_config.get('checkpoint_strategy', {})
            if strategy_cfg.get('policy') == "steps" and (step + 1) % strategy_cfg.get('value', num_steps + 1) == 0:
                self._save_checkpoint(f"step_{step+1}")
        
        # Save final model
        self._save_checkpoint("final")
        logging.info("NTP training complete.")