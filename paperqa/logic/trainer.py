# paperqa/logic/trainer.py

import os
import logging
from tqdm import tqdm
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

class PaperQADataset(Dataset):
    """PyTorch Dataset class for PaperQA data."""
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        logging.info(f"Loaded {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class PaperQATrainer:
    """PaperQA Trainer - Supports both QA and NTP tasks."""
    
    def __init__(self, config: dict, model, tokenizer):
        self.config = config
        self.training_config = config['training']
        self.dataset_config = config.get('dataset', {})
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Detect data type (QA vs Introduction)
        self.data_type = self._detect_data_type()
        logging.info(f"Detected data type: {self.data_type}")
        
        # Check for chat template usage (Mandatory)
        self.use_chat_template = self.dataset_config.get('use_chat_template', False)
        if not self.use_chat_template:
            raise ValueError("PaperQA requires use_chat_template: true. Chat template is mandatory for proper data formatting.")
        logging.info("Chat template: ENABLED (required)")
        
        # Set output directory
        self.output_dir = config.get('output_dir', 'runs/paperqa')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.training_config['learning_rate']
        )

        # Set learning rate scheduler
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
        
        # Create dataloader
        self.train_dataloader = self._create_dataloader()

    def _detect_data_type(self):
        """Automatically detect data type (QA vs Introduction)."""
        data_file = self.training_config['data_path']
        
        # Determine based on the first sample
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                sample = json.loads(first_line)
                
                if 'question' in sample and 'answer' in sample:
                    logging.info(f"Detected QA data type from sample keys: {list(sample.keys())}")
                    return 'qa'
                elif 'text' in sample:
                    logging.info(f"Detected Introduction data type from sample keys: {list(sample.keys())}")
                    return 'introduction'
        except Exception as e:
            logging.warning(f"Could not detect data type from file: {e}")
        
        # Fallback: Determine based on filename
        if 'qa' in data_file.lower():
            logging.info(f"Detected QA data type from filename: {data_file}")
            return 'qa'
        elif 'introduction' in data_file.lower():
            logging.info(f"Detected Introduction data type from filename: {data_file}")
            return 'introduction'
        
        # Default
        raise ValueError(f"Could not detect data type from file: {data_file}")

    def _collate_fn(self, batch: list[dict]) -> dict:
        """Tokenize batch data and convert to appropriate format."""
        texts = []
        
        for item in batch:
            if self.data_type == 'qa':
                # Process QA data
                question = item['question']
                answer = item['answer']
                
                # Apply chat template
                combined_text = f"Question: {question}\nAnswer: {answer}"
                messages = [{"role": "user", "content": combined_text}]
                formatted_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                
                # Add EOS token
                if not formatted_text.endswith(self.tokenizer.eos_token):
                    formatted_text += self.tokenizer.eos_token
                
                texts.append(formatted_text)
            else:
                # Process Introduction data
                raw_text = item['text']
                
                # Apply chat template
                messages = [{"role": "user", "content": raw_text}]
                formatted_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                
                # Add EOS token
                if not formatted_text.endswith(self.tokenizer.eos_token):
                    formatted_text += self.tokenizer.eos_token
                
                texts.append(formatted_text)

        if not texts:
            logging.warning("Collate function created an empty batch. Check data format.")
            return {}

        # Tokenize
        encodings = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=2048,
            return_tensors="pt"
        )
        
        # Generate Labels
        labels = encodings['input_ids'].clone()
        
        # Mask padding tokens with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encodings['input_ids'], 
            'attention_mask': encodings['attention_mask'], 
            'labels': labels
        }

    def _create_dataloader(self) -> DataLoader:
        """Create dataloader."""
        dataset = PaperQADataset(self.training_config['data_path'])
        logging.info(f"Loaded dataset from '{self.training_config['data_path']}' with {len(dataset)} samples.")
        
        if len(dataset) == 0:
            logging.error("Dataset is empty. Training cannot proceed.")
            raise ValueError("Training data is empty.")
        
        return DataLoader(
            dataset, 
            batch_size=self.training_config['batch_size'],
            shuffle=True, 
            collate_fn=self._collate_fn
        )

    def _save_model(self, checkpoint_name="final"):
        """Save the model."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        logging.info(f"Saved LoRA adapter to '{checkpoint_dir}'.")

    def train(self):
        """Main training loop."""
        self.model.train()
        num_steps = self.training_config['num_train_steps']
        
        # Create infinite iterator from dataloader
        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(range(num_steps), desc=f"Training {self.config.get('experiment_name', 'PaperQA')}")
        
        for step in progress_bar:
            try:
                batch = next(data_iterator)
            except StopIteration:
                # Restart dataloader after one epoch
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)
            
            if not batch: 
                logging.warning(f"Step {step}: Skipping empty batch.")
                continue

            # Debug info (first few steps)
            if step < 2:
                logging.info(f"--- Debug Step {step} ---")
                logging.info(f"Data type: {self.data_type}")
                logging.info(f"Batch keys: {batch.keys()}")
                logging.info(f"Input IDs shape: {batch['input_ids'].shape}")
                logging.info(f"Labels shape: {batch['labels'].shape}")
                # Check label of first sample (for padding masking verification)
                first_labels = batch['labels'][0]
                masked_count = (first_labels == -100).sum().item()
                total_count = len(first_labels)
                logging.info(f"First sample: {masked_count}/{total_count} tokens masked (padding)")

            # Move batch to GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Check for NaN
            if torch.isnan(loss):
                logging.error("Loss is NaN. Stopping training.")
                break
            
            # Backward pass and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update progress
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.7f}"
            })
            
            # Logging
            if (step + 1) % 100 == 0:
                logging.info(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")
            
            # Checkpoint save strategy
            strategy_cfg = self.training_config.get('checkpoint_strategy', {})
            if strategy_cfg.get('policy') == "steps" and (step + 1) % strategy_cfg.get('value', num_steps + 1) == 0:
                self._save_model(f"step_{step+1}")

        # Save final model
        self._save_model("final")
        logging.info("PaperQA training completed.")
