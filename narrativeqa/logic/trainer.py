# logic/trainer.py

import json
import logging

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from path_utils import OUTPUTS_DIR, resolve_with_root

class SimpleQADataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ExperimentTrainer:
    def __init__(self, config: dict, model, tokenizer):
        self.config = config
        self.training_config = config['training']
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Read mask_question option
        self.mask_question = self.training_config.get('mask_question', False)
        logging.info(f"Question masking: {'ENABLED' if self.mask_question else 'DISABLED'}")
        
        output_base = resolve_with_root(config.get('output_base_dir'), OUTPUTS_DIR)
        self.output_dir = output_base / config['experiment_name']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.training_config['learning_rate'])

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
        
        self.task_type = self.training_config.get('task_type', 'completion')

    def _collate_fn(self, batch: list[dict]) -> dict:
        texts = []
        for item in batch:
            messages = []
            if self.task_type == 'qa':
                question = item.get('question')
                answer = item.get('answer')
                if question and answer:
                    messages = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
            else:
                raw_text = item.get('text')
                if raw_text:
                    messages = [{"role": "user", "content": raw_text}]

            if messages:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                if not formatted_text.endswith(self.tokenizer.eos_token):
                    formatted_text += self.tokenizer.eos_token
                texts.append(formatted_text)

        if not texts:
            logging.warning("Collate function created an empty batch. Check data format.")
            return {}

        encodings = self.tokenizer(
            texts, padding=True, truncation=True, max_length=2048,
            return_tensors="pt"
        )
        
        # Create labels
        labels = encodings['input_ids'].clone()
        
        # Mask question part if mask_question is True
        if self.mask_question and self.task_type == 'qa':
            for idx, text in enumerate(texts):
                # Find assistant header (LLaMA 3 format)
                assistant_marker = "assistant<|end_header_id|>\n\n"
                assistant_pos = text.find(assistant_marker)
                
                if assistant_pos == -1:
                    # Try alternative format
                    assistant_marker = "assistant<|end_header_id|>"
                    assistant_pos = text.find(assistant_marker)
                
                if assistant_pos != -1:
                    # Text up to the start of assistant answer
                    pre_answer_text = text[:assistant_pos + len(assistant_marker)]
                    
                    # Tokenize to check length
                    pre_answer_tokens = self.tokenizer(
                        pre_answer_text, 
                        add_special_tokens=False,
                        return_tensors="pt"
                    )
                    mask_length = len(pre_answer_tokens['input_ids'][0])
                    
                    # Mask Question part with -100
                    labels[idx, :mask_length] = -100
                    
                    # Debug log (first batch only)
                    if idx == 0 and self.mask_question:
                        logging.info(f"Masking first {mask_length} tokens for question part")
        
        # Mask Padding tokens with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encodings['input_ids'], 
            'attention_mask': encodings['attention_mask'], 
            'labels': labels
        }

    def _create_dataloader(self) -> DataLoader:
        data_path = resolve_with_root(self.training_config['data_path'])
        dataset = SimpleQADataset(data_path)
        logging.info(f"Loaded dataset from '{data_path}' with {len(dataset)} samples.")
        if len(dataset) == 0:
            logging.error("Dataset is empty. Training cannot proceed.")
            raise ValueError("Training data is empty.")
        return DataLoader(
            dataset, batch_size=self.training_config['batch_size'],
            shuffle=True, collate_fn=self._collate_fn
        )

    def _save_model(self):
        final_adapter_path = self.output_dir / "final"
        self.model.save_pretrained(final_adapter_path)
        logging.info(f"Saved final LoRA adapter to '{final_adapter_path}'.")

    def train(self):
        self.model.train()
        num_steps = self.training_config['num_train_steps']
        dataloader = self._create_dataloader()
        data_iterator = iter(dataloader)
        
        progress_bar = tqdm(range(num_steps), desc=f"Training {self.config['experiment_name']}")
        
        for step in progress_bar:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)
            
            if not batch: 
                logging.warning(f"Step {step}: Skipping empty batch.")
                continue

            if step < 2:
                logging.info(f"--- Debug Step {step} ---")
                logging.info(f"Batch keys: {batch.keys()}")
                logging.info(f"Input IDs shape: {batch['input_ids'].shape}")
                logging.info(f"Labels shape: {batch['labels'].shape}")
                if self.mask_question:
                    # Check labels of the first sample (for masking verification)
                    first_labels = batch['labels'][0]
                    masked_count = (first_labels == -100).sum().item()
                    total_count = len(first_labels)
                    logging.info(f"First sample: {masked_count}/{total_count} tokens masked")

            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs.loss
            
            if torch.isnan(loss):
                logging.error("Loss is NaN. Stopping training.")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            if (step + 1) % 100 == 0:
                logging.info(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

        self._save_model()
        logging.info("Training completed.")