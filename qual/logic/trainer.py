import os
import json
import logging
from typing import List, Dict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler


class SimpleQADataset(Dataset):
    """Loads JSONL with either {'question','answer'} or {'text'}."""

    def __init__(self, file_path: str):
        self.data: List[Dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ExperimentTrainer:
    """LoRA trainer for QA text (generative)."""

    def __init__(self, config: dict, model, tokenizer):
        self.config = config
        self.training_config = config["training"]
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        self.mask_question = self.training_config.get("mask_question", False)
        logging.info("Question masking: %s", "ENABLED" if self.mask_question else "DISABLED")

        self.output_dir = os.path.join(
            config.get("output_base_dir", "runs/qual/experiments"),
            config["experiment_name"],
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.optimizer = AdamW(self.model.parameters(), lr=self.training_config["learning_rate"])
        num_train_steps = self.training_config["num_train_steps"]
        warmup_ratio = self.training_config.get("warmup_ratio", 0.0)
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        texts: List[str] = []
        for item in batch:
            if "question" in item and "answer" in item:
                messages = [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ]
            else:
                raw_text = item.get("text")
                messages = [{"role": "user", "content": raw_text}] if raw_text else []

            if messages:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                if not formatted.endswith(self.tokenizer.eos_token):
                    formatted += self.tokenizer.eos_token
                texts.append(formatted)

        if not texts:
            return {}

        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
        labels = enc["input_ids"].clone()

        if self.mask_question:
            for idx, text in enumerate(texts):
                marker = "assistant<|end_header_id|>\n\n"
                pos = text.find(marker)
                if pos == -1:
                    marker = "assistant<|end_header_id|>"
                    pos = text.find(marker)
                if pos != -1:
                    pre_answer = text[: pos + len(marker)]
                    tokenized = self.tokenizer(pre_answer, add_special_tokens=False, return_tensors="pt")
                    mask_len = len(tokenized["input_ids"][0])
                    labels[idx, :mask_len] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    def _create_dataloader(self) -> DataLoader:
        dataset = SimpleQADataset(self.training_config["data_path"])
        if len(dataset) == 0:
            raise ValueError("Training data is empty.")
        logging.info("Loaded %d samples from %s", len(dataset), self.training_config["data_path"])
        return DataLoader(
            dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def _save_model(self):
        final_path = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_path)
        logging.info("Saved LoRA adapter to %s", final_path)

    def train(self):
        self.model.train()
        num_steps = self.training_config["num_train_steps"]
        dataloader = self._create_dataloader()
        iterator = iter(dataloader)
        progress = tqdm(range(num_steps), desc=f"Training {self.config['experiment_name']}")

        for step in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            if not batch:
                continue

            if step < 2:
                logging.info("Debug step %d; batch keys %s", step, list(batch.keys()))

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            if torch.isnan(loss):
                logging.error("NaN loss; stopping.")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        self._save_model()
        logging.info("Training finished.")
