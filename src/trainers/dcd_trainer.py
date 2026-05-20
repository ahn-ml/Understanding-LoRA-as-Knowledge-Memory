# src/trainers/dcd_trainer.py

import os
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils import log_to_wandb
from src.models.model_loader import apply_chat_template

class SimpleDictDataset(Dataset):
    """Simple Dataset class to pass training data (list of dicts) to DataLoader."""
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class DCDTrainer:
    """
    Trainer for Deep Context Distillation (DCD/PnP).
    """
    def __init__(self, config, teacher_model, student_model, tokenizer, dataset, optimizer):
        self.config = config
        self.training_config = config['training']
        self.dcd_params = self.training_config.get('dcd_params', {
            'distillation_temp': 2.0,
            'l1_loss_weight'    : 0.1,
            'loss_mode'       : 'all',
        })
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.output_dir = config.get(
            'output_dir',
            os.path.join('runs', config['experiment']['name'])
        )

        # Device setup
        self.teacher_device = teacher_model.device
        self.student_device = student_model.device

        # Create DataLoader
        train_dataset = SimpleDictDataset(dataset)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Learning rate scheduler (Cosine Annealing)
        self.scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_config['num_train_steps'],
            eta_min=0
        )

    def _collate_fn(self, batch):
        """Tokenizes batch data to fit model inputs."""
        contexts = [item['context'] for item in batch]
        synthetic_samples = [item['synthetic_sample'] for item in batch]

        # Teacher model sees context + synthetic_sample
        teacher_inputs_text = [c + s for c, s in zip(contexts, synthetic_samples)]
        
        teacher_tokens = self.tokenizer(
            teacher_inputs_text, padding=True, truncation=True,
            max_length=4096, return_tensors="pt"
        )
        # Student model sees synthetic_sample only
        student_tokens = self.tokenizer(
            synthetic_samples, padding=True, truncation=True,
            max_length=4096, return_tensors="pt"
        )
        return teacher_tokens, student_tokens

    def _calculate_dcd_loss(self, teacher_outputs, student_outputs, student_input_ids):
        """Calculates DCD loss (KL-Divergence + L1-Regularization)."""
        student_len = student_input_ids.shape[1]
        
        loss_mode = self.dcd_params.get('loss_mode', 'all')
        
        teacher_logits = teacher_outputs.logits[:, -student_len:, :].to(self.student_device)
        student_logits = student_outputs.logits

        pad_mask = (student_input_ids != self.tokenizer.pad_token_id).float()
        temp = self.dcd_params['distillation_temp']
        
        kl_loss = torch.tensor(0.0, device=self.student_device)
        loss_l1 = torch.tensor(0.0, device=self.student_device)

        # 1. Calculate KL-Divergence Loss (Logits Distillation)
        if loss_mode in ['all', 'kl_only']:
            teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
            
            kl_loss_per_token = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
            kl_loss = (kl_loss_per_token * pad_mask).sum() / pad_mask.sum()
            kl_loss = kl_loss * (temp * temp) # Un-scale

        # 2. Calculate Hidden State L1 Loss
        if loss_mode in ['all', 'l1_only']:
            # Use only the hidden state of the last layer
            teacher_hs = teacher_outputs.hidden_states[-1][:, -student_len:, :].to(self.student_device)
            student_hs = student_outputs.hidden_states[-1]
            
            diff = torch.abs(student_hs - teacher_hs)
            masked_diff = diff * pad_mask.unsqueeze(-1)
            
            # Normalize L1 norm by the teacher hidden state's L1 norm following the paper
            l1_norm_diff = masked_diff.sum()
            teacher_norm = (torch.abs(teacher_hs) * pad_mask.unsqueeze(-1)).sum()
            loss_l1 = l1_norm_diff / (teacher_norm + 1e-8)

        # 3. Combine final loss
        if loss_mode == 'kl_only':
            total_loss = kl_loss
        elif loss_mode == 'l1_only':
            total_loss = loss_l1 * self.dcd_params['l1_loss_weight']
        else: # 'all'
            total_loss = kl_loss + loss_l1 * self.dcd_params['l1_loss_weight']
            
        return total_loss, kl_loss.item(), loss_l1.item()

    def _save_checkpoint(self, step_or_name: str):
        """Saves the LoRA weights of the student model."""
        checkpoint_dir = os.path.join(self.output_dir, str(step_or_name))
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.student_model.save_pretrained(checkpoint_dir)
        logging.info(f"Checkpoint saved: {checkpoint_dir}")

    def train(self):
        """Executes the main training loop."""
        self.student_model.train()
        num_steps = self.training_config['num_train_steps']
        strategy_cfg = self.training_config.get('checkpoint_strategy',
                                                {'policy': 'none', 'value': 0})

        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(range(num_steps), desc="DCD Training")

        for step in progress_bar:
            try:
                teacher_tokens, student_tokens = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                teacher_tokens, student_tokens = next(data_iterator)

            teacher_tokens = {k: v.to(self.teacher_device) for k, v in teacher_tokens.items()}
            student_tokens = {k: v.to(self.student_device) for k, v in student_tokens.items()}

            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_tokens, output_hidden_states=True)

            student_outputs = self.student_model(**student_tokens, output_hidden_states=True)

            loss, kl_loss, l1_loss = self._calculate_dcd_loss(
                teacher_outputs, student_outputs, student_tokens['input_ids']
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # WandB logging
            wandb_metrics = {
                "train/loss": loss.item(),
                "train/kl_loss": kl_loss,
                "train/l1_loss": l1_loss,
                "train/learning_rate": self.scheduler.get_last_lr()[0],
                "train/step": step + 1
            }
            log_to_wandb(wandb_metrics, step=step + 1)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "KL": f"{kl_loss:.4f}", 
                "L1": f"{l1_loss:.4f}"
            })

            # Save intermediate checkpoint
            if strategy_cfg.get('policy') == "steps" and strategy_cfg.get('value', 0) > 0 and (step + 1) % strategy_cfg['value'] == 0:
                self._save_checkpoint(f"step_{step+1}")
        
        # Save final model
        self._save_checkpoint("final")
        logging.info("Final LoRA saved.")
