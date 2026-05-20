#!/usr/bin/env python3
# train.py
"""
Script to perform single LoRA training based on YAML configuration file.
Called by run_experiment.py.
"""

import os
import yaml
import json
import argparse
import logging
import time
import torch
import sys
from pathlib import Path
from peft import get_peft_model, LoraConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import internal project modules
from src.datasets import get_dataset_loader
from src.trainers.ntp_trainer import NTPTrainer
from src.trainers.dcd_trainer import DCDTrainer
from src.models.llama_loader import load_llama_model_and_tokenizer
from src.models.qwen_loader import load_qwen_with_lora
from src.models import load_dcd_models, apply_lora_and_get_counts

def train(config_path: str):
    """Main function for training"""
    # 1. Load configuration and prepare environment
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    exp_name = config['experiment']['name']
    exp_group = os.path.basename(os.path.dirname(config_path))
    output_dir = config.get('output_dir')
    if not output_dir:
        if config.get('output_base_dir'):
            output_dir = os.path.join(config['output_base_dir'], exp_name)
        else:
            output_dir = os.path.join('runs', 'pb_cf', exp_group, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Add file handler (logging is already set up in run_experiment.py)
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"===== [{exp_name}] Training Start =====")

    training_cfg = config.get('training', {})
    training_method = training_cfg.get('method', 'ntp')

    # 2. Load dataset
    dataset_loader = get_dataset_loader(config)
    train_dataset = dataset_loader.load_data()
    if not train_dataset:
        logging.error("No training data found. Exiting.")
        return

    # 3. Start training
    training_start_time = time.time()
    config['output_dir'] = output_dir
    model_config = config.get('model', {})
    base_model_id = model_config.get('base_model_id')
    summary_data = {
        "experiment_name": exp_name,
        "base_model_id": base_model_id,
        "training_method": training_method,
    }

    if training_method in {"pnp_dcd", "dcd"}:
        teacher_model, student_base_model, tokenizer = load_dcd_models(config)
        student_model, _, trainable_params = apply_lora_and_get_counts(student_base_model, config)
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=training_cfg['learning_rate'])

        trainer = DCDTrainer(
            config=config,
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer
        )
        trainer.train()

        training_duration_seconds = time.time() - training_start_time
        logging.info(f"Total training time: {training_duration_seconds:.2f} seconds")

        summary_data.update({
            "training_duration_seconds": training_duration_seconds,
            "trainable_parameters": trainable_params,
            "lora_config": model_config.get('lora_config', {}),
        })
    else:
        dataset_config = config.get('dataset', {})
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if "Llama-3.1" in base_model_id or "Llama-3.2" in base_model_id:
            use_template = dataset_config.get('use_chat_template', False)
            model, tokenizer = load_llama_model_and_tokenizer(base_model_id, device, use_template)
        elif "Qwen" in base_model_id:
            model, tokenizer = load_qwen_with_lora(config)
        else:
            raise ValueError(f"Unsupported model ID: {base_model_id}")

        peft_config_dict = model_config.get('lora_config', {})
        peft_config = LoraConfig(**peft_config_dict)
        
        
        
        model = get_peft_model(model, peft_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Trainable Parameters: {trainable_params}")

        trainer = NTPTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            dataset=train_dataset
        )
        trainer.train()

        training_duration_seconds = time.time() - training_start_time
        logging.info(f"Total training time: {training_duration_seconds:.2f} seconds")

        summary_data.update({
            "training_duration_seconds": training_duration_seconds,
            "trainable_parameters": trainable_params,
            "lora_config": peft_config_dict,
        })

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    logging.info(f"Training summary saved to '{summary_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Model Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file for training")
    args = parser.parse_args()
    train(args.config)
