# narr/scripts/train_only.py

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from logic.utils import load_config, setup_logging, load_and_prepare_model, set_seed
from logic.trainer import ExperimentTrainer

def main(config_path: str):
    config = load_config(config_path)
    setup_logging(config)
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    logging.info(f"===== Training '{config['experiment_name']}' =====")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, tokenizer = load_and_prepare_model(config['base_model_id'], device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config['training']['lora_rank'],
        lora_alpha=config['training']['lora_alpha'],
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Execute training only
    trainer = ExperimentTrainer(config, peft_model, tokenizer)
    trainer.train()
    
    logging.info(f"===== Training Complete =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)