# narr/scripts/02_run_experiment.py

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from logic.utils import load_config, setup_logging, load_and_prepare_model
from logic.trainer import ExperimentTrainer
from logic.evaluation import evaluate_model

def main(config_path: str):
    # Load configuration and setup logging
    config = load_config(config_path)
    setup_logging(config)
    
    logging.info(f"===== Starting Experiment '{config['experiment_name']}' =====")
    logging.info(f"Config file: {config_path}")
    
    # --- Training Phase ---
    logging.info("--- Starting Training Phase ---")
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, tokenizer = load_and_prepare_model(config['base_model_id'], device)
    
    # Apply LoRA configuration
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
    
    # Initialize and run trainer
    trainer = ExperimentTrainer(config, peft_model, tokenizer)
    trainer.train()
    
    # --- Evaluation Phase ---
    logging.info("--- Starting Evaluation Phase ---")
    
    # Run evaluation (function reloads the model internally)
    evaluate_model(config)
    
    logging.info(f"===== Experiment '{config['experiment_name']}' Completed =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single document LoRA experiment")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"), help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)