#!/usr/bin/env python3
# paperqa/run_experiment.py

"""
PaperQA Experiment Execution Script
Can run a single experiment or batch execute multiple experiments.
"""

import argparse
import logging
import os
import sys
import yaml
import glob
from pathlib import Path

# Add current directory to sys.path
PAPERQA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PAPERQA_DIR)

from logic.trainer import PaperQATrainer
from logic.utils import (
    load_config, 
    setup_logging, 
    load_and_prepare_model, 
    apply_lora_config,
    validate_config,
    get_device
)
from peft import get_peft_model, LoraConfig

def run_single_experiment(config_path: str):
    """Runs a single experiment."""
    # 1. Load and validate configuration
    config = load_config(config_path)
    validate_config(config)
    
    # 2. Setup logging
    output_dir = setup_logging(config)
    config['output_dir'] = output_dir
    
    exp_name = config['experiment']['name']
    logging.info(f"===== PaperQA Experiment '{exp_name}' Started =====")
    logging.info(f"Config file: {config_path}")
    
    # 3. Setup device
    device = get_device()
    
    # 4. Load model and tokenizer
    base_model, tokenizer = load_and_prepare_model(config['model']['base_model_id'], device)
    
    # 5. Apply LoRA configuration
    # 
    lora_config = LoraConfig(**config['model']['lora_config'])
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    
    # 6. Run trainer
    trainer = PaperQATrainer(config, peft_model, tokenizer)
    trainer.train()
    
    logging.info(f"===== PaperQA Experiment '{exp_name}' Completed =====")
    return output_dir

def run_batch_experiments(config_dir: str, pattern: str = "*.yaml"):
    """Runs batch experiments using multiple config files in a directory."""
    config_files = glob.glob(os.path.join(config_dir, pattern))
    
    if not config_files:
        logging.error(f"No config files found in {config_dir} with pattern {pattern}")
        return
    
    logging.info(f"Found {len(config_files)} config files to run")
    
    results = []
    for config_file in config_files:
        try:
            logging.info(f"Running experiment with config: {config_file}")
            output_dir = run_single_experiment(config_file)
            results.append({
                'config_file': config_file,
                'output_dir': output_dir,
                'status': 'success'
            })
        except Exception as e:
            logging.error(f"Failed to run experiment with {config_file}: {e}")
            results.append({
                'config_file': config_file,
                'output_dir': None,
                'status': 'failed',
                'error': str(e)
            })
    
    # Result summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    logging.info(f"Batch experiment completed: {successful} successful, {failed} failed")
    
    return results

def list_available_configs(config_dir: str):
    """Lists available configuration files."""
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    
    if not config_files:
        print(f"No YAML config files found in {config_dir}")
        return
    
    print(f"Available config files in {config_dir}:")
    for config_file in sorted(config_files):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                exp_name = config.get('experiment', {}).get('name', 'Unknown')
                print(f"  - {os.path.basename(config_file)}: {exp_name}")
        except Exception as e:
            print(f"  - {os.path.basename(config_file)}: Error reading config ({e})")

def main():
    parser = argparse.ArgumentParser(description="PaperQA Experiment Execution Script")
    parser.add_argument("--config", type=str, help="Path to YAML config file for single experiment")
    parser.add_argument("--config-dir", type=str, help="Directory of config files for batch experiments")
    parser.add_argument("--pattern", type=str, default="*.yaml", help="File pattern to use for batch experiments")
    parser.add_argument("--list", action="store_true", help="List available config files")
    
    args = parser.parse_args()
    
    if args.list:
        if args.config_dir:
            list_available_configs(args.config_dir)
        else:
            # Check default config directories
            default_dirs = [
                "configs/generated/paperqa_singlelora",
                "configs/generated/paperqa_multilora"
            ]
            for config_dir in default_dirs:
                if os.path.exists(config_dir):
                    list_available_configs(config_dir)
                    print()
    elif args.config:
        # Run single experiment
        run_single_experiment(args.config)
    elif args.config_dir:
        # Run batch experiments
        run_batch_experiments(args.config_dir, args.pattern)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
