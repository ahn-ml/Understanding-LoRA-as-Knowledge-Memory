# paperqa/scripts/generate_single_lora_configs.py

import os
import re
import yaml
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Basic Experiment Settings ---
EXPERIMENT_GROUP_NAME = "paperqa_singlelora"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DIR = PROJECT_ROOT / "paperqa" / "data"
CONFIG_OUTPUT_DIR = PROJECT_ROOT / "configs" / "generated" / EXPERIMENT_GROUP_NAME

CONFIG_TEMPLATE = """
experiment:
  name: "{exp_name}"
  group: "{exp_group}"
  seed: 42

resources:
  gpus: [0]

output_base_dir: "runs/paperqa"

model:
  base_model_id: "{base_model_id}"
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: "all-linear"
    lora_dropout: 0.1
    bias: "none"
    task_type: "CAUSAL_LM"

dataset:
  name: "paperqa"
  use_chat_template: true

training:
  data_path: "{data_file_path}"
  method: "ntp"
  num_train_steps: 1000
  learning_rate: 5.0e-05
  warmup_ratio: 0.1
  batch_size: 8
  checkpoint_strategy:
    policy: "none"
"""

def get_available_data_files():
    """Finds concatenated files in the paperqa/data directory."""
    data_files = []
    
    if not os.path.exists(DATA_ROOT_DIR):
        logging.error(f"DATA_ROOT_DIR '{DATA_ROOT_DIR}' does not exist.")
        return data_files
    
    # Use only concatenated files
    concatenated_files = [
        "concatenated_introductions_bracket.jsonl",
        "concatenated_introductions_natural.jsonl",
        "concatenated_qa_bracket.jsonl", 
        "concatenated_qa_natural.jsonl"
    ]
    
    for filename in concatenated_files:
        file_path = os.path.join(DATA_ROOT_DIR, filename)
        if os.path.exists(file_path):
            data_files.append({
                'type': 'concatenated',
                'category': filename.replace('.jsonl', ''),
                'filename': filename,
                'filepath': file_path,
                'paper_id': 'all'  # Concatenated files contain all papers
            })
    
    return data_files

def generate_experiment_name(data_file_info, lora_rank=16):
    """Generates an experiment name."""
    category = data_file_info['category']
    return f"{EXPERIMENT_GROUP_NAME}_{category}_r{lora_rank}"

def generate_configs(lora_ranks=[8, 16, 32]):
    """Generates YAML configuration files for Single LoRA training."""
    
    
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Get available data files (concatenated only)
    data_files = get_available_data_files()
    
    if not data_files:
        logging.error("No available concatenated data files found.")
        return
    
    count = 0
    total_configs = len(data_files) * len(lora_ranks)
    logging.info(f"Generating total {total_configs} config files in '{CONFIG_OUTPUT_DIR}'...")
    
    for data_file_info in data_files:
        for lora_rank in lora_ranks:
            # Generate experiment name
            exp_name = generate_experiment_name(data_file_info, lora_rank)
            
            # Data file path (absolute path)
            data_file_path = os.path.abspath(data_file_info['filepath'])
            
            # Apply LoRA rank to template
            config_str = CONFIG_TEMPLATE.format(
                exp_name=exp_name,
                exp_group=EXPERIMENT_GROUP_NAME,
                base_model_id=os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                data_file_path=data_file_path
            )
            
            # Update LoRA rank settings
            config_dict = yaml.safe_load(config_str)
            config_dict['model']['lora_config']['r'] = lora_rank
            config_dict['model']['lora_config']['lora_alpha'] = lora_rank  # Set alpha = rank
            
            # Save to file
            filepath = os.path.join(CONFIG_OUTPUT_DIR, f"{exp_name}.yaml")
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            count += 1
            logging.info(f"Generated: {exp_name}.yaml")
    
    logging.info(f"Total {count} configuration files generated.")

def main():
    # Update global variables
    global EXPERIMENT_GROUP_NAME, CONFIG_OUTPUT_DIR

    parser = argparse.ArgumentParser(description='Generate PaperQA Single LoRA training config files (concatenated files only)')
    parser.add_argument('--lora-ranks', nargs='+', type=int, default=[16],
                        help='LoRA rank values (default: 16)')
    parser.add_argument('--experiment-group', type=str, default=EXPERIMENT_GROUP_NAME,
                        help='Experiment group name (default: paperqa_single_lora)')
    
    args = parser.parse_args()
    
    EXPERIMENT_GROUP_NAME = args.experiment_group
    CONFIG_OUTPUT_DIR = PROJECT_ROOT / "configs" / "generated" / EXPERIMENT_GROUP_NAME
    
    logging.info(f"Experiment Group: {EXPERIMENT_GROUP_NAME}")
    logging.info(f"LoRA ranks: {args.lora_ranks}")
    logging.info("Data Type: Using concatenated files only")
    
    generate_configs(args.lora_ranks)

if __name__ == "__main__":
    main()
