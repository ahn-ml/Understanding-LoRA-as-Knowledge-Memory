#!/usr/bin/env python3
# paperqa/scripts/02b_generate_multilora_configs.py

import os
import re
import yaml
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Basic Experiment Settings ---
EXPERIMENT_GROUP_NAME = "paperqa_multilora"
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
    r: 4
    lora_alpha: 8
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

def get_available_individual_files():
    """Finds individual files in the paperqa/data directory."""
    data_files = []
    
    if not os.path.exists(DATA_ROOT_DIR):
        logging.error(f"DATA_ROOT_DIR '{DATA_ROOT_DIR}' does not exist.")
        return data_files
    
    # Individual directories (individual_* directories)
    individual_dirs = [
        "individual_introductions_bracket",
        "individual_introductions_natural", 
        "individual_qa_bracket",
        "individual_qa_natural"
    ]
    
    for dir_name in individual_dirs:
        dir_path = os.path.join(DATA_ROOT_DIR, dir_name)
        if os.path.exists(dir_path):
            logging.info(f"Checking directory: {dir_name}")
            for filename in os.listdir(dir_path):
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(dir_path, filename)
                    paper_id = extract_paper_id(filename)
                    logging.info(f"Processing file: {filename} -> paper_id: {paper_id}")
                    data_files.append({
                        'type': 'individual',
                        'category': dir_name,
                        'filename': filename,
                        'filepath': file_path,
                        'paper_id': paper_id
                    })
    
    return data_files

def extract_paper_id(filename):
    """Extracts paper ID from the filename."""
    # Extract paper ID from patterns like paper_0_introduction.jsonl, paper_1_qa.jsonl
    match = re.match(r'paper_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def group_files_by_paper_and_category(data_files):
    """Groups files by paper ID and data category."""
    file_groups = {}
    
    for file_info in data_files:
        paper_id = file_info['paper_id']
        category = file_info['category']
        key = f"paper_{paper_id}_{category}"
        
        if key not in file_groups:
            file_groups[key] = file_info
    
    return file_groups

def generate_experiment_name(paper_id, category, lora_rank=16):
    """Generates an experiment name."""
    # Remove "individual_" prefix from category
    clean_category = category.replace("individual_", "")
    return f"{EXPERIMENT_GROUP_NAME}_paper{paper_id}_{clean_category}_r{lora_rank}"

def generate_multilora_configs(lora_ranks=[8, 16, 32]):
    """Generates YAML configuration files for Multi-LoRA training."""
    
    
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Get available individual files
    data_files = get_available_individual_files()
    
    if not data_files:
        logging.error("No available individual data files found.")
        return
    
    # Group by paper ID and data type
    file_groups = group_files_by_paper_and_category(data_files)
    
    logging.info(f"Number of file groups found: {len(file_groups)}")
    logging.info(f"File groups: {list(file_groups.keys())}")
    
    count = 0
    total_configs = len(file_groups) * len(lora_ranks)
    logging.info(f"Generating total {total_configs} config files in '{CONFIG_OUTPUT_DIR}'...")
    
    for group_key, file_info in file_groups.items():
        paper_id = file_info['paper_id']
        category = file_info['category']
        
        for lora_rank in lora_ranks:
            # Generate experiment name
            exp_name = generate_experiment_name(paper_id, category, lora_rank)
            
            # Data file path (absolute path)
            data_file_path = os.path.abspath(file_info['filepath'])
            
            # Format template with LoRA rank
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
            logging.info(f"Generated: {exp_name}.yaml (paper {paper_id}, {category})")
    
    logging.info(f"Total {count} configuration files generated.")

def main():
    # Update global variables
    global EXPERIMENT_GROUP_NAME, CONFIG_OUTPUT_DIR

    parser = argparse.ArgumentParser(description='Generate PaperQA Multi-LoRA training config files (using only individual files)')
    parser.add_argument('--lora-ranks', nargs='+', type=int, default=[16],
                        help='LoRA rank values (default: 16)')
    parser.add_argument('--experiment-group', type=str, default=EXPERIMENT_GROUP_NAME,
                        help='Experiment group name (default: paperqa_multilora)')
    
    args = parser.parse_args()
    
    EXPERIMENT_GROUP_NAME = args.experiment_group
    CONFIG_OUTPUT_DIR = PROJECT_ROOT / "configs" / "generated" / EXPERIMENT_GROUP_NAME
    
    logging.info(f"Experiment Group: {EXPERIMENT_GROUP_NAME}")
    logging.info(f"LoRA ranks: {args.lora_ranks}")
    logging.info("Data Type: Using individual files only")
    
    generate_multilora_configs(args.lora_ranks)

if __name__ == "__main__":
    main()
