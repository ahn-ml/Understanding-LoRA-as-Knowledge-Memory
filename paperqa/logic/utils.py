# paperqa/logic/utils.py

"""
PaperQA Utility Functions
Provides common functionality such as configuration loading, logging setup, and model loading.
"""

import os
import yaml
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Add config_path to config
    config['config_path'] = config_path
    
    return config

def setup_logging(config: dict) -> str:
    """Sets up logging and returns output directory."""
    exp_name = config['experiment']['name']
    exp_group = config['experiment']['group']
    
    # Create output directory
    output_base_dir = config.get('output_base_dir', 'runs/paperqa')
    output_dir = os.path.join(output_base_dir, exp_group, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set log file
    log_file = os.path.join(output_dir, "training.log")
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return output_dir

def load_and_prepare_model(base_model_id: str, device: str):
    """Loads base model and tokenizer."""
    
    logging.info(f"Loading model: {base_model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Set padding token (for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        device_map=device if device.startswith('cuda') else None,
        trust_remote_code=True
    )
    
    # Move to GPU (if CPU)
    if not device.startswith('cuda'):
        model = model.to(device)
    
    logging.info(f"Model loaded successfully on {device}")
    return model, tokenizer

def apply_lora_config(model, lora_config_dict: dict):
    """Applies LoRA configuration."""
    lora_config = LoraConfig(**lora_config_dict)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model

def save_training_summary(config: dict, training_info: dict, output_dir: str):
    """Saves training summary information."""
    summary_data = {
        "experiment_name": config['experiment']['name'],
        "base_model_id": config['model']['base_model_id'],
        "config_path": config.get('config_path', ''),
        "output_dir": output_dir,
        **training_info
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary_data, f, indent=2, allow_unicode=True, default_flow_style=False)
    
    logging.info(f"Training summary saved to: {summary_path}")
    return summary_path

def get_device():
    """Returns available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logging.info("CUDA not available. Using CPU.")
    
    return device

def validate_config(config: dict):
    """Validates required fields in the configuration file."""
    required_sections = ['experiment', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    required_experiment = ['name']
    for field in required_experiment:
        if field not in config['experiment']:
            raise ValueError(f"Missing required field in experiment: {field}")
    
    required_model = ['base_model_id', 'lora_config']
    for field in required_model:
        if field not in config['model']:
            raise ValueError(f"Missing required field in model: {field}")
    
    required_training = ['batch_size', 'learning_rate', 'num_train_steps']
    for field in required_training:
        if field not in config['training']:
            raise ValueError(f"Missing required field in training: {field}")
    
    logging.info("Configuration validation passed")

def detect_data_type_from_file(file_path: str) -> str:
    """Detects data type based on file path and content."""
    # Determine by filename
    if 'qa' in file_path.lower():
        return 'qa'
    elif 'introduction' in file_path.lower():
        return 'introduction'
    
    # Determine by first sample
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            
            if 'question' in sample and 'answer' in sample:
                return 'qa'
            elif 'text' in sample:
                return 'introduction'
    except Exception as e:
        logging.warning(f"Could not detect data type from file: {e}")
    
    # Default value
    return 'introduction'

def format_training_time(seconds: float) -> str:
    """Formats training time into human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
