# logic/utils.py

import os
import yaml
import logging
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AzureOpenAI

from path_utils import LOGS_DIR


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config: dict):
    """Setup logging for the experiment (console and file)."""
    exp_name = config['experiment_name']
    log_dir = config.get('log_dir', str(LOGS_DIR))
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"{exp_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    logging.info(f"Logging setup complete. Logs saved to '{log_filename}'.")

def get_azure_client(config: dict) -> AzureOpenAI:
    """Initialize and return Azure OpenAI client."""
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        raise ValueError("Please set environment variables: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT")
    
    api_version = config.get('api_config', {}).get('api_version', '2024-10-21')
        
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key
    )

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_and_prepare_model(model_id: str, device: str = "cuda"):
    """Load base model and tokenizer with default settings."""
    logging.info(f"Loading model and tokenizer: '{model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("tokenizer.pad_token missing, set to eos_token.")
        
    return model, tokenizer