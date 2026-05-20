# src/utils.py

import os
import sys
import gc
import random
import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Proceeding without wandb logging.")


def setup_logging(output_dir: str):
    """
    Configures experiment logging to output to both file and console.

    Args:
        output_dir (str): Directory where the log file will be saved.
    """
    log_filename = os.path.join(output_dir, "experiment.log")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler: Save logs to file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler: Print logs to console (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("Logging setup complete.")
    logging.info(f"All logs will be recorded to console and file: {log_filename}")


def set_seed(seed: int):
    """
    Fixes all random seeds for experiment reproducibility.

    Args:
        seed (int): Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducibility, disable CUDNN benchmark and use deterministic algorithms
        # (May cause slight performance degradation)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    logging.info(f"All random seeds fixed to {seed}.")


def setup_wandb(config: dict, output_dir: str):
    """
    Initializes and configures the WandB experiment.
    
    Args:
        config (dict): Experiment configuration dictionary
        output_dir (str): Output directory path
    """
    if not WANDB_AVAILABLE:
        logging.info("WandB not available. Skipping wandb setup.")
        return
        
    # Skip if WandB is disabled
    wandb_config = config.get('wandb', {})
    if not wandb_config.get('enabled', True):
        logging.info("WandB is disabled.")
        return
    
    try:
        # WandB project setup
        project_name = wandb_config.get('project', 'loram-experiments')
        experiment_name = config['experiment']['name']
        
        # Initialize WandB
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            dir=output_dir,
            resume="allow"
        )
        
        logging.info(f"WandB experiment initialization complete: {project_name}/{experiment_name}")
        
    except Exception as e:
        logging.warning(f"WandB initialization failed: {e}")
        logging.info("Proceeding with the experiment without WandB.")


def log_to_wandb(metrics: dict, step: int = None):
    """
    Logs metrics to WandB.
    
    Args:
        metrics (dict): Dictionary of metrics to log
        step (int, optional): Step number
    """
    if not WANDB_AVAILABLE:
        return
        
    try:
        if wandb.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    except Exception as e:
        logging.warning(f"WandB logging failed: {e}")


def finish_wandb():
    """
    Finishes the WandB experiment.
    """
    if not WANDB_AVAILABLE:
        return
        
    try:
        if wandb.run is not None:
            wandb.finish()
            logging.info("WandB experiment finished successfully.")
    except Exception as e:
        logging.warning(f"WandB finish failed: {e}")


def count_tokens(text: str, tokenizer=None) -> int:
    """
    Counts the number of tokens in the text.
    
    Args:
        text (str): Text to count tokens for
        tokenizer: Tokenizer to use (if None, estimates based on whitespace)
    
    Returns:
        int: Number of tokens
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.warning(f"Failed to count tokens with tokenizer: {e}")
    
    # Estimate based on whitespace split (approximate)
    return len(text.split())


def estimate_token_ratio(original_text: str, augmented_text: str, tokenizer=None) -> float:
    """
    Calculates the token expansion ratio of the augmented text.
    
    Args:
        original_text (str): Original text
        augmented_text (str): Augmented text  
        tokenizer: Tokenizer to use
    
    Returns:
        float: Expansion ratio (augmented / original)
    """
    original_tokens = count_tokens(original_text, tokenizer)
    augmented_tokens = count_tokens(augmented_text, tokenizer)
    
    if original_tokens == 0:
        return 0.0
    
    return augmented_tokens / original_tokens


def batch_count_tokens(texts: List[str], tokenizer=None) -> List[int]:
    """
    Counts tokens for multiple texts in a batch.
    
    Args:
        texts (List[str]): List of texts to count tokens for
        tokenizer: Tokenizer to use
    
    Returns:
        List[int]: List of token counts for each text
    """
    return [count_tokens(text, tokenizer) for text in texts]


def calculate_data_statistics(df: pd.DataFrame, text_column: str = 'text', 
                            tokenizer=None) -> Dict[str, Any]:
    """
    Calculates statistical information of the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        text_column (str): Column name containing text
        tokenizer: Tokenizer for token counting
    
    Returns:
        Dict[str, Any]: Statistical information
    """
    if text_column not in df.columns:
        logging.error(f"Column '{text_column}' not found in DataFrame")
        return {}
    
    texts = df[text_column].astype(str).tolist()
    token_counts = batch_count_tokens(texts, tokenizer)
    
    stats = {
        'total_samples': len(df),
        'total_tokens': sum(token_counts),
        'avg_tokens_per_sample': np.mean(token_counts),
        'median_tokens_per_sample': np.median(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'std_tokens': np.std(token_counts),
        'empty_samples': sum(1 for text in texts if not text.strip())
    }
    
    return stats


def clean_gpu_memory():
    """
    Cleans up GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("GPU memory cleanup complete.")


def get_model_memory_usage() -> Dict[str, float]:
    """
    Returns the current GPU memory usage of the model.
    
    Returns:
        Dict[str, float]: Memory usage info (in GB)
    """
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        
        memory_info[f'gpu_{i}'] = {
            'allocated_gb': allocated,
            'reserved_gb': reserved, 
            'total_gb': total,
            'utilization': allocated / total if total > 0 else 0
        }
    
    return memory_info