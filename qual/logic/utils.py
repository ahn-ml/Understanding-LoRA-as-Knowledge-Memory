import os
import yaml
import logging
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Configure console + file logging."""
    exp_name = config["experiment_name"]
    log_dir = config.get("log_dir", "qual/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logging.info("Logging to %s", log_path)


def get_openrouter_client(config: dict | None = None) -> OpenAI:
    """Return an OpenRouter client. Requires OPENROUTER_API_KEY."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in the environment.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def get_openrouter_headers() -> dict:
    """Optional OpenRouter headers for rankings."""
    headers = {}
    site_url = os.environ.get("OPENROUTER_SITE_URL")
    app_name = os.environ.get("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_model(model_id: str, device: str = "cuda"):
    """Load base model/tokenizer and set pad_token if missing."""
    logging.info("Loading model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("pad_token missing; set to eos_token")
    return model, tokenizer
