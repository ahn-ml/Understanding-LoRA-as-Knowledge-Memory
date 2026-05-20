#!/usr/bin/env python3
"""
Train a chunk-level LoRA adapter for multi-LoRA on QuALITY.
Each config points to a chunk JSONL; we train on the chunk text only.
"""

import argparse
import logging
import os
import sys
from peft import get_peft_model, LoraConfig
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qual.logic.utils import load_config, setup_logging, load_and_prepare_model, set_seed
from qual.logic.trainer import ExperimentTrainer


def main(config_path: str):
    config = load_config(config_path)
    setup_logging(config)
    set_seed(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, tokenizer = load_and_prepare_model(config["base_model_id"], device)

    lora_cfg = LoraConfig(
        r=config["training"]["lora_rank"],
        lora_alpha=config["training"]["lora_alpha"],
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()

    trainer = ExperimentTrainer(config, peft_model, tokenizer)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a chunk-level LoRA adapter.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
