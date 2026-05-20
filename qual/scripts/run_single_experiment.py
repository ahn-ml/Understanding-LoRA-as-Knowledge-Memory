#!/usr/bin/env python3
"""
Train + evaluate a single LoRA experiment for QuALITY MCQ.
"""

import argparse
import logging
import os
import sys
from peft import get_peft_model, LoraConfig, PeftModel
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qual.logic.utils import load_config, setup_logging, load_and_prepare_model, set_seed
from qual.logic.trainer import ExperimentTrainer
from qual.logic.evaluation import MCQEvaluator, load_eval_data


def main(config_path: str):
    config = load_config(config_path)
    setup_logging(config)
    set_seed(config.get("seed", 42))

    logging.info("=== Starting experiment: %s ===", config["experiment_name"])

    # Train
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

    # Evaluate (reload base + adapter for clean state)
    logging.info("=== Evaluation ===")
    eval_data = load_eval_data(config["evaluation"]["eval_data_path"])
    base_model_eval, tokenizer_eval = load_and_prepare_model(config["base_model_id"], device)
    adapter_path = os.path.join(
        config.get("output_base_dir", "runs/qual/experiments"),
        config["experiment_name"],
        "final",
    )
    model_eval = PeftModel.from_pretrained(base_model_eval, adapter_path)
    evaluator = MCQEvaluator(model_eval, tokenizer_eval, device=device)
    results = evaluator.evaluate(eval_data)

    summary_path = os.path.join(
        config.get("output_base_dir", "runs/qual/experiments"),
        config["experiment_name"],
        "evaluation_results.json",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    import json

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Accuracy: %.4f (hard: %.4f)", results["accuracy"], results["hard_accuracy"])
    logging.info("Saved eval to %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single QuALITY LoRA experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()
    main(args.config)
