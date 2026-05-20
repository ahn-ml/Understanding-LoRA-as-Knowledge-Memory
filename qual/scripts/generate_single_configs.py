#!/usr/bin/env python3
"""
Generate single-LoRA configs for all QuALITY articles across four models.
"""

import os
import sys
import json
import logging
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TRAIN_DIR_ARTICLE = Path("qual/data/train")
TRAIN_DIR_CHUNK = Path("qual/data/train_chunkqa")
EVAL_DIR = Path("qual/data/eval")
CONFIG_ROOT = Path("configs/generated/qual/single")

# Fixed HP (match narr style)
HP = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 5.0e-4,
    "num_train_steps": 250,
    "batch_size": 32,
    "warmup_ratio": 0.1,
}

MODELS = {
    "qwen1p7": os.environ.get("LORAM_QWEN1P7_MODEL", "Qwen/Qwen3-1.7B"),
    "qwen8b": os.environ.get("LORAM_QWEN8B_MODEL", "Qwen/Qwen3-8B"),
    "llama8b": os.environ.get("LORAM_LLAMA8B_MODEL", os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")),
    "llama1b": os.environ.get("LORAM_LLAMA1B_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
}


def build_config(exp_name: str, model_id: str, train_path: str, eval_path: str) -> dict:
    return {
        "experiment_name": exp_name,
        "base_model_id": model_id,
        "output_base_dir": f"runs/qual/experiments/{exp_name.split('__')[0]}",
        "log_dir": "runs/qual/logs/experiments",
        "seed": 42,
        "training": {
            "data_path": train_path,
            "task_type": "qa",
            "lora_rank": HP["lora_rank"],
            "lora_alpha": HP["lora_alpha"],
            "learning_rate": HP["learning_rate"],
            "num_train_steps": HP["num_train_steps"],
            "batch_size": HP["batch_size"],
            "warmup_ratio": HP["warmup_ratio"],
            "mask_question": False,
        },
        "evaluation": {
            "eval_data_path": eval_path,
            "max_new_tokens": 8,
        },
    }


def pick_train_file(idx: int):
    chunk_path = TRAIN_DIR_CHUNK / f"article_{idx}.jsonl"
    if chunk_path.exists():
        return chunk_path
    art_path = TRAIN_DIR_ARTICLE / f"article_{idx}.jsonl"
    if art_path.exists():
        return art_path
    return None


def main():
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    train_sources = sorted(TRAIN_DIR_CHUNK.glob("article_*.jsonl")) or sorted(TRAIN_DIR_ARTICLE.glob("article_*.jsonl"))
    if not train_sources:
        logging.error("No train files found. Run 04_build_train_sets.py first.")
        return

    total = 0
    for tag, model_id in MODELS.items():
        model_dir = CONFIG_ROOT / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        for path in train_sources:
            idx = int(path.stem.split("_")[1])
            train_path = pick_train_file(idx)
            if not train_path:
                continue
            eval_path = EVAL_DIR / f"article_{idx}.jsonl"
            if not eval_path.exists():
                logging.warning("Missing eval for article %d; skipping.", idx)
                continue
            exp_name = f"{tag}__article{idx}"
            cfg = build_config(exp_name, model_id, str(train_path), str(eval_path))
            out_path = model_dir / f"{exp_name}.yaml"
            with open(out_path, "w", encoding="utf-8") as f:
                import yaml

                yaml.dump(cfg, f, sort_keys=False, indent=2, allow_unicode=True)
            total += 1

    logging.info("Generated %d configs under %s", total, CONFIG_ROOT)


if __name__ == "__main__":
    main()
