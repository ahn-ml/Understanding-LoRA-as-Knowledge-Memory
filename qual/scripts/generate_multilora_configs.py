#!/usr/bin/env python3
"""
Generate chunk-level multi-LoRA configs for all QuALITY articles.
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

CHUNK_DIR = Path("qual/data/chunks")
CHUNK_QA_DIR = Path("qual/data/chunk_qa")
CONFIG_ROOT = Path("configs/generated/qual/multi_lora")

MODELS = {
    "qwen1p7": os.environ.get("LORAM_QWEN1P7_MODEL", "Qwen/Qwen3-1.7B"),
    "qwen8b": os.environ.get("LORAM_QWEN8B_MODEL", "Qwen/Qwen3-8B"),
    "llama8b": os.environ.get("LORAM_LLAMA8B_MODEL", os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")),
    "llama1b": os.environ.get("LORAM_LLAMA1B_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
}

HP = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 5.0e-4,
    "num_train_steps": 250,
    "batch_size": 8,
    "warmup_ratio": 0.1,
}


def main():
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    chunk_folders = sorted(CHUNK_DIR.glob("article_*"))
    total = 0
    for tag, model_id in MODELS.items():
        model_dir = CONFIG_ROOT / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        for folder in chunk_folders:
            idx = int(folder.name.split("_")[1])
            for chunk_file in folder.glob("chunks_*.jsonl"):
                size = int(chunk_file.stem.split("_")[1])
                # we will train one adapter per chunk line; store input path and chunk id
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        cid = rec["chunk_id"]
                        exp_name = f"{tag}__article{idx}_s{size}_c{cid}"
                        qa_path = CHUNK_QA_DIR / f"article_{idx}" / f"chunk_{size}_{cid}_qa.jsonl"
                        if not qa_path.exists() or qa_path.stat().st_size == 0:
                            logging.debug("Skip %s (missing QA %s)", exp_name, qa_path)
                            continue
                        cfg = {
                            "experiment_name": exp_name,
                            "base_model_id": model_id,
                            "output_base_dir": f"runs/qual/{tag}/multi_lora/training",
                            "log_dir": "runs/qual/logs/multi_lora",
                            "seed": 42,
                            "training": {
                                "data_path": str(qa_path),
                                "chunk_id": cid,
                                "chunk_size": size,
                                "batch_size": HP["batch_size"],
                                "lora_rank": HP["lora_rank"],
                                "lora_alpha": HP["lora_alpha"],
                                "learning_rate": HP["learning_rate"],
                                "num_train_steps": HP["num_train_steps"],
                                "warmup_ratio": HP["warmup_ratio"],
                            },
                        }
                        out_path = model_dir / f"{exp_name}.yaml"
                        import yaml

                        with open(out_path, "w", encoding="utf-8") as f_out:
                            yaml.dump(cfg, f_out, sort_keys=False, indent=2, allow_unicode=True)
                        total += 1
    logging.info("Generated %d multi-LoRA configs under %s", total, CONFIG_ROOT)


if __name__ == "__main__":
    main()
