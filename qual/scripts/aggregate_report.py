#!/usr/bin/env python3
"""
Aggregate QuALITY MCQ accuracies for 12 combos across four models.
Outputs JSON + prints table.
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_TAGS = ["qwen1p7", "qwen8b", "llama8b", "llama1b"]
COMBOS = [
    ("base", "base"),
    ("base_icl", "base"),
    ("base_rag", "base"),
    ("slora_none", "single"),
    ("slora_icl", "single"),
    ("slora_rag", "single"),
    ("mlora1_none", "multi"),
    ("mlora1_icl", "multi"),
    ("mlora1_rag", "multi"),
    ("mlora3_none", "multi"),
    ("mlora3_icl", "multi"),
    ("mlora3_rag", "multi"),
]

# Baseline outputs for older runs were stored under full model names.
BASELINE_FOLDERS = {
    "qwen1p7": "Qwen3-1.7B",
    "qwen8b": "Qwen3-8B",
    "llama8b": "Llama-3.1-8B-Instruct",
    "llama1b": "Llama-3.2-1B-Instruct",
}


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_for_model(tag: str):
    rows = defaultdict(list)
    baseline_root_candidates = [
        Path(f"runs/qual/{tag}/baselines"),
        Path(f"runs/qual/{BASELINE_FOLDERS.get(tag, tag)}/baselines"),
    ]
    def baseline_root_exists(p: Path) -> bool:
        return p.exists() and any(p.iterdir())
    baseline_root = next((p for p in baseline_root_candidates if baseline_root_exists(p)), None)

    # base / icl / rag
    if baseline_root:
        for mode in ["base", "icl", "rag"]:
            for art in range(115):
                path = baseline_root / mode / f"article_{art}" / "summary.json"
                data = load_json(path)
                if data and "accuracy" in data:
                    # store under keys the rest of the code expects
                    key = "base" if mode == "base" else f"base_{mode}"
                    rows[key].append(data["accuracy"])
    # single none
    exp_dir = Path(f"runs/qual/experiments/{tag}")
    if exp_dir.exists():
        for sub in exp_dir.iterdir():
            res = sub / "evaluation_results.json"
            data = load_json(res)
            if data and "accuracy" in data:
                rows["slora_none"].append(data["accuracy"])
    # single icl/rag
    for art in range(115):
        for ctx in ["icl", "rag"]:
            path = Path(f"runs/qual/{tag}/single_eval/article_{art}/single_{ctx}.json")
            data = load_json(path)
            if data and "accuracy" in data:
                rows[f"slora_{ctx}"].append(data["accuracy"])
    # multi
    for art in range(115):
        for k in [1, 3]:
            for ctx in ["none", "icl", "rag"]:
                path = Path(f"runs/qual/{tag}/multi_lora/eval/article_{art}/mlora{k}_{ctx}.json")
                data = load_json(path)
                if data and "accuracy" in data:
                    rows[f"mlora{k}_{ctx}"].append(data["accuracy"])
    # compute averages
    summary = {}
    for combo, _ in COMBOS:
        vals = rows.get(combo, [])
        summary[combo] = sum(vals) / len(vals) if vals else 0.0
    return summary


def main():
    summary_all = {}
    for tag in MODEL_TAGS:
        summary_all[tag] = collect_for_model(tag)

    out_path = Path("runs/qual/report_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)

    # print table
    headers = ["model"] + [c for c, _ in COMBOS]
    print("\t".join(headers))
    for tag in MODEL_TAGS:
        row = [tag] + [f"{summary_all[tag].get(c, 0):.4f}" for c, _ in COMBOS]
        print("\t".join(row))
    logging.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()
