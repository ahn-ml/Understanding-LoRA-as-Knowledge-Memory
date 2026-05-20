#!/usr/bin/env python3
"""
Prepare QuALITY articles and eval sets.
- Groups validation split by article
- Saves source text and per-article eval JSONL (question/options/answer/hard)
- Writes article_map.json for reference
"""

import os
import sys
import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATASET_ID = "emozilla/quality"
OUTPUT_ROOT = Path("qual/data")
SOURCE_DIR = OUTPUT_ROOT / "source"
EVAL_DIR = OUTPUT_ROOT / "eval"


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading QuALITY validation set...")
    ds = load_dataset(DATASET_ID, split="validation")

    articles = sorted(list(set(ds["article"])))
    logging.info("Found %d unique articles.", len(articles))

    article_map = []

    # Group rows by article text
    grouped = {}
    for row in ds:
        grouped.setdefault(row["article"], []).append(row)

    for idx, article_text in enumerate(tqdm(articles, desc="Processing articles")):
        article_map.append({"article_index": idx, "article_preview": article_text[:200] + "..."} )

        source_path = SOURCE_DIR / f"article_{idx}.txt"
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(article_text)

        eval_items = []
        for row in grouped[article_text]:
            eval_items.append(
                {
                    "question": row["question"],
                    "options": row["options"],
                    "answer": row["answer"],
                    "hard": row.get("hard", False),
                }
            )

        eval_path = EVAL_DIR / f"article_{idx}.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for item in eval_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUTPUT_ROOT / "article_map.json", "w", encoding="utf-8") as f:
        json.dump(article_map, f, ensure_ascii=False, indent=2)

    logging.info("Done. Sources in %s, eval sets in %s", SOURCE_DIR, EVAL_DIR)


if __name__ == "__main__":
    main()
