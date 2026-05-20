#!/usr/bin/env python3
"""
Build training JSONL per article.
- mode=article: use article-level QA (qual/data/qa)
- mode=chunk: use chunk-level QA (qual/data/chunk_qa) for a given chunk_size and merge all chunks
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QA_DIR = Path("qual/data/qa")
SUMMARY_DIR = Path("qual/data/summaries")
CHUNK_QA_DIR = Path("qual/data/chunk_qa")
TRAIN_ARTICLE_DIR = Path("qual/data/train")
TRAIN_CHUNK_DIR = Path("qual/data/train_chunkqa")


def build_article(args):
    TRAIN_ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
    qa_files = sorted(QA_DIR.glob("article_*_qa.jsonl"))
    targets = [f for f in qa_files if args.start <= int(f.stem.split("_")[1]) < args.end]
    logging.info("Building ARTICLE-level train files for %d articles (%d-%d)", len(targets), args.start, args.end - 1)
    for path in tqdm(targets, desc="Train build (article)"):
        idx = int(path.stem.split("_")[1])
        out_path = TRAIN_ARTICLE_DIR / f"article_{idx}.jsonl"
        if out_path.exists():
            logging.info("Skip existing %s", out_path.name)
            continue
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qa = json.loads(line)
                q, a = qa.get("question", ""), qa.get("answer", "")
                if q and a:
                    records.append({"question": q, "answer": a})
        if args.include_summaries:
            summary_path = SUMMARY_DIR / f"article_{idx}_summaries.jsonl"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = json.loads(line).get("summary", "")
                        if s:
                            records.append({"text": s})
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_chunk(args):
    TRAIN_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    article_dirs = sorted(CHUNK_QA_DIR.glob("article_*"))
    targets = [d for d in article_dirs if args.start <= int(d.name.split("_")[1]) < args.end]
    logging.info(
        "Building CHUNK-level train files for %d articles (%d-%d), chunk_size=%d",
        len(targets),
        args.start,
        args.end - 1,
        args.chunk_size,
    )
    for adir in tqdm(targets, desc="Train build (chunk QA)"):
        idx = int(adir.name.split("_")[1])
        out_path = TRAIN_CHUNK_DIR / f"article_{idx}.jsonl"
        if out_path.exists():
            logging.info("Skip existing %s", out_path.name)
            continue
        records = []
        qa_files = sorted(adir.glob(f"chunk_{args.chunk_size}_*_qa.jsonl"))
        if not qa_files:
            logging.warning("No chunk QA files for article %d (size %d)", idx, args.chunk_size)
            continue
        for qf in qa_files:
            with open(qf, "r", encoding="utf-8") as f:
                for line in f:
                    qa = json.loads(line)
                    q, a = qa.get("question", ""), qa.get("answer", "")
                    if q and a:
                        records.append({"question": q, "answer": a})
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Assemble training files.")
    parser.add_argument("--mode", choices=["article", "chunk"], default="chunk", help="Use article QA or chunk QA")
    parser.add_argument("--chunk_size", type=int, default=768, help="Chunk size when mode=chunk")
    parser.add_argument("--include_summaries", action="store_true", help="Include summaries (article mode only)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    args = parser.parse_args()

    if args.mode == "article":
        build_article(args)
        logging.info("Training data saved to %s", TRAIN_ARTICLE_DIR)
    else:
        build_chunk(args)
        logging.info("Training data saved to %s", TRAIN_CHUNK_DIR)


if __name__ == "__main__":
    main()
