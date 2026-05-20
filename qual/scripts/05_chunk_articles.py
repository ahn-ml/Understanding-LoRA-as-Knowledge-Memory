#!/usr/bin/env python3
"""
Chunk QuALITY articles for RAG / multi-LoRA.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SOURCE_DIR = Path("qual/data/source")
CHUNK_DIR = Path("qual/data/chunks")


def main():
    parser = argparse.ArgumentParser(description="Chunk QuALITY articles.")
    parser.add_argument("--tokenizer_id", type=str, default=os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    # 기본: 평균 8개 내외 청크를 얻기 위해 768 토큰 사용
    parser.add_argument("--sizes", type=int, nargs="+", default=[768])
    parser.add_argument("--overlap_ratio", type=float, default=0.1)
    args = parser.parse_args()

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    files = sorted(SOURCE_DIR.glob("article_*.txt"))
    logging.info("Chunking %d articles with tokenizer %s", len(files), args.tokenizer_id)

    for path in tqdm(files, desc="Chunking"):
        idx = int(path.stem.split("_")[1])
        text = path.read_text(encoding="utf-8")
        for size in args.sizes:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=int(size * args.overlap_ratio),
                length_function=lambda t: len(tokenizer.encode(t, add_special_tokens=False)),
                separators=["\n\n", "\n", ". ", "? ", "! ", " "],
            )
            chunks = splitter.split_text(text)
            out_dir = CHUNK_DIR / f"article_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"chunks_{size}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for i, chunk_text in enumerate(chunks):
                    f.write(json.dumps({"chunk_id": i, "text": chunk_text}, ensure_ascii=False) + "\n")

    logging.info("Chunks saved to %s", CHUNK_DIR)


if __name__ == "__main__":
    main()
