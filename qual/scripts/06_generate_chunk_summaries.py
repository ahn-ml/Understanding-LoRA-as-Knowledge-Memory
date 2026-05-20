#!/usr/bin/env python3
"""
Generate summaries for each chunk (per article, per chunk size) for multi-LoRA/RAG.
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

from qual.logic.utils import get_openrouter_client, get_openrouter_headers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHUNK_DIR = Path("qual/data/chunks")
OUT_DIR = Path("qual/data/chunk_summaries")

PROMPT = """Summarize the following passage in 2-3 sentences, retaining key facts and entities.
Be concise and avoid extra commentary.

[PASSAGE]
{text}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate chunk-level summaries.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = get_openrouter_client({})
    headers = get_openrouter_headers()

    article_dirs = sorted(CHUNK_DIR.glob("article_*"))
    targets = [d for d in article_dirs if args.start <= int(d.name.split("_")[1]) < args.end]
    logging.info("Generating chunk summaries for %d articles (%d-%d)", len(targets), args.start, args.end - 1)

    for adir in tqdm(targets, desc="Chunk summaries"):
        idx = int(adir.name.split("_")[1])
        for chunk_file in adir.glob("chunks_*.jsonl"):
            size = int(chunk_file.stem.split("_")[1])
            out_dir = OUT_DIR / f"article_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"chunk_summaries_{size}.jsonl"
            if out_path.exists():
                logging.info("Skip existing %s", out_path.name)
                continue
            records = []
            with open(chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    text = rec["text"]
                    try:
                        resp = client.chat.completions.create(
                            model="openai/gpt-4.1",
                            messages=[{"role": "user", "content": PROMPT.format(text=text)}],
                            temperature=0.3,
                            max_tokens=512,
                            response_format={"type": "text"},
                            extra_headers=headers,
                        )
                        summary = resp.choices[0].message.content.strip()
                    except Exception as e:
                        logging.error("Summary failed for article %d chunk %s: %s", idx, rec["chunk_id"], e)
                        summary = ""
                    records.append({"chunk_id": rec["chunk_id"], "summary": summary})
            with open(out_path, "w", encoding="utf-8") as f_out:
                for r in records:
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info("Chunk summaries saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
