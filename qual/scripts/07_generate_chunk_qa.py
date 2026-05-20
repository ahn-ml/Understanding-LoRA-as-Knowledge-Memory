#!/usr/bin/env python3
"""
Generate QA pairs per chunk using chunk summaries (for multi-LoRA training).
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

CHUNK_SUMMARY_DIR = Path("qual/data/chunk_summaries")
OUT_DIR = Path("qual/data/chunk_qa")

PROMPT = """You are an expert question-answer generator.
Based on the summary below, generate exactly 10 diverse WH-questions with concise answers (<=12 words).
Output MUST be a JSON object with key "qa_pairs" -> list of objects with "question" and "answer".

[SUMMARY]
{summary}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate QA per chunk summary.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = get_openrouter_client({})
    headers = get_openrouter_headers()

    article_dirs = sorted(CHUNK_SUMMARY_DIR.glob("article_*"))
    targets = [d for d in article_dirs if args.start <= int(d.name.split("_")[1]) < args.end]
    logging.info("Generating chunk QA for %d articles (%d-%d)", len(targets), args.start, args.end - 1)

    for adir in tqdm(targets, desc="Chunk QA"):
        idx = int(adir.name.split("_")[1])
        for summary_file in adir.glob("chunk_summaries_*.jsonl"):
            size = int(summary_file.stem.split("_")[2])
            out_dir = OUT_DIR / f"article_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # write one QA file per chunk id
            # iterate summaries
            with open(summary_file, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    cid = rec["chunk_id"]
                    out_path = out_dir / f"chunk_{size}_{cid}_qa.jsonl"
                    if out_path.exists():
                        continue
                    summary = rec.get("summary", "")
                    try:
                        resp = client.chat.completions.create(
                            model="openai/gpt-4.1",
                            messages=[{"role": "user", "content": PROMPT.format(summary=summary)}],
                            temperature=0.7,
                            max_tokens=1024,
                            response_format={"type": "json_object"},
                            extra_headers=headers,
                        )
                        content = resp.choices[0].message.content
                        data = json.loads(content)
                        pairs = data.get("qa_pairs", [])
                        with open(out_path, "w", encoding="utf-8") as f_out:
                            for qa in pairs:
                                q = qa.get("question") or qa.get("Q")
                                a = qa.get("answer") or qa.get("A")
                                if q and a:
                                    f_out.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logging.error("QA generation failed for article %d chunk %d: %s", idx, cid, e)

    logging.info("Chunk QA saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
