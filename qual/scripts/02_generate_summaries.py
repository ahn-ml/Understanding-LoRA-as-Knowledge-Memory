#!/usr/bin/env python3
"""
Generate summaries for each QuALITY article using Azure GPT-4.1.
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

SUMMARY_PROMPT = """Summarize the following article in a concise paragraph that preserves key facts and narrative flow.
Avoid extraneous commentary; keep it grounded in the text.

[ARTICLE]
{article_text}
"""

SOURCE_DIR = Path("qual/data/source")
SUMMARY_DIR = Path("qual/data/summaries")


def main():
    parser = argparse.ArgumentParser(description="Generate Azure summaries for QuALITY articles.")
    parser.add_argument("--num_summaries", type=int, default=1, help="Summaries per article.")
    parser.add_argument("--start", type=int, default=0, help="Start article index (inclusive).")
    parser.add_argument("--end", type=int, default=10_000, help="End article index (exclusive).")
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    client = get_openrouter_client({})
    headers = get_openrouter_headers()

    files = sorted(SOURCE_DIR.glob("article_*.txt"))
    targets = [f for f in files if args.start <= int(f.stem.split("_")[1]) < args.end]
    logging.info("Generating summaries for %d articles (%d-%d)", len(targets), args.start, args.end - 1)

    for path in tqdm(targets, desc="Summaries"):
        idx = int(path.stem.split("_")[1])
        out_path = SUMMARY_DIR / f"article_{idx}_summaries.jsonl"
        if out_path.exists():
            logging.info("Skip existing %s", out_path.name)
            continue

        article_text = path.read_text(encoding="utf-8")
        records = []
        for _ in range(args.num_summaries):
            prompt = SUMMARY_PROMPT.format(article_text=article_text)
            try:
                resp = client.chat.completions.create(
                    model="openai/gpt-4.1",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=2048,
                    response_format={"type": "text"},
                    extra_headers=headers,
                )
                summary = resp.choices[0].message.content.strip()
                records.append({"summary": summary})
            except Exception as e:
                logging.error("Summary generation failed for article %d: %s", idx, e)

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info("All summaries done. Saved to %s", SUMMARY_DIR)


if __name__ == "__main__":
    main()
