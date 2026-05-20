#!/usr/bin/env python3
"""
Generate QA pairs from Azure summaries (iterative refinement, narr-style).
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qual.logic.utils import get_openrouter_client, get_openrouter_headers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QA_PROMPT = """You are an expert in creating high-quality, fact-based, short-answer question pairs for training language models.

Based on the following summary text, {instruction}

Requirements:
1. Use WH-questions (Who, What, Where, When, Why, How, etc.).
2. Answers must be concise (under 12 words) and grounded only in the summary.
3. Output MUST be a JSON object with a single key "qa_pairs" whose value is a list of objects with "question" and "answer".

[SUMMARY]
{summary_text}
{existing_block}
"""

SUMMARY_DIR = Path("qual/data/summaries")
QA_DIR = Path("qual/data/qa")


def generate_batch(client, summary: str, num: int, existing: list, headers: dict) -> list:
    if existing:
        instruction = "generate NEW question-answer pairs that cover important facts NOT covered by the existing pairs."
        existing_block = "[EXISTING]\n" + "\n".join(
            [f"- Q: {e['question']} A: {e['answer']}" for e in existing]
        )
    else:
        instruction = f"generate exactly {num} diverse question-answer pairs."
        existing_block = ""

    prompt = QA_PROMPT.format(instruction=instruction, summary_text=summary, existing_block=existing_block)
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"},
            extra_headers=headers,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        pairs = data.get("qa_pairs", [])
        processed = []
        for item in pairs:
            if isinstance(item, dict):
                q = item.get("question") or item.get("Q")
                a = item.get("answer") or item.get("A")
                if q and a:
                    processed.append({"question": q, "answer": a})
        return processed
    except Exception as e:
        logging.error("QA generation failed: %s", e)
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from summaries.")
    parser.add_argument("--initial_count", type=int, default=40, help="Initial QA count.")
    parser.add_argument("--refine_iterations", type=int, default=2, help="Number of refinement rounds.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    args = parser.parse_args()

    QA_DIR.mkdir(parents=True, exist_ok=True)
    client = get_openrouter_client({})
    headers = get_openrouter_headers()

    summary_files = sorted(SUMMARY_DIR.glob("article_*_summaries.jsonl"))
    targets = [f for f in summary_files if args.start <= int(f.stem.split("_")[1]) < args.end]
    logging.info("Generating QA for %d articles (%d-%d)", len(targets), args.start, args.end - 1)

    for path in tqdm(targets, desc="QA generation"):
        idx = int(path.stem.split("_")[1])
        out_path = QA_DIR / f"article_{idx}_qa.jsonl"
        if out_path.exists():
            logging.info("Skip existing %s", out_path.name)
            continue

        summaries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                summaries.append(json.loads(line)["summary"])
        if not summaries:
            logging.warning("No summaries for article %d", idx)
            continue

        # use first summary
        summary = summaries[0]
        all_pairs = []

        initial = generate_batch(client, summary, num=args.initial_count, existing=[], headers=headers)
        all_pairs.extend(initial)

        for _ in range(args.refine_iterations):
            new_pairs = generate_batch(client, summary, num=0, existing=all_pairs, headers=headers)
            all_pairs.extend(new_pairs)
            time.sleep(1)

        with open(out_path, "w", encoding="utf-8") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logging.info("QA generation complete. Output: %s", QA_DIR)


if __name__ == "__main__":
    main()
