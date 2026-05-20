#!/usr/bin/env python3
# scripts/01_prepare_source_doc.py

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from logic.utils import load_config
from path_utils import DATA_DIR, resolve_with_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_source_data(config: dict, force_redownload: bool = False):
    """
    Retrieves data for the doc_id specified in the config from the NarrativeQA dataset
    and saves the following to the local data folder:
    1. Full original source data (including summaries)
    2. Evaluation question data
    """
    doc_id = config['doc_id']
    source_file_path = DATA_DIR / "source_document.jsonl"
    eval_file_path = resolve_with_root(config['evaluation']['eval_data_path'])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eval_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Download and filter original data
    if not source_file_path.exists() or force_redownload:
        logging.info("Local source data missing or redownload requested. Downloading from Hugging Face...")
        try:
            full_ds = load_dataset("narrativeqa", trust_remote_code=True)
            combined_ds = concatenate_datasets([full_ds['train'], full_ds['validation'], full_ds['test']])
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

        doc_data_list = [item for item in combined_ds if item['document']['id'] == doc_id]

        if not doc_data_list:
            raise FileNotFoundError(f"Document ID '{doc_id}' not found in the dataset.")

        with open(source_file_path, 'w', encoding='utf-8') as f:
            for item in doc_data_list:
                f.write(json.dumps(item) + '\n')
        logging.info(f"Saved source data for document ID '{doc_id}' to '{source_file_path}'.")
    else:
        logging.info(f"Using local file '{source_file_path}'. Use --force-redownload option to redownload.")

    # 2. Extract and save evaluation questions
    eval_questions = []
    seen_questions = set()
    with open(source_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question_text = item['question']['text']
            if question_text not in seen_questions:
                eval_questions.append({
                    "question": question_text,
                    "answers": [ans['text'] for ans in item['answers']]
                })
                seen_questions.add(question_text)

    with open(eval_file_path, 'w', encoding='utf-8') as f:
        for item in eval_questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logging.info(f"Saved {len(eval_questions)} evaluation questions (after deduplication) to '{eval_file_path}'.")
    logging.info("Source data preparation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare source document and evaluation questions for the experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to the main configuration file")
    parser.add_argument("--force-redownload", action="store_true", help="Force redownload even if the local file exists")
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_source_data(config, args.force_redownload)