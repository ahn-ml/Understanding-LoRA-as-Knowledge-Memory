# narr/scripts/01_prepare_data.py

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from logic.utils import load_config  # noqa: E402
from path_utils import DATA_DIR  # noqa: E402

def normalize_narrativeqa(text: str) -> str:
    """Normalize NarrativeQA text (remove headers/footers and tags)."""
    if "*** START OF THIS PROJECT" in text:
        text = text.split("*** START OF THIS PROJECT")[1]
    if "***START OF THE PROJECT" in text:
        text = text.split("***START OF THE PROJECT")[1]
    if "*** END OF THIS PROJECT" in text:
        text = text.split("*** END OF THIS PROJECT")[0]
    if "***END OF THE PROJECT" in text:
        text = text.split("***END OF THE PROJECT")[0]
    
    text = text.split("<pre>")[-1]
    text = text.split("</pre>")[0]
    text = text.replace("<b>", "").replace("</b>", "")
    text = text.replace("[Illustration]", "")
    
    return text.strip()

def main(config_path: str):
    """
    1. Download and normalize source data using doc_id, save source_document.jsonl.
    2. Chunk the normalized text and save to chunks.jsonl.
    """
    config = load_config(config_path)
    doc_id = config['doc_id']
    base_model_id = config['base_model_id']
    
    chunks_dir = DATA_DIR / "chunks" / f"doc_{doc_id}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    source_file_path = DATA_DIR / "source_document.jsonl"

    # 1. Download, normalize, and save data
    logging.info(f"Searching for document ID '{doc_id}' in NarrativeQA dataset...")
    full_ds = concatenate_datasets([
        load_dataset("deepmind/narrativeqa", split="train"),
        load_dataset("deepmind/narrativeqa", split="validation"),
        load_dataset("deepmind/narrativeqa", split="test")
    ])

    doc_data_list = []
    normalized_text = ""
    
    for item in full_ds:
        if item['document']['id'] == doc_id:
            if not normalized_text:
                # Perform normalization only once
                normalized_text = normalize_narrativeqa(item['document']['text'])
                item['document']['text'] = normalized_text
            else:
                item['document']['text'] = normalized_text
            doc_data_list.append(item)

    if not doc_data_list:
        raise FileNotFoundError(f"Document ID '{doc_id}' not found.")

    with open(source_file_path, 'w', encoding='utf-8') as f:
        for item in doc_data_list:
            f.write(json.dumps(item) + '\n')
    logging.info(f"Saved normalized source data to '{source_file_path}'.")

    # 2. Text chunking
    logging.info("Starting text chunking...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=200,
        length_function=lambda text: len(tokenizer.encode(text))
    )
    chunks = text_splitter.split_text(normalized_text)
    
    chunk_output_path = chunks_dir / "chunks.jsonl"
    with open(chunk_output_path, 'w', encoding='utf-8') as f:
        for i, chunk_text in enumerate(chunks):
            f.write(json.dumps({"chunk_id": i, "text": chunk_text}) + '\n')
            
    logging.info(f"Split source text into {len(chunks)} chunks and saved to '{chunk_output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data normalization and chunking")
    parser.add_argument("--config", type=str, required=True, help="Path to base configuration file")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args.config)