# narr/scripts/05_chunk_documents.py

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_narrativeqa(text: str) -> str:
    """Normalize NarrativeQA text."""
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

def chunk_text_with_overlap(text: str, tokenizer, chunk_size: int = 2048, overlap_size: int = 200):
    """Split text into chunks with overlap based on tokens."""
    tokens = tokenizer.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        
        # Move back by overlap_size for the next chunk
        start = end - overlap_size
    
    return chunks

def process_documents(doc_ids: list, output_dir: Path, model_path: Path):
    """Chunk documents and save them."""
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    logging.info("Loading NarrativeQA dataset...")
    full_ds = load_dataset("deepmind/narrativeqa", split="train+validation+test")
    
    all_chunks_data = []
    chunk_global_id = 0
    
    for doc_id in doc_ids:
        logging.info(f"\nProcessing document: {doc_id}")
        
        # Find document
        doc_text = None
        for item in full_ds:
            if item['document']['id'].startswith(doc_id):
                doc_text = normalize_narrativeqa(item['document']['text'])
                full_doc_id = item['document']['id']
                break
        
        if doc_text is None:
            logging.error(f"Document {doc_id} not found!")
            continue
        
        # Chunking
        chunks = chunk_text_with_overlap(doc_text, tokenizer, chunk_size=2048, overlap_size=200)
        logging.info(f"  Created {len(chunks)} chunks")
        
        doc_chunks = []
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_data = {
                'global_chunk_id': chunk_global_id,
                'doc_id': full_doc_id,
                'doc_short': doc_id,
                'chunk_idx': chunk_idx,
                'text': chunk_text,
                'token_count': len(tokenizer.encode(chunk_text))
            }
            doc_chunks.append(chunk_data)
            all_chunks_data.append(chunk_data)
            chunk_global_id += 1
        
        # Save chunks per document
        doc_output_dir = output_dir / f"doc_{doc_id}"
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        doc_chunks_path = doc_output_dir / "chunks.json"
        with open(doc_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(doc_chunks, f, indent=2, ensure_ascii=False)
        
        logging.info(f"  Saved to {doc_chunks_path}")
    
    # Save overall chunk metadata
    metadata_path = output_dir / "all_chunks_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_chunks': len(all_chunks_data),
            'documents': doc_ids,
            'chunk_size': 2048,
            'overlap_size': 200,
            'chunks': all_chunks_data
        }, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\nTotal chunks created: {len(all_chunks_data)}")
    logging.info(f"Metadata saved to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Chunks NarrativeQA documents.")
    parser.add_argument("--doc-ids", type=Path, default=DATA_DIR / "doc_ids.json", help="Path to document ID JSON to process")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "Llama-3.1-8B-Instruct", help="Path to tokenizer model")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "multi_lora" / "chunks", help="Directory to save chunks")
    args = parser.parse_args()

    if args.doc_ids.exists():
        with open(args.doc_ids, 'r', encoding='utf-8') as f:
            doc_ids = json.load(f)
    else:
        raise FileNotFoundError(f"Doc ID file not found: {args.doc_ids}")

    logging.info(f"Starting chunking for {len(doc_ids)} documents")
    logging.info(f"Chunk size: 2048 tokens, Overlap: 200 tokens")

    process_documents(doc_ids, args.output_dir, args.model_path)

    logging.info("\nChunking complete!")


if __name__ == "__main__":
    main()