#!/usr/bin/env python3
# scripts/06_generate_chunk_summaries.py

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from openai import AzureOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_azure_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client"""
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        raise ValueError("Please set environment variables: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT")
    
    return AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint=endpoint,
        api_key=api_key
    )

def create_batch_requests_for_document(doc_id: str, chunks: list, deployment_name: str = "gpt-4.1") -> list:
    """Generate batch requests for all chunks in a document"""
    batch_requests = []
    
    for chunk_data in chunks:
        chunk_text = chunk_data['text']
        global_chunk_id = chunk_data['global_chunk_id']
        
        # Modified prompt for chunks
        prompt = f"""
You are given a segment from a literary work.
Your task: Generate a single **faithful, detailed summary** in the following style:

- Start with a clear statement of the **setting, background, and main characters** present in this segment.
- Retell the **events strictly in chronological order** as they appear in this segment.
- Include all **important actions, conversations, conflicts, and revelations** in this segment.
- Do **not** add interpretation, analysis, symbolism, themes, imagery, or commentary.
- Write in clear, neutral prose in the past tense, using concise factual sentences.
- Be thorough and comprehensive for this segment.

Your output MUST be a valid JSON object with a single key "summary".

[TEXT SEGMENT]
{chunk_text}
"""
        
        batch_request = {
            "custom_id": f"chunk_{global_chunk_id}",
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": deployment_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"}
            }
        }
        batch_requests.append(batch_request)
    
    return batch_requests

def submit_batch_job(client: AzureOpenAI, doc_id: str, batch_requests: list, batch_dir: Path) -> str:
    """Create batch file, upload, and submit job"""
    
    # Save batch file
    batch_file_path = batch_dir / f"doc_{doc_id}_summary_batch.jsonl"
    with open(batch_file_path, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    logging.info(f"  Saved batch file: {batch_file_path}")
    
    try:
        # Upload file
        with open(batch_file_path, 'rb') as f:
            file_response = client.files.create(file=f, purpose="batch")
        
        logging.info(f"  Uploaded file (file_id: {file_response.id})")
        
        # Submit batch job
        batch_response = client.batches.create(
            input_file_id=file_response.id,
            endpoint="/chat/completions",
            completion_window="24h"
        )
        
        logging.info(f"  Submitted batch job (batch_id: {batch_response.id})")
        return batch_response.id
        
    except Exception as e:
        logging.error(f"Batch submission failed for doc {doc_id}: {e}")
        raise

def wait_for_batch_completion(client: AzureOpenAI, batch_id: str, doc_id: str, check_interval: int = 30) -> str:
    """Wait for batch job completion"""
    logging.info(f"Waiting for doc {doc_id} batch completion...")
    
    start_time = time.time()
    while True:
        try:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            
            elapsed = int((time.time() - start_time) / 60)
            
            if status == "completed":
                logging.info(f"Doc {doc_id} batch completed after {elapsed} minutes!")
                return batch_status.output_file_id
            elif status in ["failed", "cancelled", "expired"]:
                logging.error(f"Error: Doc {doc_id} batch failed: {status}")
                raise Exception(f"Batch job failed: {status}")
            else:
                logging.info(f"  Doc {doc_id}: Status={status}, Elapsed={elapsed}min")
                time.sleep(check_interval)
                
        except Exception as e:
            logging.error(f"Error checking batch status: {e}")
            time.sleep(check_interval)

def process_batch_results(client: AzureOpenAI, output_file_id: str, doc_id: str, chunks: list) -> dict:
    """Process and save batch results"""
    try:
        # Download result file
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')
        
        summaries = {}
        
        for raw_response in raw_responses:
            try:
                response_data = json.loads(raw_response)
                custom_id = response_data.get('custom_id', '')
                
                if response_data.get('response', {}).get('status_code') == 200:
                    content = response_data['response']['body']['choices'][0]['message']['content']
                    data = json.loads(content)
                    summary = data.get('summary', '')
                    
                    if summary and custom_id.startswith('chunk_'):
                        chunk_id = int(custom_id.replace('chunk_', ''))
                        summaries[chunk_id] = summary
                else:
                    error_info = response_data.get('error', {})
                    logging.error(f"API error for {custom_id}: {error_info}")
                    
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Failed to parse response: {e}")
                continue
        
        logging.info(f"Doc {doc_id}: Processed {len(summaries)} summaries")
        return summaries
        
    except Exception as e:
        logging.error(f"Failed to process batch results: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate chunk summaries using Azure Batch.")
    parser.add_argument("--doc-ids", type=Path, default=DATA_DIR / "doc_ids.json", help="JSON list of doc_ids to process")
    parser.add_argument("--chunks-dir", type=Path, default=DATA_DIR / "multi_lora" / "chunks", help="Directory containing chunk JSONs")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "multi_lora" / "summaries", help="Directory to save summaries")
    parser.add_argument("--batch-dir", type=Path, default=DATA_DIR / "multi_lora" / "batch_files", help="Directory to save batch request files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.batch_dir.mkdir(parents=True, exist_ok=True)

    client = get_azure_client()

    if args.doc_ids.exists():
        with open(args.doc_ids, 'r', encoding='utf-8') as f:
            doc_ids = json.load(f)
    else:
        doc_ids = [p.name.replace("doc_", "") for p in (args.chunks_dir).glob("doc_*")]

    metadata_path = args.chunks_dir / "all_chunks_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    all_summaries = {}
    batch_jobs = {}

    for doc_id in doc_ids:
        logging.info(f"\nProcessing document: {doc_id}")
        doc_chunks_path = args.chunks_dir / f"doc_{doc_id}" / "chunks.json"
        with open(doc_chunks_path, 'r') as f:
            chunks = json.load(f)

        batch_requests = create_batch_requests_for_document(doc_id, chunks)
        batch_id = submit_batch_job(client, doc_id, batch_requests, args.batch_dir)
        batch_jobs[doc_id] = batch_id

    for doc_id, batch_id in batch_jobs.items():
        output_file_id = wait_for_batch_completion(client, batch_id, doc_id)
        doc_chunks_path = args.chunks_dir / f"doc_{doc_id}" / "chunks.json"
        with open(doc_chunks_path, 'r') as f:
            chunks = json.load(f)

        doc_summaries = process_batch_results(client, output_file_id, doc_id, chunks)
        all_summaries.update(doc_summaries)

        doc_output_path = args.output_dir / f"doc_{doc_id}_summaries.json"
        with open(doc_output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_summaries, f, indent=2, ensure_ascii=False)

    all_summaries_path = args.output_dir / "all_chunk_summaries.json"
    with open(all_summaries_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(all_summaries),
            'summaries': all_summaries,
            'metadata': metadata
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"\nSummary generation complete!")
    logging.info(f"Total summaries: {len(all_summaries)}")
    logging.info(f"Output: {all_summaries_path}")


if __name__ == "__main__":
    main()