# scripts/02_prepare_all_qa.py

import json
import logging
import os
import sys
import time
from pathlib import Path
from openai import AzureOpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_azure_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client."""
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        raise ValueError("Please set environment variables: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT")
    
    return AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint=endpoint,
        api_key=api_key
    )

def generate_qa_iteratively(client: AzureOpenAI, deployment_name: str, summary: str, 
                           num_to_gen: int = None, existing_qas: list = None) -> list:
    """Iteratively generate QA pairs based on summary."""
    if existing_qas is None:
        existing_qas = []

    if not existing_qas:
        generation_instruction = f"generate exactly {num_to_gen} diverse question-answer pairs"
        refinement_prompt_part = ""
    else:
        generation_instruction = "generate a new batch of diverse question-answer pairs"
        existing_qa_str = "\n".join([f"- Q: {item['question']} A: {item['answer']}" for item in existing_qas])
        refinement_prompt_part = f"""
        The following question-answer pairs have already been created.
        Your task is to generate NEW questions that cover important facts from the summary text that are NOT covered by the existing pairs.
        
        [EXISTING QA PAIRS]
        {existing_qa_str}
        """

    system_prompt = "You are an expert in creating high-quality, fact-based, short-answer question pairs for training language models."
    user_prompt = f"""
    Based on the following summary text, {generation_instruction} that cover key facts.
    {refinement_prompt_part}

    **CRITICAL REQUIREMENTS:**
    1. **Question Type:** Generate **"WH-questions"** (Who, What, Where, When, etc.).
    2. **Answer Style:** Answers must be **concise** (ideally under 10 words).
    3. **Content:** Questions must be answerable *only* from the provided text.
    4. **Format:** Your output MUST be a valid JSON object with a single key "qa_pairs".
    
    [SUMMARY TEXT]
    {summary}
    """
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, 
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        content = json.loads(response_text)
        qa_pairs_raw = content.get("qa_pairs", [])
        
        # Handle response keys whether 'Q'/'A' or 'question'/'answer'
        processed_pairs = []
        if isinstance(qa_pairs_raw, list):
            for item in qa_pairs_raw:
                if isinstance(item, dict):
                    question = item.get('question') or item.get('Q')
                    answer = item.get('answer') or item.get('A')
                    if question and answer:
                        processed_pairs.append({"question": question, "answer": answer})
        
        if processed_pairs:
            return processed_pairs

    except Exception as e:
        logging.error(f"Error generating or parsing QA: {e}")
    
    logging.warning("Failed to parse valid QA pairs. Returning empty list.")
    return []

def process_single_document(client: AzureOpenAI, doc_id: str, deployment_name: str = "gpt-4.1"):
    """Process single document for iterative QA generation."""
    
    # Read summary file
    summary_path = DATA_DIR / "multi_doc" / "summaries" / f"doc_{doc_id}_summary.txt"
    if not summary_path.exists():
        logging.error(f"Summary not found for doc {doc_id}")
        return False
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        gold_summary = f.read()
    
    logging.info(f"Processing doc {doc_id[:8]}... (summary length: {len(gold_summary)})")
    
    # Iterative QA generation settings
    all_qa_pairs = []
    initial_qa_count = 50
    refinement_iterations = 4
    
    # Initial QA generation
    logging.info(f"  - Generating initial {initial_qa_count} QAs...")
    new_qas = generate_qa_iteratively(client, deployment_name, gold_summary, 
                                     num_to_gen=initial_qa_count, existing_qas=[])
    all_qa_pairs.extend(new_qas)
    logging.info(f"    Generated: {len(new_qas)} QAs (Total: {len(all_qa_pairs)})")
    
    # Refinement iterations
    for i in range(refinement_iterations):
        logging.info(f"  - Refinement iteration {i+1}/{refinement_iterations}...")
        new_qas = generate_qa_iteratively(client, deployment_name, gold_summary, 
                                         existing_qas=all_qa_pairs)
        all_qa_pairs.extend(new_qas)
        logging.info(f"    Generated: {len(new_qas)} QAs (Total: {len(all_qa_pairs)})")
        
        # Consider API rate limits
        time.sleep(1)
    
    # Save QA pairs
    qa_path = DATA_DIR / "multi_doc" / "qa" / f"doc_{doc_id}_qa.jsonl"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(qa_path, 'w', encoding='utf-8') as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    logging.info(f"Saved {len(all_qa_pairs)} QAs for doc {doc_id[:8]}...")
    return True

def load_eval_data_for_document(doc_id: str) -> bool:
    """Prepare evaluation data for each document (using original NarrativeQA questions)."""
    from datasets import load_dataset, concatenate_datasets
    
    # Find questions for the specific document in the full dataset
    full_ds = concatenate_datasets([
        load_dataset("deepmind/narrativeqa", split="train"),
        load_dataset("deepmind/narrativeqa", split="validation"),
        load_dataset("deepmind/narrativeqa", split="test")
    ])
    
    eval_questions = []
    for item in full_ds:
        if item['document']['id'] == doc_id:
            eval_questions.append({
                'question': item['question']['text'],
                'answers': [item['answers'][0]['text'], item['answers'][1]['text']]
            })
    
    if eval_questions:
        # Save evaluation data
        eval_path = DATA_DIR / "multi_doc" / "eval" / f"doc_{doc_id}_eval.jsonl"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            for item in eval_questions:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"  Saved {len(eval_questions)} eval questions for doc {doc_id[:8]}...")
        return True
    else:
        logging.warning(f"  No eval questions found for doc {doc_id[:8]}...")
        return False

def main():
    """Main execution function."""
    # 1. Load doc_ids
    doc_ids_path = DATA_DIR / "doc_ids.json"
    with open(doc_ids_path, 'r') as f:
        doc_ids = json.load(f)
    
    logging.info(f"Starting QA generation for {len(doc_ids)} documents")
    
    # 2. Initialize Azure Client
    client = get_azure_client()
    
    # 3. Process documents sequentially
    successful = 0
    failed = []
    
    for idx, doc_id in enumerate(tqdm(doc_ids, desc="Processing documents"), 1):
        logging.info(f"\n[{idx}/{len(doc_ids)}] Processing document: {doc_id}")
        
        try:
            # Generate QA
            success = process_single_document(client, doc_id)
            
            # Prepare evaluation data
            eval_success = load_eval_data_for_document(doc_id)
            
            if success:
                successful += 1
            else:
                failed.append(doc_id)
                
        except Exception as e:
            logging.error(f"Failed to process doc {doc_id}: {e}")
            failed.append(doc_id)
            continue
    
    # 4. Final statistics
    print("\n" + "="*50)
    print("QA generation complete")
    print(f"Successfully processed: {successful}/{len(doc_ids)} documents")
    if failed:
        print(f"Failed documents: {[f[:8] for f in failed]}")
    print(f"QA output directory: {DATA_DIR / 'multi_doc' / 'qa'}")
    print(f"Eval output directory: {DATA_DIR / 'multi_doc' / 'eval'}")
    print("="*50)

if __name__ == "__main__":
    main()
