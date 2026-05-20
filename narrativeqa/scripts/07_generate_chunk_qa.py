# narr/scripts/07_generate_chunk_qa.py

import argparse
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
    """Generate QA pairs, handling prompts for new or iterative generation."""
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
        
        # Parsing
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
        logging.error(f"Failed to generate QA: {e}")
    
    logging.warning("Returning empty list")
    return []

def process_chunk_qa(client: AzureOpenAI, chunk_id: int, summary: str, deployment_name: str = "gpt-4.1") -> list:
    """Generate QA for a single chunk (Initial batch + 2 iterations)."""
    
    all_qa_pairs = []
    
    # Initial generation (target 40)
    logging.info(f"  Chunk {chunk_id}: Generating initial 40 QAs...")
    initial_qas = generate_qa_iteratively(client, deployment_name, summary, num_to_gen=40, existing_qas=[])
    all_qa_pairs.extend(initial_qas)
    logging.info(f"    Generated {len(initial_qas)} QAs")
    
    # 2 Iterations
    for iteration in range(2):
        logging.info(f"  Chunk {chunk_id}: Iteration {iteration+1}/2...")
        new_qas = generate_qa_iteratively(client, deployment_name, summary, existing_qas=all_qa_pairs)
        all_qa_pairs.extend(new_qas)
        logging.info(f"    Generated {len(new_qas)} QAs (Total: {len(all_qa_pairs)})")
        time.sleep(1)  # Prevent API overload
    
    return all_qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Generates QA pairs based on chunk summaries.")
    parser.add_argument("--summaries-path", type=Path, default=DATA_DIR / "multi_lora" / "summaries" / "all_chunk_summaries.json", help="Path to chunk summary metadata JSON")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "multi_lora" / "qa", help="Directory to save generated QA")
    parser.add_argument("--deployment", type=str, default="gpt-4.1", help="Azure deployment name")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = get_azure_client()

    with open(args.summaries_path, 'r') as f:
        data = json.load(f)
        summaries = data['summaries']

    logging.info(f"Loaded {len(summaries)} chunk summaries")

    all_qa_data = {}

    for chunk_id_str, summary in tqdm(summaries.items(), desc="Generating QA for chunks"):
        chunk_id = int(chunk_id_str)
        logging.info(f"\nProcessing chunk {chunk_id}")

        qa_pairs = process_chunk_qa(client, chunk_id, summary, args.deployment)

        all_qa_data[chunk_id] = {
            'chunk_id': chunk_id,
            'total_qa': len(qa_pairs),
            'qa_pairs': qa_pairs
        }

        chunk_qa_path = args.output_dir / f"chunk_{chunk_id}_qa.jsonl"
        with open(chunk_qa_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    metadata_path = args.output_dir / "all_qa_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        total_qa = sum(data['total_qa'] for data in all_qa_data.values())
        json.dump({
            'total_chunks': len(all_qa_data),
            'total_qa_pairs': total_qa,
            'avg_qa_per_chunk': total_qa / len(all_qa_data) if all_qa_data else 0,
            'chunks': all_qa_data
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"\nQA generation complete!")
    logging.info(f"Total QA pairs: {sum(data['total_qa'] for data in all_qa_data.values())}")
    logging.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()