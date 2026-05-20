# scripts/03_evaluate_icl.py

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import evaluate
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from path_utils import DATA_DIR, OUTPUTS_DIR  # noqa: E402

DEFAULT_MODEL_ID = os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

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

def load_and_prepare_model(model_id: str, device: str = "cuda"):
    """Load model and tokenizer."""
    logging.info(f"Loading model and tokenizer: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")
        
    return model, tokenizer

def load_document_and_eval(doc_id):
    """Load document and evaluation data."""
    # Find source file
    source_path = DATA_DIR / "multi_doc" / "source" / f"doc_{doc_id}_source.txt"
    if not source_path.exists():
        # Load directly from raw dataset
        from datasets import load_dataset
        logging.info(f"Loading document {doc_id} from dataset...")
        full_ds = load_dataset("deepmind/narrativeqa", split="train+validation+test")
        context = None
        for item in full_ds:
            if item['document']['id'] == doc_id:
                context = normalize_narrativeqa(item['document']['text'])
                # Save as source file
                source_path.parent.mkdir(parents=True, exist_ok=True)
                with open(source_path, 'w', encoding='utf-8') as f:
                    f.write(context)
                logging.info(f"Saved source to {source_path}")
                break
        if context is None:
            raise ValueError(f"Document {doc_id} not found in dataset")
    else:
        with open(source_path, 'r', encoding='utf-8') as f:
            context = f.read()
    
    # Load evaluation data
    eval_path = DATA_DIR / "multi_doc" / "eval" / f"doc_{doc_id}_eval.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval data not found: {eval_path}")
    
    eval_data = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    logging.info(f"Loaded {len(eval_data)} eval questions for {doc_id[:8]}")
    return context, eval_data

def load_or_create_chunks(doc_id: str, context: str, tokenizer, chunk_size: int = 2048, overlap: int = 200):
    """Chunk document or load existing chunks."""
    chunks_path = DATA_DIR / "multi_doc" / "chunks" / f"doc_{doc_id}_chunks.json"
    
    if chunks_path.exists():
        logging.info(f"Loading existing chunks from {chunks_path}")
        with open(chunks_path, 'r') as f:
            return json.load(f)
    
    # Create chunks
    logging.info(f"Creating chunks for document {doc_id[:8]}")
    tokens = tokenizer.encode(context)
    chunks = []
    
    start = 0
    chunk_id = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append({
            'chunk_id': chunk_id,
            'text': chunk_text,
            'start_token': start,
            'end_token': end
        })
        chunk_id += 1
        
        if end >= len(tokens):
            break
        
        # Next start point considering overlap
        start = end - overlap
    
    # Save chunks
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Created and saved {len(chunks)} chunks")
    return chunks

def setup_retriever(chunks: list, device: str = "cuda"):
    """Setup FAISS-based retriever."""
    logging.info("Setting up retriever with BGE-large-en-v1.5")
    embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    
    # Extract chunk text
    texts = [chunk['text'] for chunk in chunks]
    
    # Create embeddings
    logging.info(f"Creating embeddings for {len(texts)} chunks")
    embeddings = embed_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    def retrieve(query: str, top_k: int = 3):
        query_embedding = embed_model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
        scores, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
        return indices[0].tolist(), scores[0].tolist()
    
    return retrieve

def evaluate_icl(doc_id, model, tokenizer, device="cuda"):
    """Basic ICL evaluation (full document)."""
    context, eval_data = load_document_and_eval(doc_id)
    
    predictions = []
    references = []
    
    # Check context length
    context_tokens = len(tokenizer.encode(context))
    logging.info(f"Context length: {context_tokens} tokens")
    
    for item in tqdm(eval_data, desc=f"ICL eval {doc_id[:8]}"):
        question = item['question']
        ref_answers = item['answers']
        
        # ICL Prompt
        user_prompt = f"{context}\n\nAnswer the following question. Give only the answer, and no extra commentary, formatting, or chattiness.\n\nQuestion: {question}"
        messages = [{"role": "user", "content": user_prompt}]
        
        # Apply chat template for Llama3 support
        try:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False  # For Llama3
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # Token limit (120K)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=120000)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        predictions.append(generated_text.strip())
        references.append(ref_answers)
    
    # Calculate ROUGE
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    
    return {
        'scores': results,
        'evaluation_data': [
            {'question': eval_data[i]['question'],
             'prediction': predictions[i],
             'references': references[i]}
            for i in range(len(predictions))
        ],
        'method': 'icl_full'
    }

def evaluate_rag(doc_id, model, tokenizer, device="cuda", top_k=3):
    """RAG evaluation (top-k chunk retrieval)."""
    context, eval_data = load_document_and_eval(doc_id)
    
    # Load or create chunks
    chunks = load_or_create_chunks(doc_id, context, tokenizer)
    
    # Setup Retriever
    retrieve = setup_retriever(chunks, device)
    
    predictions = []
    references = []
    retrieval_log = []
    
    for item in tqdm(eval_data, desc=f"RAG eval {doc_id[:8]}"):
        question = item['question']
        ref_answers = item['answers']
        
        # Retrieve top-k chunks
        chunk_indices, scores = retrieve(question, top_k=top_k)
        
        # Combine retrieved chunk texts
        retrieved_texts = [chunks[idx]['text'] for idx in chunk_indices]
        combined_context = "\n\n".join(retrieved_texts)
        
        # Save retrieval log
        retrieval_log.append({
            'question': question,
            'retrieved_chunks': chunk_indices,
            'scores': scores
        })
        
        # RAG Prompt
        user_prompt = f"""Context:
{combined_context}

Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness.

Question: {question}"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        # Apply chat template for Llama3 support
        try:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False  # For Llama3
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # Token limit (RAG is shorter)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        predictions.append(generated_text.strip())
        references.append(ref_answers)
    
    # Calculate ROUGE
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    
    return {
        'scores': results,
        'evaluation_data': [
            {'question': eval_data[i]['question'],
             'prediction': predictions[i],
             'references': references[i],
             'retrieved_chunks': retrieval_log[i]['retrieved_chunks'],
             'retrieval_scores': retrieval_log[i]['scores']}
            for i in range(len(predictions))
        ],
        'method': 'rag',
        'top_k': top_k,
        'num_chunks': len(chunks)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_id', type=str, required=True)
    parser.add_argument('--method', type=str, choices=['icl', 'rag', 'both'], default='both')
    parser.add_argument('--top_k', type=int, default=3, help='Number of chunks for RAG')
    args = parser.parse_args()
    
    doc_id = args.doc_id
    doc_short = doc_id[:8]
    
    logging.info(f"Starting evaluation for document {doc_short}")
    logging.info(f"Method: {args.method}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = DEFAULT_MODEL_ID
    logging.info(f"Loading model: {model_path}")
    
    model, tokenizer = load_and_prepare_model(model_path, device)
    model.eval()
    
    # Execute evaluation
    output_dir = OUTPUTS_DIR / "icl_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.method in ['icl', 'both']:
        logging.info("Running ICL evaluation...")
        icl_results = evaluate_icl(doc_id, model, tokenizer, device)
        
        # Save ICL results
        icl_output_path = output_dir / f"icl_{doc_short}_results.json"
        with open(icl_output_path, 'w') as f:
            json.dump(icl_results, f, indent=2)
        
        logging.info(f"ICL results saved: {icl_output_path}")
        logging.info(f"ICL ROUGE-L: {icl_results['scores']['rougeL']:.4f}")
    
    if args.method in ['rag', 'both']:
        logging.info(f"Running RAG evaluation (top-{args.top_k})...")
        rag_results = evaluate_rag(doc_id, model, tokenizer, device, top_k=args.top_k)
        
        # Save RAG results
        rag_output_path = output_dir / f"rag_top{args.top_k}_{doc_short}_results.json"
        with open(rag_output_path, 'w') as f:
            json.dump(rag_results, f, indent=2)
        
        logging.info(f"RAG results saved: {rag_output_path}")
        logging.info(f"RAG ROUGE-L: {rag_results['scores']['rougeL']:.4f}")

if __name__ == "__main__":
    main()
