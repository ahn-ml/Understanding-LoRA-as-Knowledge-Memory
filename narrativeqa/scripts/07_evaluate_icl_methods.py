#!/usr/bin/env python3
# scripts/07_evaluate_icl_methods.py
"""
Evaluation of Methods 1, 2, 3: ICL, Gold-summary ICL, RAG - Single Document Processing
"""

import os
import json
import logging
import torch
import time
import faiss
import numpy as np
import argparse
import gc
import sys
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
sys.path.append(str(ROOT_DIR))

from path_utils import DATA_DIR, OUTPUTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ICLEvaluator:
    """In-Context Learning Evaluator."""
    
    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        
        logging.info(f"Loading model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def evaluate_fulltext_icl(self, doc: Dict, eval_questions: List[Dict]) -> List[Dict]:
        """Method 1: Full-text ICL."""
        predictions = []
        full_text = doc['text']
        
        for item in tqdm(eval_questions, desc="Full-text ICL"):
            question = item['question']
            
            prompt = f"""Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness.

Text: {full_text}

Question: {question}
Answer:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=20000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            ).strip()
            
            predictions.append({
                'question': question,
                'prediction': generated_text,
                'reference': item['answers'],
                'method': 'fulltext_icl',
                'inference_time': inference_time
            })
        
        return predictions
    
    def evaluate_summary_icl(self, doc: Dict, eval_questions: List[Dict]) -> List[Dict]:
        """Method 2: Gold summary ICL."""
        predictions = []
        summary = doc['summary']
        
        for item in tqdm(eval_questions, desc="Summary ICL"):
            question = item['question']
            
            prompt = f"""Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness.

Summary: {summary}

Question: {question}
Answer:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            ).strip()
            
            predictions.append({
                'question': question,
                'prediction': generated_text,
                'reference': item['answers'],
                'method': 'summary_icl',
                'inference_time': inference_time
            })
        
        return predictions
    
    def evaluate_rag(self, doc: Dict, eval_questions: List[Dict], 
                      chunks_dir: Path, chunk_size: int = 2048) -> List[Dict]:
        """Method 3: RAG (top-3 chunks)."""
        predictions = []
        
        chunks = self._load_chunks(doc['id'], chunks_dir, chunk_size)
        if not chunks:
            logging.error(f"No chunks found for doc {doc['id']}")
            return predictions
        
        retriever = self._setup_retriever([c['text'] for c in chunks])
        
        for item in tqdm(eval_questions, desc="RAG"):
            question = item['question']
            
            top_indices = retriever(question, top_k=3)
            context_chunks = [chunks[i]['text'] for i in top_indices]
            context = "\n\n".join(context_chunks)

            prompt = f"""Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness.

Context:
{context}

Question: {question}
Answer:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            ).strip()
            
            predictions.append({
                'question': question,
                'prediction': generated_text,
                'reference': item['answers'],
                'method': 'rag',
                'retrieved_chunks': top_indices,
                'inference_time': inference_time
            })
        
        return predictions
    
    def _load_chunks(self, doc_id: str, chunks_dir: Path, chunk_size: int) -> List[Dict]:
        """Load chunks."""
        chunks_path = Path(chunks_dir) / f"doc_{doc_id}" / f"chunks_{chunk_size}.jsonl"
        chunks = []
        
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunks.append(json.loads(line))
        
        return chunks
    
    def _setup_retriever(self, texts: List[str]):
        """Setup FAISS-based retriever."""
        embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=self.device)
        embeddings = embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.cpu().numpy())
        
        def retrieve(query: str, top_k: int = 3) -> List[int]:
            query_embedding = embed_model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            _, indices = index.search(query_embedding.cpu().numpy(), top_k)
            return indices[0].tolist()
        
        return retrieve

def calculate_metrics(predictions: List[Dict]) -> Dict:
    """Calculate evaluation metrics."""
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for pred in predictions:
        pred_text = pred['prediction']
        ref_texts = pred['reference']
        
        max_score = 0
        for ref in ref_texts:
            score = scorer.score(ref, pred_text)['rougeL'].fmeasure
            max_score = max(max_score, score)
        
        scores.append(max_score)
    
    avg_score = np.mean(scores) if scores else 0
    avg_time = np.mean([p['inference_time'] for p in predictions])
    
    return {
        'rouge_l': avg_score,
        'avg_inference_time': avg_time,
        'num_samples': len(predictions)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL methods for single document")
    parser.add_argument("--doc_id", type=str, required=True,
                        help="Document ID to evaluate")
    parser.add_argument("--doc_index", type=int, required=True,
                        help="Document index in the file")
    parser.add_argument("--input_path", type=Path,
                        default=DATA_DIR / "source_document.jsonl",
                        help="Path to documents")
    parser.add_argument("--eval_dir", type=Path, default=DATA_DIR / "multi_doc" / "eval",
                        help="Directory containing evaluation questions")
    parser.add_argument("--chunks_dir", type=Path, default=DATA_DIR / "multi_doc" / "chunks",
                        help="Directory containing chunks")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS_DIR / "icl_results",
                        help="Output directory for predictions")
    parser.add_argument("--model_id", type=str, 
                         default=DEFAULT_MODEL_ID,
                         help="Model ID or path")
    parser.add_argument("--methods", nargs='+', default=['fulltext_icl', 'summary_icl', 'rag'],
                        help="Methods to evaluate")
    args = parser.parse_args()
    
    # Load single document
    logging.info(f"Loading document {args.doc_id} at index {args.doc_index}")
    doc = None
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == args.doc_index:
                doc = json.loads(line)
                break
    
    if not doc or doc['id'] != args.doc_id:
        logging.error(f"Document {args.doc_id} not found at index {args.doc_index}")
        return
    
    # Load evaluation questions
    # Note: Filename format should match the output from the preparation script
    eval_path = args.eval_dir / f"doc_{args.doc_id}_eval.jsonl"
    eval_questions = []
    if eval_path.exists():
        with open(eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                eval_questions.append(json.loads(line))
    else:
        logging.error(f"Eval file not found: {eval_path}")
        return

    # Initialize evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = ICLEvaluator(args.model_id, device)
    
    # Evaluate each method
    for method in args.methods:
        logging.info(f"--- Starting evaluation for method: {method} ---")
        doc_results = {}
        predictions = []

        if method == 'fulltext_icl':
            predictions = evaluator.evaluate_fulltext_icl(doc, eval_questions)
        elif method == 'summary_icl':
            predictions = evaluator.evaluate_summary_icl(doc, eval_questions)
        elif method == 'rag':
            predictions = evaluator.evaluate_rag(doc, eval_questions, args.chunks_dir)
        
        if not predictions:
            logging.warning(f"No predictions generated for method: {method}")
            continue

        doc_results[method] = {
            'predictions': predictions,
            'metrics': calculate_metrics(predictions)
        }
        
        # Save results (create sub-folders)
        method_output_dir = args.output_dir / method
        method_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = method_output_dir / f"doc_{args.doc_id}_predictions.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results for method '{method}' saved to {output_path}")

        gc.collect()
        torch.cuda.empty_cache()
    
    logging.info(f"Document {args.doc_id} evaluation complete")
    
    # Clean up memory
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
