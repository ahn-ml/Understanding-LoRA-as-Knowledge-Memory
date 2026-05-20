#!/usr/bin/env python3
# analysis/evaluate_baselines_fast_llama.py
"""
Evaluate ICL and RAG baselines using Llama 3.1 base model (No VLLM).
- Supports evaluation for both PhoneBook (Accuracy) and CounterFact (Efficacy).
- Performs sequential inference and time measurement using the standard transformers library.
- RAG-C: Semantic search using FAISS vector DB (512 token chunking).
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineEvaluatorLlama:
    def __init__(self, model_path: str, device: str):
        self.device = device
        logging.info(f"Loading base model: {model_path}")
        
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            max_position_embeddings=24576,  # Llama 3.1 context window
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.model_max_length = 24576  # Explicitly set context window
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
        # Embedding model for RAG
        self.embed_model = None

    def _load_rag_index(self, dataset: str, size_str: str) -> Tuple[faiss.Index, List[str], Dict]:
        """Loads FAISS index and chunks."""
        if self.embed_model is None:
            self.embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            
        # Vector DB file paths
        db_basename = f"{dataset}_{'qa' if dataset == 'phonebook' else 'edit'}_{size_str}"
        vector_db_dir = os.path.join(PROJECT_ROOT, "runs", "rags", "vector_db")
        index_path = os.path.join(vector_db_dir, f"{db_basename}_bge_large.index")
        chunks_path = os.path.join(vector_db_dir, f"{db_basename}_bge_large_chunks.json")
        metadata_path = os.path.join(vector_db_dir, f"{db_basename}_bge_large_metadata.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector DB index not found: {index_path}")
            
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load chunk data
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        logging.info(f"RAG index loaded: {len(chunks)} chunks, embedding dim: {metadata['embedding_dim']}")
        return index, chunks, metadata
    
    def _retrieve_context_rag(self, query: str, index: faiss.Index, chunks: List[str], top_k: int = 3) -> str:
        """Semantic search using FAISS (RAG-C)."""
        

        # Query embedding
        query_embedding = self.embed_model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Retrieve top_k most similar chunks
        scores, indices = index.search(query_embedding, min(top_k, len(chunks)))
        
        # Combine retrieved chunks into context
        retrieved_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
        return "\n\n".join(retrieved_chunks)

    def _load_icl_context(self, context_path: str) -> str:
        """Loads the full context for ICL."""
        

        df = pd.read_csv(context_path)
        context = "".join(df['text'].astype(str).tolist())
        
        # Check token count (for warning purposes)
        num_tokens = len(self.tokenizer.encode(context, add_special_tokens=False))
        logging.info(f"ICL context loaded: {num_tokens} tokens")
        
        return context

    def evaluate_phonebook(self, eval_data_path: str, baseline_type: str, context_path: Optional[str]):
        # Load data
        df_eval = pd.read_csv(eval_data_path)
        qa_data = []
        for _, row in df_eval.iterrows():
            match = re.search(r"Question: (.*?)\s*Answer: (.*)", row['text'])
            if match:
                qa_data.append({"question": match.group(1).strip(), "answer": match.group(2).strip()})

        # Prepare context
        context = ""
        index, chunks = None, None
        
        if baseline_type == "icl":
            context = self._load_icl_context(context_path)
        elif baseline_type == "rag":
            # RAG-C: Load FAISS index
            size_match = re.search(r'_(\d+K)\.csv', eval_data_path)
            size_str = size_match.group(1) if size_match else "3K"
            index, chunks, _ = self._load_rag_index("phonebook", size_str)
        
        # Sequential inference and timing
        correct = 0
        inference_times = []
        for item in tqdm(qa_data, desc=f"Evaluating PhoneBook {baseline_type}"):
            question = item['question']
            
            current_context = ""
            if baseline_type == 'base':
                user_message = f"Question: {question}\nAnswer:"
            else:
                if baseline_type == 'icl':
                    current_context = context
                elif baseline_type == 'rag':
                    current_context = self._retrieve_context_rag(question, index, chunks, top_k=3)
                user_message = f"Based on the context below, answer the question.\n\nContext:\n{current_context}\n\nQuestion: {question}\nAnswer:"

            inputs = self.tokenizer(
                user_message, 
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    do_sample=False, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()
            inference_times.append(end_time - start_time)

            pred_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Extract phone number pattern
            phone_pattern = r'\d{3}-\d{3}-\d{4}'
            phone_matches = re.findall(phone_pattern, pred_text)
            if phone_matches:
                pred_text = phone_matches[0]
            
            if "".join(re.findall(r'\d', pred_text)) == "".join(re.findall(r'\d', item['answer'])):
                correct += 1
        
        accuracy = (correct / len(qa_data)) * 100 if qa_data else 0.0
        avg_latency_ms = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0
        return {"accuracy": accuracy, "avg_inference_time_ms": avg_latency_ms}

    def evaluate_counterfact(self, size_str: str, baseline_type: str, context_path: Optional[str]):
        # Load data
        meta_path = os.path.join(PROJECT_ROOT, "data", "pb_cf", f"counterfact_edit_{size_str}_meta.json")
        master_json_path = os.path.join(PROJECT_ROOT, "data", "pb_cf", "counterfact.json")
        eval_dataset = build_eval_dataset(meta_path, master_json_path)

        # Prepare context
        context = ""
        index, chunks = None, None
        
        if baseline_type == "icl":
            context = self._load_icl_context(context_path)
        elif baseline_type == "rag":
            index, chunks, _ = self._load_rag_index("counterfact", size_str)

        # Sequential inference and timing
        num_success = 0
        inference_times = []
        for item in tqdm(eval_dataset, desc=f"Evaluating CounterFact {baseline_type}"):
            prompt_prefix = item["prompt_prefix"]
            targets = [item["target_new"], item["target_true"]]
            
            full_prompt = prompt_prefix
            if baseline_type == 'icl':
                full_prompt = f"Context:\n{context}\n\n{prompt_prefix}"
            elif baseline_type == 'rag':
                retrieved_context = self._retrieve_context_rag(prompt_prefix, index, chunks, top_k=3)
                full_prompt = f"Context:\n{retrieved_context}\n\n{prompt_prefix}"

            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            log_probs = get_next_token_log_probs(self.model, self.tokenizer, full_prompt, targets, self.device)
            if log_probs.get(targets[0], -float('inf')) > log_probs.get(targets[1], -float('inf')):
                num_success += 1
        
        efficacy_score = (num_success / len(eval_dataset)) * 100 if eval_dataset else 0.0
        avg_latency_ms = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0
        return {"efficacy_score": efficacy_score, "avg_inference_time_ms": avg_latency_ms}

def build_eval_dataset(meta_path: str, master_json_path: str) -> list:
   """Function to build dataset required for evaluation."""
   with open(meta_path, 'r', encoding='utf-8') as f:
       meta_data = json.load(f)
   subjects_to_eval = set(meta_data.get("subjects", []))

   with open(master_json_path, 'r', encoding='utf-8') as f:
       master_data = json.load(f)

   eval_set = []
   for record in master_data:
       rewrite_req = record.get("requested_rewrite", {})
       subject = rewrite_req.get("subject")
       if subject in subjects_to_eval:
           prompt_template = rewrite_req.get("prompt")
           target_new = rewrite_req.get("target_new", {}).get("str")
           target_true = rewrite_req.get("target_true", {}).get("str")

           if prompt_template and target_new and target_true:
               if '{}' in prompt_template:
                   prompt_template = prompt_template.replace('{}', '{subject}')
               prompt_prefix = prompt_template.format(subject=subject)
               eval_set.append({
                   "prompt_prefix": prompt_prefix,
                   "target_new": target_new,
                   "target_true": target_true,
               })
   return eval_set

def get_next_token_log_probs(model, tokenizer, prompt_prefix: str, targets: list, device):
   """Function to calculate log probabilities of target words."""
   inputs = tokenizer(
       prompt_prefix, 
       return_tensors="pt",
       truncation=True,
       max_length=tokenizer.model_max_length
   ).to(device)
   
   with torch.no_grad():
       outputs = model(**inputs)
       next_token_logits = outputs.logits[:, -1, :]
   log_probs = F.log_softmax(next_token_logits, dim=-1)
   
   target_log_probs = {}
   for target_str in targets:
       try:
           target_token_id = tokenizer.encode(target_str, add_special_tokens=False)[0]
           target_log_probs[target_str] = log_probs[0, target_token_id].item()
       except IndexError:
           target_log_probs[target_str] = -float('inf')
   return target_log_probs

def main():
   parser = argparse.ArgumentParser(description="[Llama] Fast baseline evaluation for PhoneBook & CounterFact.")
   parser.add_argument("--model_path", type=str, required=True)
   parser.add_argument("--dataset", type=str, required=True, choices=["phonebook", "counterfact"])
   parser.add_argument("--baseline_type", type=str, required=True, choices=["base", "icl", "rag"])
   parser.add_argument("--eval_data_path", type=str, required=True)
   parser.add_argument("--output_path", type=str, required=True)
   parser.add_argument("--context_path", type=str, default=None, help="Path to ICL context file or RAG source file.")
   parser.add_argument("--gpu_id", type=int, default=0)
   args = parser.parse_args()
   
   device = f"cuda:{args.gpu_id}"
   evaluator = BaselineEvaluatorLlama(args.model_path, device)
   
   results = {}
   if args.dataset == "phonebook":
       results = evaluator.evaluate_phonebook(args.eval_data_path, args.baseline_type, args.context_path)
   else: # counterfact
       size_match = re.search(r'_(\d+K)\.csv', args.eval_data_path)
       if not size_match:
           raise ValueError("Could not find size (e.g., 10K) in filename during CounterFact evaluation.")
       size_str = size_match.group(1)
       results = evaluator.evaluate_counterfact(size_str, args.baseline_type, args.context_path)

   output_dir = os.path.dirname(args.output_path)
   os.makedirs(output_dir, exist_ok=True)
   
   summary = {
       "experiment_name": os.path.basename(output_dir),
       "dataset": args.dataset,
       "baseline_type": args.baseline_type,
       **results
   }
   
   with open(args.output_path, "w") as f:
       json.dump(summary, f, indent=2)
   
   logging.info(f"Results saved: {args.output_path}")

if __name__ == "__main__":
   main()
