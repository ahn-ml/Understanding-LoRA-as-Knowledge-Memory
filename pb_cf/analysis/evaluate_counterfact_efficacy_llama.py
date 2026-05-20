#!/usr/bin/env python3
# analysis/evaluate_counterfact_efficacy_llama.py
"""
Evaluates the Efficacy of CounterFact LoRA checkpoints fine-tuned with Llama 3.1 (No VLLM).
- This script runs independently, loading the base model and LoRA adapters directly and merging them.
- Does not apply Chat Templates, adhering to CounterFact dataset policies.
- Measures Efficacy Score and average time taken for Log-Likelihood calculations.
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from tqdm import tqdm
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel

# Import internal project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from src.models.llama_loader import load_llama_model_and_tokenizer

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_eval_dataset(meta_path: str, master_json_path: str) -> list:
    """Constructs the dataset required for evaluation."""
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
    """
    Calculates log probabilities for target words.
    
    """
    inputs = tokenizer(prompt_prefix, return_tensors="pt").to(device)
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
    parser = argparse.ArgumentParser(description="[Llama] Evaluate CounterFact Efficacy sequentially.")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="final")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    # --article_index kept for compatibility
    parser.add_argument("--article_index", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    exp_name = os.path.basename(args.exp_dir)
    output_dir = os.path.dirname(args.output_path)
    
    try:
        # --- 1. Load Configuration and Evaluation Data ---
        config_path = os.path.join(args.exp_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_file = config.get('dataset', {}).get('data_file_path', '')
        size_match = re.search(r'_(\d+K)\.csv', data_file)
        if not size_match:
            raise ValueError("Could not find size (e.g., 10K) in data file name.")
        size_str = size_match.group(1)

        meta_path = os.path.join(PROJECT_ROOT, "data", "pb_cf", f"counterfact_edit_{size_str}_meta.json")
        master_json_path = os.path.join(PROJECT_ROOT, "data", "pb_cf", "counterfact.json")
        eval_dataset = build_eval_dataset(meta_path, master_json_path)

        # --- 2. Load Model and Merge LoRA ---
        base_model_id = config.get('model', {}).get('base_model_id', 'meta-llama/Llama-3.1-8B-Instruct')
        model, tokenizer = load_llama_model_and_tokenizer(
            model_id=base_model_id, device=device,
            use_chat_template=config.get('dataset', {}).get('use_chat_template', False)
        )
        
        lora_path = os.path.join(args.exp_dir, args.checkpoint_name)
        model = PeftModel.from_pretrained(model, lora_path)
        
        # 
        model = model.merge_and_unload()
        model.to(device)
        model.eval()

        # --- 3. Sequential Inference and Time Measurement ---
        num_success = 0
        inference_times = []
        for item in tqdm(eval_dataset, desc=f"Evaluating Efficacy {exp_name}"):
            prompt_prefix = item["prompt_prefix"]
            targets = [item["target_new"], item["target_true"]]
            
            start_time = time.time()
            log_probs = get_next_token_log_probs(model, tokenizer, prompt_prefix, targets, device)
            inference_times.append(time.time() - start_time)
            
            if log_probs.get(targets[0], -float('inf')) > log_probs.get(targets[1], -float('inf')):
                num_success += 1
                
        # --- 4. Final Result Calculation and Saving ---
        efficacy_score = (num_success / len(eval_dataset)) * 100 if eval_dataset else 0.0
        avg_latency_ms = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0

        train_summary_path = os.path.join(args.exp_dir, 'training_summary.json')
        trainable_params = None
        if os.path.exists(train_summary_path):
            with open(train_summary_path, 'r') as f:
                trainable_params = json.load(f).get('trainable_parameters')

        summary_data = {
            "experiment_name": exp_name, "efficacy_score": efficacy_score,
            "avg_inference_time_ms": avg_latency_ms, "trainable_parameters": trainable_params
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
                
        logging.info(f"Evaluation complete. Efficacy: {efficacy_score:.2f}%, Avg Inference Time: {avg_latency_ms:.2f}ms")

    except Exception as e:
        logging.error(f"Error occurred: {exp_name} | {e}", exc_info=True)

if __name__ == "__main__":
    main()
