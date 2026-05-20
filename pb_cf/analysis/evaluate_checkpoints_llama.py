#!/usr/bin/env python3
# analysis/evaluate_checkpoints_llama.py
"""
Evaluates Llama 3.1 PhoneBook LoRA checkpoints fine-tuned on the dataset.
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
import yaml
from peft import PeftModel

# Import internal project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from src.datasets.phonebook import PhoneBookLoader
from src.models.llama_loader import load_llama_model_and_tokenizer

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_phone_number(text: str) -> str:
    """Normalizes phone number answers for accuracy measurement."""
    return "".join(re.findall(r'\d', text))

def main():
    parser = argparse.ArgumentParser(description="[Llama] Evaluate PhoneBook checkpoints sequentially using Transformers.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory from the training run.")
    parser.add_argument("--checkpoint_name", type=str, default="final", help="Name of the checkpoint to evaluate.")
    parser.add_argument("--output_path", type=str, required=True, help="Full path to save the evaluation summary.json.")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use for this evaluation task.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    exp_name = os.path.basename(args.exp_dir)
    checkpoint_path = os.path.join(args.exp_dir, args.checkpoint_name)
    output_dir = os.path.dirname(args.output_path)
    
    logging.info(f"Starting Llama PhoneBook sequential evaluation: {exp_name} on {device}")

    try:
        # --- 1. Load Configuration and Evaluation Data ---
        config_path = os.path.join(args.exp_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_loader = PhoneBookLoader(config)
        test_data = data_loader.load_data_for_eval()

        # --- 2. Load Model and Tokenizer using Llama Loader ---
        base_model_id = config.get('model', {}).get('base_model_id', 'meta-llama/Llama-3.1-8B-Instruct')
        use_template = config.get('dataset', {}).get('use_chat_template', False)
        
        model, tokenizer = load_llama_model_and_tokenizer(
            model_id=base_model_id, 
            device=device,
            use_chat_template=use_template
        )

        

        logging.info(f"Merging LoRA checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        model.to(device)
        model.eval()

        # --- 3. Sequential Inference and Timing ---
        correct = 0
        inference_times = []
        raw_predictions = []
        
        for item in tqdm(test_data, desc=f"Evaluating {exp_name}"):
            user_message = f"Question: What is the phone number of {item['metadata']['name']}? Answer:"
            
            # Branch based on Chat template usage
            if use_template and tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": user_message}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = user_message
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_times.append(time.time() - start_time)

            pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Extract and clean phone number pattern
            # Extract only the first phone number pattern (XXX-XXX-XXXX format)
            

            phone_pattern = r'\d{3}-\d{3}-\d{4}'
            phone_matches = re.findall(phone_pattern, pred_text)
            
            if phone_matches:
                # Use the first matched phone number
                pred_text_clean = phone_matches[0]
            else:
                # If no pattern found, use the first line or full text
                pred_text_clean = pred_text.split('\n')[0].strip()
                # Remove "Answer:" prefix if present
                if pred_text_clean.startswith("Answer:"):
                    pred_text_clean = pred_text_clean.replace("Answer:", "").strip()
            
            # Check correctness (using cleaned text)
            ground_truth = item['target_answer']
            is_correct = normalize_phone_number(pred_text_clean) == normalize_phone_number(ground_truth)
            
            if is_correct:
                correct += 1
            
            # Save raw prediction (both raw and cleaned versions)
            raw_predictions.append({
                "name": item['metadata']['name'],
                "question": user_message,
                "ground_truth": ground_truth,
                "model_output_raw": pred_text,
                "model_output_cleaned": pred_text_clean,
                "is_correct": is_correct
            })

        # --- 4. Calculate and Save Final Results ---
        accuracy = (correct / len(test_data)) * 100 if test_data else 0.0
        avg_latency_ms = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0

        # Retrieve trainable parameters from training_summary.json
        train_summary_path = os.path.join(args.exp_dir, 'training_summary.json')
        trainable_params = None
        if os.path.exists(train_summary_path):
            with open(train_summary_path, 'r') as f:
                trainable_params = json.load(f).get('trainable_parameters')

        summary_data = {
            "experiment_name": exp_name, 
            "accuracy": accuracy,
            "avg_inference_time_ms": avg_latency_ms, 
            "trainable_parameters": trainable_params,
            "num_evaluated": len(test_data),
            "num_correct": correct
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save raw predictions
        raw_predictions_path = os.path.join(output_dir, "raw_predictions.jsonl")
        with open(raw_predictions_path, 'w', encoding='utf-8') as f:
            for pred in raw_predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        
        logging.info(f"Raw predictions saved: {raw_predictions_path}")
        logging.info(f"Evaluation complete. Accuracy: {accuracy:.2f}%, Avg Inference Time: {avg_latency_ms:.2f}ms")

    except Exception as e:
        logging.error(f"Error occurred: {exp_name} | {e}", exc_info=True)

if __name__ == "__main__":
    main()
