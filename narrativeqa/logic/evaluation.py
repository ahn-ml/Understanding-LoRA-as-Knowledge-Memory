# logic/evaluation.py

import json
import logging

import evaluate
import torch
from peft import PeftModel
from tqdm import tqdm

from logic.utils import load_and_prepare_model
from path_utils import OUTPUTS_DIR, resolve_with_root


def evaluate_model(config: dict):
    """Evaluates the performance of the trained LoRA model."""
    logging.info("Starting evaluation...")
    
    eval_config = config['evaluation']
    
    # 1. Load evaluation data
    eval_data = []
    eval_data_path = resolve_with_root(eval_config['eval_data_path'])
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
            
    # 2. Load base model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_and_prepare_model(config['base_model_id'], device)
    
    # 3. Apply trained LoRA adapter
    output_base = resolve_with_root(config.get('output_base_dir'), OUTPUTS_DIR)
    adapter_path = output_base / config['experiment_name'] / "final"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Could not find trained LoRA adapter: {adapter_path}")
        
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    logging.info(f"Successfully loaded LoRA adapter from '{adapter_path}'.")
    
    # 4. Generate answers and collect results
    predictions = []
    references = []
    
    for item in tqdm(eval_data, desc="Evaluating"):
        question = item['question']
        ref_answers = item['answers']
        
        # Add instructions to existing user prompt
        user_prompt = f"Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness.\n\nQuestion: {question}"
        
        messages = [{"role": "user", "content": user_prompt}]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=eval_config['max_new_tokens'],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part, excluding the prompt
        generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        predictions.append(generated_text.strip())
        references.append(ref_answers)
        
    # 5. Calculate ROUGE-L scores
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    logging.info(f"Evaluation results: {results}")
    
    # 6. Save final results
    output_dir = output_base / config['experiment_name']
    results_path = output_dir / "evaluation_results.json"

    final_output = {
        "scores": results,
        "evaluation_data": []
    }
    for i in range(len(predictions)):
        final_output["evaluation_data"].append({
            "question": eval_data[i]["question"],
            "prediction": predictions[i],
            "references": references[i]
        })

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Saved evaluation results to '{results_path}'.")