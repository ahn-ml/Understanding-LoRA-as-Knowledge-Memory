# /scripts/13_evaluate_single_doc.py

import argparse
import json
import logging
import sys
from pathlib import Path

import evaluate
import faiss
import numpy as np
import torch
import yaml
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from logic.utils import load_and_prepare_model  # noqa: E402
from path_utils import DATA_DIR, OUTPUTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SingleDocEvaluator:
    def __init__(self, doc_id: str, training_exp: str, eval_config: dict):
        self.doc_id = doc_id
        self.doc_short = doc_id[:8]
        self.training_exp = training_exp
        self.eval_config = eval_config
        self.top_k = eval_config['top_k']
        self.combination_type = eval_config.get('combination_type', 'none')
        
        # Setup paths
        self.lora_base_dir = OUTPUTS_DIR / "multi_lora" / training_exp
        self.output_dir = OUTPUTS_DIR / "multi_lora" / f"{training_exp}_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Embedding model
        self.embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
        
        # Load chunks for this document
        chunks_file = DATA_DIR / "multi_lora" / "chunks" / f"doc_{doc_id}" / "chunks.json"
        with open(chunks_file) as f:
            chunks = json.load(f)
            self.doc_chunks = [c['global_chunk_id'] for c in chunks]
        
        # Load summary and create index
        self._setup_retrieval()
    
    def _setup_retrieval(self):
        """Setup summary index for this document."""
        summaries_file = DATA_DIR / "multi_lora" / "summaries" / f"doc_{self.doc_id}_summaries.json"
        with open(summaries_file) as f:
            summaries_dict = json.load(f)
        
        self.summaries = []
        self.chunk_mapping = []
        
        for chunk_id in sorted(self.doc_chunks):
            if str(chunk_id) in summaries_dict:
                self.summaries.append(summaries_dict[str(chunk_id)])
                self.chunk_mapping.append(chunk_id)
        
        # Generate embeddings
        embeddings = self.embed_model.encode(self.summaries, convert_to_tensor=False, normalize_embeddings=True)
        
        # FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
    
    def retrieve(self, question: str):
        """Retrieve Top-k chunks."""
        query_embedding = self.embed_model.encode([question], convert_to_tensor=False, normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_embedding).astype('float32'), self.top_k)
        chunk_ids = [self.chunk_mapping[idx] for idx in indices[0]]
        return chunk_ids, scores[0].tolist()
    
    def merge_loras(self, model, chunk_ids: list):
        """Merge LoRAs (logic consistent with other scripts)."""
        if self.combination_type == 'none':
            chunk_id = chunk_ids[0]
            adapter_name = f"chunk_{chunk_id}"
            adapter_path = self.lora_base_dir / f"{self.training_exp}_chunk_{chunk_id}" / "final"
            
            if adapter_path.exists():
                model.load_adapter(adapter_path, adapter_name)
                model.set_adapter(adapter_name)
                return True
                
        elif self.combination_type == 'ties':
            adapter_names = []
            for chunk_id in chunk_ids:
                adapter_name = f"chunk_{chunk_id}"
                adapter_path = self.lora_base_dir / f"{self.training_exp}_chunk_{chunk_id}" / "final"
                
                if adapter_path.exists():
                    model.load_adapter(adapter_path, adapter_name)
                    adapter_names.append(adapter_name)
            
            if adapter_names:
                model.add_weighted_adapter(
                    adapters=adapter_names,
                    weights=[1/len(adapter_names)] * len(adapter_names),
                    combination_type="ties",
                    adapter_name="merged",
                    density=self.eval_config.get('density', 0.5),
                    majority_sign_method=self.eval_config.get('majority_sign_method', 'total')
                )
                model.set_adapter("merged")
                return True
        
        return False
    
    def evaluate(self, model, tokenizer, device):
        """Evaluate document."""
        # Load Eval data
        eval_file = None
        eval_dir = DATA_DIR / "multi_doc" / "eval"
        for f in eval_dir.glob("*.jsonl"):
            if self.doc_short in f.name or self.doc_id in f.name:
                eval_file = f
                break
        
        if not eval_file:
            logging.error(f"No eval file for {self.doc_id}")
            return None
        
        eval_data = []
        with open(eval_file) as f:
            for line in f:
                eval_data.append(json.loads(line))
        
        predictions = []
        references = []
        retrieval_log = []
        
        for item in tqdm(eval_data, desc=f"Eval {self.doc_short}"):
            question = item['question']
            
            # Retrieval
            chunk_ids, scores = self.retrieve(question)
            retrieval_log.append({
                "question": question,
                "retrieved_chunks": chunk_ids,
                "scores": scores
            })
            
            # Merge LoRA
            if not self.merge_loras(model, chunk_ids):
                predictions.append("")
                references.append(item['answers'])
                continue
            
            # Generation
            user_prompt = f"Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness.\n\nQuestion: {question}"
            messages = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            predictions.append(text.strip())
            references.append(item['answers'])
            
            # Cleanup adapters
            if self.combination_type != 'none' and "merged" in model.peft_config:
                model.delete_adapter("merged")
            for cid in chunk_ids:
                name = f"chunk_{cid}"
                if name in model.peft_config:
                    model.delete_adapter(name)
        
        # ROUGE calculation
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
        
        return {
            'doc_id': self.doc_id,
            'scores': results,
            'predictions': predictions,
            'references': references,
            'retrieval_log': retrieval_log
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize Evaluator
    evaluator = SingleDocEvaluator(
        config['doc_id'],
        config['training_exp'],
        config
    )
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = config.get('base_model_id', str(PROJECT_ROOT / "models" / "Llama-3.1-8B-Instruct"))
    base_model, tokenizer = load_and_prepare_model(base_model_id, device)
    
    # Initialize PeftModel (with temporary adapter)
    temp_lora = next((OUTPUTS_DIR / "multi_lora" / config['training_exp']).glob("*/final"))
    model = PeftModel.from_pretrained(base_model, temp_lora, "temp")
    model.delete_adapter("temp")
    model.eval()
    
    # Evaluation
    result = evaluator.evaluate(model, tokenizer, device)
    
    if result:
        # Save results
        method = config.get('combination_type', 'none')
        if method == 'ties':
            method_name = f"top{config['top_k']}ties"
        else:
            method_name = f"top{config['top_k']}"
        
        output_file = evaluator.output_dir / f"{config['doc_short']}_{method_name}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Saved: {output_file}")
        print(f"ROUGE-L: {result['scores']['rougeL']:.4f}")

if __name__ == "__main__":
    main()