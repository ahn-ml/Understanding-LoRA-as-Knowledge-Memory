# scripts/10_evaluate_multilora.py

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

class MultiLoRAEvaluator:
    def __init__(self, training_exp_name: str, eval_config: dict, target_docs: list[str], base_model_id: str):
        self.training_exp_name = training_exp_name
        self.eval_config = eval_config
        self.top_k = eval_config['top_k']
        self.combination_type = eval_config.get('combination_type', 'none')
        self.base_model_id = base_model_id
        
        self.lora_base_dir = OUTPUTS_DIR / "multi_lora" / "training" / training_exp_name
        self.output_dir = OUTPUTS_DIR / "multi_lora" / "eval" / training_exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Embedding model
        self.embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
        
        self.target_docs = target_docs
        
        # Load chunk mapping per document
        self.doc_chunks = {}
        chunks_dir = DATA_DIR / "multi_lora" / "chunks"
        
        for doc_id in self.target_docs:
            doc_chunks_file = chunks_dir / f"doc_{doc_id}" / "chunks.json"
            if doc_chunks_file.exists():
                with open(doc_chunks_file, 'r') as f:
                    chunks = json.load(f)
                    self.doc_chunks[doc_id] = [chunk['global_chunk_id'] for chunk in chunks]
                    logging.info(f"Loaded {len(chunks)} chunks for doc {doc_id}")

    def _load_summaries_for_doc(self, doc_id: str):
        """Load summaries for a specific document and create index."""
        if doc_id not in self.doc_chunks:
            logging.error(f"No chunks found for doc {doc_id}")
            return None
        
        doc_chunk_ids = self.doc_chunks[doc_id]
        
        # Load summary for the document
        summaries_path = DATA_DIR / "multi_lora" / "summaries" / f"doc_{doc_id}_summaries.json"
        if not summaries_path.exists():
            logging.error(f"Summaries not found: {summaries_path}")
            return None
            
        with open(summaries_path, 'r') as f:
            summaries_dict = json.load(f)
        
        # Sort by chunk ID
        doc_summaries = []
        chunk_mapping = []
        
        for chunk_id in sorted(doc_chunk_ids):
            if str(chunk_id) in summaries_dict:
                doc_summaries.append(summaries_dict[str(chunk_id)])
                chunk_mapping.append(chunk_id)
        
        logging.info(f"Doc {doc_id}: Creating index for {len(doc_summaries)} summaries")
        
        if not doc_summaries:
            return None
        
        # Create embedding and index for this document
        embeddings = self.embed_model.encode(doc_summaries, convert_to_tensor=False, normalize_embeddings=True)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        
        return index, chunk_mapping
        
    def retrieve_chunks(self, question: str) -> tuple:
        """Search Top-k chunks for the question."""
        query_embedding = self.embed_model.encode([question], convert_to_tensor=False, normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_embedding).astype('float32'), self.top_k)
    
        # Convert index to actual chunk_id
        actual_chunk_ids = [self.chunk_id_mapping[idx] for idx in indices[0].tolist()]
        return actual_chunk_ids, scores[0].tolist()
    
    def merge_loras(self, model, chunk_ids: list, scores: list):
        """Merge multiple LoRA adapters."""
        if self.combination_type == 'none':
            # Use Top-1 only
            chunk_id = chunk_ids[0]
            adapter_name = f"chunk_{chunk_id}"
            adapter_path = self.lora_base_dir / f"{self.training_exp_name}_chunk_{chunk_id}" / "final"
            
            if not adapter_path.exists():
                logging.error(f"Adapter not found: {adapter_path}")
                return False
            
            model.load_adapter(adapter_path, adapter_name)
            model.set_adapter(adapter_name)
            
        elif self.combination_type == 'linear':
            # Linear merge
            adapter_names = []
            for chunk_id in chunk_ids:
                adapter_name = f"chunk_{chunk_id}"
                adapter_path = self.lora_base_dir / f"{self.training_exp_name}_chunk_{chunk_id}" / "final"
                
                if adapter_path.exists():
                    model.load_adapter(adapter_path, adapter_name)
                    adapter_names.append(adapter_name)
            
            if len(adapter_names) > 0:
                weights = self.eval_config.get('weights', [1/len(adapter_names)] * len(adapter_names))
                model.add_weighted_adapter(
                    adapters=adapter_names[:len(weights)],
                    weights=weights[:len(adapter_names)],
                    combination_type="linear",
                    adapter_name="merged"
                )
                model.set_adapter("merged")
            
        elif self.combination_type == 'cat':
            # Concatenation
            adapter_names = []
            for chunk_id in chunk_ids:
                adapter_name = f"chunk_{chunk_id}"
                adapter_path = self.lora_base_dir / f"{self.training_exp_name}_chunk_{chunk_id}" / "final"
                
                if adapter_path.exists():
                    model.load_adapter(adapter_path, adapter_name)
                    adapter_names.append(adapter_name)
            
            if len(adapter_names) > 0:
                model.add_weighted_adapter(
                    adapters=adapter_names,
                    weights=[1] * len(adapter_names),
                    combination_type="cat",
                    adapter_name="merged"
                )
                model.set_adapter("merged")
                
        elif self.combination_type == 'ties':
            # TIES merge
            adapter_names = []
            for chunk_id in chunk_ids:
                adapter_name = f"chunk_{chunk_id}"
                adapter_path = self.lora_base_dir / f"{self.training_exp_name}_chunk_{chunk_id}" / "final"
                
                if adapter_path.exists():
                    model.load_adapter(adapter_path, adapter_name)
                    adapter_names.append(adapter_name)
            
            if len(adapter_names) > 0:
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
    
    def evaluate_document(self, doc_id: str, model, tokenizer, device):
        """Evaluate a single document."""
        
        # Create index for this document
        result = self._load_summaries_for_doc(doc_id)
        if result is None:
            return None
        
        doc_index, doc_chunk_mapping = result
        
        # Find eval file
        eval_dir = DATA_DIR / "multi_doc" / "eval"
        eval_path = None
        for file in eval_dir.glob("*.jsonl"):
            if doc_id in file.name:
                eval_path = file
                break
        
        if not eval_path:
            logging.error(f"Eval data not found for doc {doc_id}")
            return None
        
        logging.info(f"Using eval file: {eval_path.name}")
        
        eval_data = []
        with open(eval_path, 'r') as f:
            for line in f:
                eval_data.append(json.loads(line))
        
        predictions = []
        references = []
        retrieval_log = []
        
        for item in tqdm(eval_data, desc=f"Evaluating {doc_id}"):
            question = item['question']
            
            # Search only within this document's chunks
            query_embedding = self.embed_model.encode([question], convert_to_tensor=False, normalize_embeddings=True)
            scores, indices = doc_index.search(np.array(query_embedding).astype('float32'), self.top_k)
            
            # Convert to actual chunk_id
            chunk_ids = [doc_chunk_mapping[idx] for idx in indices[0].tolist()]
            
            retrieval_log.append({
                "question": question,
                "retrieved_chunks": chunk_ids,
                "scores": scores[0].tolist()
            })
            
            # Merge LoRA
            success = self.merge_loras(model, chunk_ids, scores[0].tolist())
            if not success:
                predictions.append("")
                references.append(item['answers'])
                continue
            
            # Generate answer
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            predictions.append(generated_text.strip())
            references.append(item['answers'])
            
            # Clean up adapters
            if self.combination_type != 'none':
                if "merged" in model.peft_config:
                    model.delete_adapter("merged")
            for cid in chunk_ids:
                adapter_name = f"chunk_{cid}"
                if adapter_name in model.peft_config:
                    model.delete_adapter(adapter_name)
        
        # Calculate ROUGE
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
        
        return {
            'doc_id': doc_id,
            'scores': results,
            'predictions': predictions,
            'references': references,
            'retrieval_log': retrieval_log
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_exp', type=str, required=True, help='Training experiment name')
    parser.add_argument('--eval_config', type=Path, required=True, help='Evaluation config path')
    parser.add_argument('--doc_ids', type=Path, default=DATA_DIR / "doc_ids.json", help='JSON list of doc_ids to evaluate')
    parser.add_argument('--base_model', type=str, default=str(PROJECT_ROOT / "models" / "Llama-3.1-8B-Instruct"), help='Base model path or identifier')
    args = parser.parse_args()
    
    with open(args.eval_config, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    if args.doc_ids.exists():
        with open(args.doc_ids, 'r') as f:
            doc_ids = json.load(f)
    else:
        raise FileNotFoundError(f"Doc ID file not found: {args.doc_ids}")
    
    evaluator = MultiLoRAEvaluator(args.training_exp, eval_config, doc_ids, args.base_model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, tokenizer = load_and_prepare_model(args.base_model, device)
    
    first_lora = next(evaluator.lora_base_dir.glob("*/final"))
    model = PeftModel.from_pretrained(base_model, first_lora, "temp")
    model.delete_adapter("temp")
    model.eval()
    
    all_results = []
    
    for doc_id in doc_ids:
        logging.info(f"\nEvaluating document: {doc_id}")
        result = evaluator.evaluate_document(doc_id, model, tokenizer, device)
        if result:
            all_results.append(result)
    
    eval_name = args.eval_config.stem
    output_path = evaluator.output_dir / f"{eval_name}_results.json"
    
    avg_scores = {}
    if all_results:
        for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            avg_scores[metric] = np.mean([r['scores'][metric] for r in all_results])
    
    final_output = {
        'training_exp': args.training_exp,
        'eval_config': eval_config,
        'avg_scores': avg_scores,
        'document_results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    logging.info(f"\nEvaluation complete!")
    logging.info(f"Average ROUGE-L: {avg_scores.get('rougeL', 0):.4f}")
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()