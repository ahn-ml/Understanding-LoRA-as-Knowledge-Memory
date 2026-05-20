# /scripts/12_generate_final_multilora_configs.py

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import CONFIGS_DIR, DATA_DIR


def generate_final_configs(doc_ids_path: Path):
    with open(doc_ids_path, 'r') as f:
        doc_ids = json.load(f)
    
    # Optimal hyperparameters
    optimal_hp = {
        'num_train_steps': 150,
        'learning_rate': 5.0e-4,
        'lora_rank': 4,
        'lora_alpha': 8,
        'batch_size': 32,
        'warmup_ratio': 0.1
    }
    
    # Load chunk metadata
    chunks_dir = DATA_DIR / 'multi_lora' / 'chunks'
    all_chunks = []
    
    for doc_id in doc_ids:
        chunk_file = chunks_dir / f'doc_{doc_id}' / 'chunks.json'
        if chunk_file.exists():
            with open(chunk_file) as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
    
    print(f"Total chunks to train: {len(all_chunks)}")
    
    # Generate training configs
    config_dir = CONFIGS_DIR / 'multi_lora' / 'final'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    base_config = {
        'base_model_id': os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        'seed': 42,
        'training': {
            'task_type': 'qa',
            'mask_question': False,
            **optimal_hp
        },
        'output_base_dir': 'runs/multi_lora/final',
        'log_dir': 'logs/multi_lora/final',
        'data_dir': 'data/multi_lora'
    }
    
    # Save config for each chunk
    for chunk in all_chunks:
        chunk_id = chunk['global_chunk_id']
        config = base_config.copy()
        config['experiment_name'] = f'final_chunk_{chunk_id}'
        config['chunk_id'] = chunk_id
        config['doc_id'] = chunk['doc_id']
        config['training'] = base_config['training'].copy()
        config['training']['data_path'] = f'data/multi_lora/qa/chunk_{chunk_id}_qa.jsonl'
        
        config_path = config_dir / f'chunk_{chunk_id}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Generate evaluation configs
    eval_dir = CONFIGS_DIR / 'multi_lora' / 'final_eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate configs per document x evaluation method
    for doc_id in doc_ids:
        doc_short = doc_id[:8]
        
        # Top-1 config
        config_top1 = {
            'doc_id': doc_id,
            'doc_short': doc_short,
            'top_k': 1,
            'combination_type': 'none',
            'training_exp': 'final'
        }
        with open(eval_dir / f'{doc_short}_top1.yaml', 'w') as f:
            yaml.dump(config_top1, f)
        
        # Top-3 TIES config
        config_ties = {
            'doc_id': doc_id,
            'doc_short': doc_short,
            'top_k': 3,
            'combination_type': 'ties',
            'density': 0.5,
            'majority_sign_method': 'total',
            'training_exp': 'final'
        }
        with open(eval_dir / f'{doc_short}_top3ties.yaml', 'w') as f:
            yaml.dump(config_ties, f)
    
    print(f"Generated {len(all_chunks)} training configs")
    print(f"Generated {len(doc_ids)*2} eval configs (40 docs × 2 methods)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final multi-LoRA config generator")
    parser.add_argument("--doc-ids", type=Path, default=DATA_DIR / "doc_ids.json", help="Path to doc_ids JSON")
    args = parser.parse_args()
    generate_final_configs(args.doc_ids)
