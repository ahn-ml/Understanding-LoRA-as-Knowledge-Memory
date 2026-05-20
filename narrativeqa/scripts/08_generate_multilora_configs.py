# scripts/08_generate_multilora_configs.py

import json
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import CONFIGS_DIR, DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_multilora_configs():
    """Generate config files for Multi-LoRA experiments."""
    
    # Base configuration
    base_config = {
        'base_model_id': os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        'seed': 42,
        'api_config': {
            'api_version': '2024-10-21',
            'deployment_name': 'gpt-4.1'
        },
        'data_prep': {
            'initial_qa_count': 40,
            'refinement_iterations': 2
        },
        'training': {
            'task_type': 'qa',
            'mask_question': False
        },
        'evaluation': {
            'max_new_tokens': 50,
            'use_instruction': False
        }
    }
    
    # Training hyperparameter experiments
    training_experiments = [
        # Baseline
        {'num_train_steps': 250, 'learning_rate': 5.0e-4, 'lora_rank': 4, 'lora_alpha': 8, 'batch_size': 32, 'warmup_ratio': 0.1},
        
        # Adjust steps
        {'num_train_steps': 150, 'learning_rate': 5.0e-4, 'lora_rank': 4, 'lora_alpha': 8, 'batch_size': 32, 'warmup_ratio': 0.1},
        {'num_train_steps': 350, 'learning_rate': 5.0e-4, 'lora_rank': 4, 'lora_alpha': 8, 'batch_size': 32, 'warmup_ratio': 0.1},
        
        # Adjust learning rate
        {'num_train_steps': 250, 'learning_rate': 3.0e-4, 'lora_rank': 4, 'lora_alpha': 8, 'batch_size': 32, 'warmup_ratio': 0.1},
        {'num_train_steps': 250, 'learning_rate': 7.0e-4, 'lora_rank': 4, 'lora_alpha': 8, 'batch_size': 32, 'warmup_ratio': 0.1},
        
        # Adjust Rank/Alpha ratio
        {'num_train_steps': 250, 'learning_rate': 5.0e-4, 'lora_rank': 4, 'lora_alpha': 4, 'batch_size': 32, 'warmup_ratio': 0.1},
        {'num_train_steps': 250, 'learning_rate': 5.0e-4, 'lora_rank': 4, 'lora_alpha': 16, 'batch_size': 32, 'warmup_ratio': 0.1},
    ]
    
    # Load chunk metadata
    chunks_metadata_path = DATA_DIR / "multi_lora" / "chunks" / "all_chunks_metadata.json"
    with open(chunks_metadata_path, 'r') as f:
        metadata = json.load(f)
        chunks = metadata['chunks']
    
    logging.info(f"Loaded {len(chunks)} chunks metadata")
    
    # For each training experiment
    for exp_idx, hp_set in enumerate(training_experiments):
        # Generate experiment name
        exp_name = f"exp_{exp_idx:02d}"
        for key, value in hp_set.items():
            if key == 'num_train_steps':
                exp_name += f"_steps{value}"
            elif key == 'learning_rate':
                exp_name += f"_lr{str(value).replace('0.', '').replace('-', 'm')}"
            elif key == 'lora_rank':
                exp_name += f"_r{value}"
            elif key == 'lora_alpha':
                exp_name += f"_a{value}"
            elif key == 'batch_size':
                exp_name += f"_bs{value}"
        
        # Create experiment directory
        exp_dir = CONFIGS_DIR / 'multi_lora' / 'training' / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\nGenerating configs for {exp_name}")
        logging.info(f"  Hyperparameters: {hp_set}")
        
        # Generate config for each chunk
        for chunk_data in chunks:
            chunk_id = chunk_data['global_chunk_id']
            doc_short = chunk_data['doc_short']
            
            config = base_config.copy()
            config['api_config'] = base_config['api_config'].copy()
            config['data_prep'] = base_config['data_prep'].copy()
            config['training'] = base_config['training'].copy()
            config['evaluation'] = base_config['evaluation'].copy()
            
            # Chunk-specific settings
            config['experiment_name'] = f"{exp_name}_chunk_{chunk_id}"
            config['doc_id'] = chunk_data['doc_id']
            config['chunk_id'] = chunk_id
            config['output_base_dir'] = f'runs/multi_lora/training/{exp_name}'
            config['log_dir'] = f'logs/multi_lora/training/{exp_name}'
            config['data_dir'] = 'data/multi_lora'
            
            # Path settings
            config['data_prep']['gold_summary_path'] = "data/multi_lora/summaries/all_chunk_summaries.json"
            config['training']['data_path'] = f"data/multi_lora/qa/chunk_{chunk_id}_qa.jsonl"
            
            # Evaluation data is at document level
            config['evaluation']['eval_data_path'] = f"data/multi_doc/eval/doc_{doc_short}_eval.jsonl"
            
            # Apply hyperparameters
            config['training'].update(hp_set)
            
            # Save config
            config_path = exp_dir / f"chunk_{chunk_id}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        logging.info(f"  Generated {len(chunks)} chunk configs")
    
    # Generate evaluation configs (separate)
    eval_configs = [
        {'top_k': 1, 'combination_type': 'none'},  # Use top-1 without merging
        {'top_k': 3, 'combination_type': 'linear', 'weights': [1/3, 1/3, 1/3]},
        {'top_k': 5, 'combination_type': 'linear', 'weights': [0.3, 0.25, 0.2, 0.15, 0.1]},
        {'top_k': 3, 'combination_type': 'cat'},
        {'top_k': 3, 'combination_type': 'ties', 'density': 0.5, 'majority_sign_method': 'total'},
    ]
    
    eval_dir = CONFIGS_DIR / 'multi_lora' / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    for eval_idx, eval_config in enumerate(eval_configs):
        eval_name = f"eval_{eval_idx:02d}_top{eval_config['top_k']}_{eval_config['combination_type']}"
        eval_path = eval_dir / f"{eval_name}.yaml"
        
        with open(eval_path, 'w') as f:
            yaml.dump(eval_config, f, default_flow_style=False)
    
    logging.info(f"\nTotal training experiments: {len(training_experiments)}")
    logging.info(f"Total eval configurations: {len(eval_configs)}")
    logging.info(f"Total configs generated: {len(training_experiments) * len(chunks) + len(eval_configs)}")

if __name__ == "__main__":
    generate_multilora_configs()
