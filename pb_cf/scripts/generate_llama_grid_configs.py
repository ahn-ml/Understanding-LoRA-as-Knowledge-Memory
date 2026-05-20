#!/usr/bin/env python3
# scripts/generate_llama_grid_configs.py
"""
Llama 3.1 Grid Search 실험을 위한 YAML 설정 파일을 생성합니다.
- Instruct 모델 사용
- PhoneBook: Chat template 사용
- CounterFact: Chat template 미사용
"""

import os
import yaml
import logging
from itertools import product
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXP_CODENAME = "llama_grid"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_OUTPUT_DIR = PROJECT_ROOT / "configs" / "generated" / EXP_CODENAME
DATA_ROOT_DIR = PROJECT_ROOT / "data" / "pb_cf"

RANKS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SIZES_K = range(1, 21)
DATASETS = ["phonebook", "counterfact"]

CONFIG_TEMPLATE = """
experiment:
  name: "{exp_name}"
  seed: 42

resources:
  gpus: [0]

model:
  base_model_id: "{base_model_id}"
  lora_config:
    r: {rank}
    lora_alpha: {rank}
    target_modules: "all-linear"
    lora_dropout: 0.1
    bias: "none"
    task_type: "CAUSAL_LM"

dataset:
  name: "{dataset_name}"
  data_file_path: "{data_file_path}"
  use_chat_template: {use_chat_template}

training:
  method: "ntp"
  num_train_steps: 1500
  learning_rate: 5.0e-05
  batch_size: 8
  checkpoint_strategy:
    policy: "none"

evaluation:
  script: "{eval_script}"
"""

def generate_configs():
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    base_model_id = os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    
    total_configs = 0
    
    combinations = product(DATASETS, SIZES_K, RANKS)
    
    for dataset, size_k, rank in tqdm(list(combinations), desc="Generating Configs"):
        size_str = f"{size_k}K"
        
        if dataset == "phonebook":
            data_filename = f"phonebook_qa_{size_str}.csv"
            use_chat_template = True  # PhoneBook은 chat template 사용
            eval_script = "pb_cf/analysis/evaluate_checkpoints_llama.py"
        else:
            data_filename = f"counterfact_edit_{size_str}.csv"
            use_chat_template = False  # CounterFact는 chat template 미사용
            eval_script = "pb_cf/analysis/evaluate_counterfact_efficacy_llama.py"

        data_file_path = os.path.join(DATA_ROOT_DIR, data_filename)
        exp_name = f"{EXP_CODENAME}_{dataset}_ds{size_str}_r{rank}"
        
        config_str = CONFIG_TEMPLATE.format(
            exp_name=exp_name,
            rank=rank,
            base_model_id=base_model_id,
            dataset_name=dataset,
            data_file_path=str(Path(data_file_path).resolve()),
            use_chat_template=str(use_chat_template).lower(),
            eval_script=eval_script
        )
        
        config_data = yaml.safe_load(config_str)
        filepath = CONFIG_OUTPUT_DIR / f"{exp_name}.yaml"
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, sort_keys=False, indent=2)
            
        total_configs += 1
            
    logging.info(f"총 {total_configs}개의 설정 파일을 '{CONFIG_OUTPUT_DIR}'에 생성했습니다.")

if __name__ == "__main__":
    generate_configs()
