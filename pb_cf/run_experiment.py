#!/usr/bin/env python3
# run_experiment.py - Experiment Orchestration Script (Process Call Only)
import os
import yaml
import logging
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging, set_seed

def main(config_path: str, train_only: bool = False, eval_only: bool = False):
    """Experiment Orchestration: Runs training and evaluation as separate processes."""
    
    # 1. Load configuration and set defaults
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    exp_name = config['experiment']['name']

    output_dir = config.get('output_base_dir')
    if not output_dir:
        try:
            exp_group = os.path.basename(os.path.dirname(config_path))
        except Exception:
            exp_group = "default_exp"
        output_dir = os.path.join('runs', 'pb_cf', exp_group, exp_name)
    else:
        output_dir = os.path.join(output_dir, exp_name)

    os.makedirs(output_dir, exist_ok=True)

    config['output_dir'] = output_dir


    shutil.copy(config_path, os.path.join(output_dir, "config.yaml"))
    
    setup_logging(output_dir)
    set_seed(config['experiment']['seed'])

    logging.info(f"===== Experiment Started: {exp_name} =====")
    logging.info(f"All results will be saved to: {output_dir}")

    # 2. Execute Training (Separate Process)
    if not eval_only:
        logging.info("===== Training Process Started =====")
        train_command = [sys.executable, str(SCRIPT_DIR / 'train.py'), '--config', config_path]
        
        logging.info(f"Executing training command: {' '.join(train_command)}")
        
        try:

            result = subprocess.run(train_command, check=True)
            
            logging.info("Training completed!")
        except subprocess.CalledProcessError as e:
            # Since stdout/stderr are streamed, log the error code specifically.
            logging.error(f"Error during training process! Return code: {e.returncode}")
            logging.error("Check logs above for details.")
            return
        except FileNotFoundError:
            logging.error("train.py script not found.")
            return

    # 3. Execute Evaluation (Separate Process)
    if not train_only and 'evaluation' in config and 'script' in config['evaluation']:
        logging.info("===== Evaluation Process Started =====")
        eval_script_path = Path(config['evaluation']['script'])
        if not eval_script_path.is_absolute():
            eval_script_path = PROJECT_ROOT / eval_script_path
        
        # Unify evaluation result summary filename
        eval_summary_path = os.path.join(output_dir, "evaluation_results.json")
        
        # Construct arguments for evaluation script
        # Since gpu_id is managed by CUDA_VISIBLE_DEVICES, always use 0 inside the script.
        eval_command = [
            sys.executable, str(eval_script_path),
            '--exp_dir', output_dir,
            '--output_path', eval_summary_path,
            '--gpu_id', '0' 
        ]
        
        logging.info(f"Executing evaluation command: {' '.join(eval_command)}")
        
        try:

            result = subprocess.run(eval_command, check=True)

            logging.info(f"Evaluation completed! Results saved to {eval_summary_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error during evaluation process! Return code: {e.returncode}")
            logging.error("Check logs above for details.")
        except FileNotFoundError:
            logging.error(f"Evaluation script not found: {eval_script_path}")

    logging.info(f"===== Experiment Finished: {exp_name} =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Execution Framework (Process Orchestration)")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file")
    parser.add_argument("--train-only", action="store_true", help="Run training only")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    
    if args.train_only and args.eval_only:
        print("--train-only and --eval-only cannot be used simultaneously.")
        sys.exit(1)
    
    main(args.config, train_only=args.train_only, eval_only=args.eval_only)
