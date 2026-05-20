#!/usr/bin/env python3
# scripts/11_analyze_multilora_results.py

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from path_utils import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_hyperparameters(exp_name: str) -> dict:
    """Extract hyperparameters from experiment name"""
    hp = {}
    
    # Steps
    if 'steps' in exp_name:
        hp['steps'] = int(exp_name.split('steps')[1].split('_')[0])
    
    # Learning rate
    if 'lr' in exp_name:
        lr_str = exp_name.split('lr')[1].split('_')[0]
        lr_str = lr_str.replace('em', 'e-')
        hp['lr'] = float(lr_str)
    
    # LoRA rank
    if '_r' in exp_name:
        hp['rank'] = int(exp_name.split('_r')[1].split('_')[0])
    
    # LoRA alpha
    if '_a' in exp_name:
        hp['alpha'] = int(exp_name.split('_a')[1].split('_')[0])
    
    # Batch size
    if 'bs' in exp_name:
        hp['batch_size'] = int(exp_name.split('bs')[1].split('_')[0])
    
    return hp

def extract_eval_config(eval_name: str) -> dict:
    """Extract settings from evaluation config name"""
    config = {}
    
    # Top-k
    if 'top' in eval_name:
        config['top_k'] = int(eval_name.split('top')[1].split('_')[0])
    
    # Combination type
    for comb_type in ['linear', 'cat', 'ties', 'none']:
        if comb_type in eval_name:
            config['combination'] = comb_type
            break
    
    return config

def load_all_results():
    """Load all evaluation results"""
    results_base = OUTPUTS_DIR / "multi_lora" / "eval"
    
    all_results = []
    
    # Iterate through each training experiment directory
    for training_exp_dir in results_base.glob("exp_*"):
        if not training_exp_dir.is_dir():
            continue
        
        training_exp_name = training_exp_dir.name
        training_hp = extract_hyperparameters(training_exp_name)
        
        # Iterate through each evaluation result file
        for result_file in training_exp_dir.glob("eval_*_results.json"):
            eval_name = result_file.stem.replace('_results', '')
            eval_config = extract_eval_config(eval_name)
            
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                    result_entry = {
                        'training_exp': training_exp_name,
                        'eval_name': eval_name,
                        **training_hp,
                        **eval_config,
                        'rouge1': data['avg_scores']['rouge1'],
                        'rouge2': data['avg_scores']['rouge2'],
                        'rougeL': data['avg_scores']['rougeL'],
                        'rougeLsum': data['avg_scores']['rougeLsum']
                    }
                    all_results.append(result_entry)
                    
            except Exception as e:
                logging.error(f"Failed to load {result_file}: {e}")
    
    return pd.DataFrame(all_results)

def create_heatmap(df: pd.DataFrame):
    """Create Training HP x Evaluation Config Heatmap"""
    # Create pivot table (based on rougeL)
    pivot_table = df.pivot_table(
        values='rougeL',
        index='training_exp',
        columns='eval_name',
        aggfunc='mean'
    )
    
    # Draw heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'ROUGE-L'})
    plt.title('Multi-LoRA Performance Matrix\n(Training Experiment × Evaluation Configuration)')
    plt.xlabel('Evaluation Configuration')
    plt.ylabel('Training Experiment')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = OUTPUTS_DIR / "multi_lora" / "performance_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Heatmap saved to {output_path}")
    plt.close()

def analyze_by_hyperparameter(df: pd.DataFrame):
    """Analyze impact by each hyperparameter"""
    results = {}
    
    # Analyze by training hyperparameters
    for hp in ['steps', 'lr', 'rank', 'alpha', 'batch_size']:
        if hp in df.columns:
            hp_analysis = df.groupby(hp)['rougeL'].agg(['mean', 'std', 'count'])
            results[f'training_{hp}'] = hp_analysis
    
    # Analyze by evaluation settings
    for config in ['top_k', 'combination']:
        if config in df.columns:
            config_analysis = df.groupby(config)['rougeL'].agg(['mean', 'std', 'count'])
            results[f'eval_{config}'] = config_analysis
    
    return results

def find_best_configurations(df: pd.DataFrame, top_n: int = 5):
    """Find best performance configurations"""
    # Overall best performance
    best_overall = df.nlargest(top_n, 'rougeL')
    
    # Best training config per evaluation setting
    best_by_eval = {}
    for eval_name in df['eval_name'].unique():
        eval_df = df[df['eval_name'] == eval_name]
        if not eval_df.empty:
            best_by_eval[eval_name] = eval_df.nlargest(1, 'rougeL').iloc[0].to_dict()
    
    # Best evaluation config per training setting
    best_by_training = {}
    for training_exp in df['training_exp'].unique():
        training_df = df[df['training_exp'] == training_exp]
        if not training_df.empty:
            best_by_training[training_exp] = training_df.nlargest(1, 'rougeL').iloc[0].to_dict()
    
    return {
        'best_overall': best_overall.to_dict('records'),
        'best_by_eval': best_by_eval,
        'best_by_training': best_by_training
    }

def generate_report(df: pd.DataFrame):
    """Generate comprehensive report"""
    report = []
    
    report.append("="*60)
    report.append("MULTI-LORA EXPERIMENT ANALYSIS REPORT")
    report.append("="*60)
    
    # Basic Statistics
    report.append(f"\nBasic Statistics:")
    report.append(f"  Total experiments: {len(df)}")
    report.append(f"  Unique training configs: {df['training_exp'].nunique()}")
    report.append(f"  Unique eval configs: {df['eval_name'].nunique()}")
    report.append(f"  Average ROUGE-L: {df['rougeL'].mean():.4f} (±{df['rougeL'].std():.4f})")
    
    # Best Performance
    best_row = df.loc[df['rougeL'].idxmax()]
    report.append(f"\nBest Configuration:")
    report.append(f"  Training: {best_row['training_exp']}")
    report.append(f"  Evaluation: {best_row['eval_name']}")
    report.append(f"  ROUGE-L: {best_row['rougeL']:.4f}")
    
    # Hyperparameter Analysis
    hp_analysis = analyze_by_hyperparameter(df)
    
    report.append(f"\n🔬 Hyperparameter Analysis:")
    for hp_name, analysis_df in hp_analysis.items():
        if not analysis_df.empty:
            report.append(f"\n  {hp_name}:")
            for idx, row in analysis_df.iterrows():
                report.append(f"    {idx}: {row['mean']:.4f} (±{row['std']:.4f}, n={int(row['count'])})")
    
    # Best by Evaluation Setting
    if 'combination' in df.columns:
        report.append(f"\nBest by Combination Method:")
        for comb in df['combination'].unique():
            comb_df = df[df['combination'] == comb]
            if not comb_df.empty:
                best = comb_df.loc[comb_df['rougeL'].idxmax()]
                report.append(f"  {comb}: {best['rougeL']:.4f} ({best['training_exp']})")
    
    # Effect of Top-k
    if 'top_k' in df.columns:
        report.append(f"\n📍 Effect of Top-k:")
        top_k_analysis = df.groupby('top_k')['rougeL'].agg(['mean', 'std'])
        for k, row in top_k_analysis.iterrows():
            report.append(f"  k={k}: {row['mean']:.4f} (±{row['std']:.4f})")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)

def main():
    logging.info("Loading all Multi-LoRA evaluation results...")
    
    # Load results
    df = load_all_results()
    
    if df.empty:
        logging.error("No results found!")
        return
    
    logging.info(f"Loaded {len(df)} evaluation results")
    
    # Create heatmap
    create_heatmap(df)
    
    # Find best configurations
    best_configs = find_best_configurations(df)
    
    # Generate report
    report = generate_report(df)
    print(report)
    
    # Save results
    output_dir = OUTPUTS_DIR / "multi_lora"
    
    # Save DataFrame
    df.to_csv(output_dir / "all_results.csv", index=False)
    logging.info(f"Results saved to {output_dir / 'all_results.csv'}")
    
    # Save best configurations
    with open(output_dir / "best_configurations.json", 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    # Save report
    with open(output_dir / "analysis_report.txt", 'w') as f:
        f.write(report)
    
    # Save summary statistics
    summary_stats = {
        'total_experiments': len(df),
        'avg_rouge1': float(df['rouge1'].mean()),
        'avg_rouge2': float(df['rouge2'].mean()),
        'avg_rougeL': float(df['rougeL'].mean()),
        'avg_rougeLsum': float(df['rougeLsum'].mean()),
        'best_rougeL': float(df['rougeL'].max()),
        'best_config': df.loc[df['rougeL'].idxmax()].to_dict()
    }
    
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logging.info(f"\nAnalysis complete!")
    logging.info(f"  - CSV: {output_dir / 'all_results.csv'}")
    logging.info(f"  - Heatmap: {output_dir / 'performance_heatmap.png'}")
    logging.info(f"  - Report: {output_dir / 'analysis_report.txt'}")
    logging.info(f"  - Best configs: {output_dir / 'best_configurations.json'}")

if __name__ == "__main__":
    main()