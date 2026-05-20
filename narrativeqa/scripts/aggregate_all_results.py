# /scripts/aggregate_all_results.py

import json
from pathlib import Path

import pandas as pd

from path_utils import DATA_DIR, OUTPUTS_DIR


def aggregate_all_results():
    """Integrated analysis of LoRA (80) + ICL (80) results."""
    
    # 1. Collect all doc_ids
    all_doc_ids = []
    
    # Original 40
    if (DATA_DIR / "doc_ids_original.json").exists():
        with open(DATA_DIR / "doc_ids_original.json", 'r') as f:
            all_doc_ids.extend(json.load(f))
    
    # Medium 40
    with open(DATA_DIR / "doc_ids.json", 'r') as f:
        all_doc_ids.extend(json.load(f))
    
    print(f"Total documents: {len(all_doc_ids)}")
    
    # 2. Collect LoRA results
    lora_results = []
    for doc_id in all_doc_ids:
        doc_short = doc_id[:8]
        result_path = OUTPUTS_DIR / f"multi_doc/doc_{doc_short}_iterative_qa/evaluation_results.json"
        
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
                lora_results.append({
                    'doc_id': doc_id,
                    'method': 'LoRA',
                    'rouge1': data['scores']['rouge1'],
                    'rouge2': data['scores']['rouge2'],
                    'rougeL': data['scores']['rougeL'],
                    'rougeLsum': data['scores']['rougeLsum']
                })
    
    # 3. Collect ICL results
    icl_results = []
    for doc_id in all_doc_ids:
        doc_short = doc_id[:8]
        result_path = OUTPUTS_DIR / f"icl_results/icl_{doc_short}_results.json"
        
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
                icl_results.append({
                    'doc_id': doc_id,
                    'method': 'ICL',
                    'rouge1': data['scores']['rouge1'],
                    'rouge2': data['scores']['rouge2'],
                    'rougeL': data['scores']['rougeL'],
                    'rougeLsum': data['scores']['rougeLsum']
                })
    
    # 4. Combined DataFrame
    all_results = pd.DataFrame(lora_results + icl_results)
    
    # 5. Statistical Analysis
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: LoRA vs ICL")
    print("="*60)
    
    # Average by Method
    print("\nAverage Scores by Method:")
    method_avg = all_results.groupby('method')[['rouge1', 'rouge2', 'rougeL', 'rougeLsum']].mean()
    print(method_avg.round(4))
    
    # Standard Deviation by Method
    print("\nStandard Deviation by Method:")
    method_std = all_results.groupby('method')[['rouge1', 'rouge2', 'rougeL', 'rougeLsum']].std()
    print(method_std.round(4))
    
    # Document-wise comparison (LoRA vs ICL for the same document)
    print("\nDocument-wise Comparison (ROUGE-L):")
    comparison = []
    for doc_id in set(all_results['doc_id']):
        lora_score = all_results[(all_results['doc_id']==doc_id) & (all_results['method']=='LoRA')]['rougeL'].values
        icl_score = all_results[(all_results['doc_id']==doc_id) & (all_results['method']=='ICL')]['rougeL'].values
        
        if len(lora_score) > 0 and len(icl_score) > 0:
            comparison.append({
                'doc_id': doc_id[:8],
                'LoRA': lora_score[0],
                'ICL': icl_score[0],
                'Diff': lora_score[0] - icl_score[0]
            })
    
    comp_df = pd.DataFrame(comparison).sort_values('Diff', ascending=False)
    print(f"\nTop 5 where LoRA > ICL:")
    print(comp_df.head().to_string(index=False))
    print(f"\nTop 5 where ICL > LoRA:")
    print(comp_df.tail().to_string(index=False))
    
    # Win rate
    lora_wins = (comp_df['Diff'] > 0).sum()
    total_compared = len(comp_df)
    print(f"\nWin Rate: LoRA wins {lora_wins}/{total_compared} ({100*lora_wins/total_compared:.1f}%)")
    
    # Save
    all_results.to_csv(OUTPUTS_DIR / "combined_results.csv", index=False)
    comp_df.to_csv(OUTPUTS_DIR / "comparison_results.csv", index=False)
    
    # Final Report
    report = {
        'summary': {
            'total_documents': len(all_doc_ids),
            'lora_evaluated': len(lora_results),
            'icl_evaluated': len(icl_results),
            'lora_avg_rougeL': float(method_avg.loc['LoRA', 'rougeL']),
            'icl_avg_rougeL': float(method_avg.loc['ICL', 'rougeL']),
            'lora_win_rate': float(lora_wins / total_compared)
        },
        'method_statistics': method_avg.to_dict(),
        'all_results': all_results.to_dict('records')
    }
    
    with open(OUTPUTS_DIR / "final_comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to {OUTPUTS_DIR}")
    print("="*60)

if __name__ == "__main__":
    aggregate_all_results()