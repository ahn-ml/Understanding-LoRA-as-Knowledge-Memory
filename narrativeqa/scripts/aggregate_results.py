# scripts/aggregate_results.py

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from path_utils import DATA_DIR, OUTPUTS_DIR

def load_single_result(result_path: Path) -> dict:
    """Load a single evaluation result file."""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading {result_path}: {e}")
        return None

def aggregate_results(doc_ids_path: Path = DATA_DIR / "doc_ids.json"):
    """Aggregate evaluation results for documents."""
    
    output_base = OUTPUTS_DIR / "multi_doc"
    
    with open(doc_ids_path, 'r') as f:
        doc_ids = json.load(f)
    
    print(f"Aggregating results for {len(doc_ids)} documents...\n")
    
    # Collect results
    all_results = []
    successful_docs = []
    failed_docs = []
    
    for doc_id in doc_ids:
        doc_short = doc_id[:8]
        result_path = output_base / f"doc_{doc_short}_iterative_qa" / "evaluation_results.json"
        
        if result_path.exists():
            result_data = load_single_result(result_path)
            if result_data and 'scores' in result_data:
                scores = result_data['scores']
                all_results.append({
                    'doc_id': doc_id,
                    'doc_short': doc_short,
                    'rouge1': scores.get('rouge1', 0),
                    'rouge2': scores.get('rouge2', 0),
                    'rougeL': scores.get('rougeL', 0),
                    'rougeLsum': scores.get('rougeLsum', 0),
                    'num_eval_samples': len(result_data.get('evaluation_data', []))
                })
                successful_docs.append(doc_id)
            else:
                failed_docs.append(doc_id)
                print(f"Warning: Invalid result format for doc {doc_short}")
        else:
            failed_docs.append(doc_id)
            print(f"Error: Result not found for doc {doc_short}")
    
    if not all_results:
        print("\nError: No results found! Check if experiments have completed.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate statistics
    rouge_cols = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    
    stats = {
        'mean': df[rouge_cols].mean(),
        'std': df[rouge_cols].std(),
        'min': df[rouge_cols].min(),
        'max': df[rouge_cols].max(),
        'median': df[rouge_cols].median()
    }
    
    # Print results
    print("\n" + "="*60)
    print("INDIVIDUAL DOCUMENT SCORES")
    print("="*60)
    
    # Sort individual scores by rougeL
    df_sorted = df.sort_values('rougeL', ascending=False)
    
    for idx, row in df_sorted.iterrows():
        print(f"\n📄 Document: {row['doc_short']}")
        print(f"   ROUGE-1: {row['rouge1']:.4f}")
        print(f"   ROUGE-2: {row['rouge2']:.4f}")
        print(f"   ROUGE-L: {row['rougeL']:.4f}")
        print(f"   ROUGE-Lsum: {row['rougeLsum']:.4f}")
        print(f"   Eval samples: {row['num_eval_samples']}")
    
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    
    print("\nMean Scores:")
    for metric in rouge_cols:
        print(f"   {metric}: {stats['mean'][metric]:.4f} (±{stats['std'][metric]:.4f})")
    
    print("\nDistribution:")
    for metric in rouge_cols:
        print(f"   {metric}:")
        print(f"      Min: {stats['min'][metric]:.4f}")
        print(f"      Median: {stats['median'][metric]:.4f}")
        print(f"      Max: {stats['max'][metric]:.4f}")
    
    # Save to CSV
    csv_path = output_base / "aggregate_results.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nIndividual results saved to: {csv_path}")
    
    # Save statistics to a separate CSV
    stats_df = pd.DataFrame(stats)
    stats_csv_path = output_base / "aggregate_statistics.csv"
    stats_df.to_csv(stats_csv_path)
    print(f"Statistics saved to: {stats_csv_path}")
    
    # Create JSON summary report
    report = {
        'metadata': {
            'total_documents': len(doc_ids),
            'successful': len(successful_docs),
            'failed': len(failed_docs),
            'timestamp': datetime.now().isoformat()
        },
        'aggregate_scores': {
            'mean': stats['mean'].to_dict(),
            'std': stats['std'].to_dict(),
            'min': stats['min'].to_dict(),
            'max': stats['max'].to_dict(),
            'median': stats['median'].to_dict()
        },
        'individual_scores': df_sorted.to_dict('records'),
        'failed_documents': failed_docs
    }
    
    report_path = output_base / "final_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {report_path}")
    
    # Top/Bottom performers
    print("\n" + "="*60)
    print("TOP 5 PERFORMERS (by ROUGE-L)")
    print("="*60)
    top5 = df_sorted.head(5)
    for idx, row in top5.iterrows():
        print(f"   {row['doc_short']}: {row['rougeL']:.4f}")
    
    print("\n" + "="*60)
    print("📉 BOTTOM 5 PERFORMERS (by ROUGE-L)")
    print("="*60)
    bottom5 = df_sorted.tail(5)
    for idx, row in bottom5.iterrows():
        print(f"   {row['doc_short']}: {row['rougeL']:.4f}")
    
    # Completion message
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE!")
    print("="*60)
    print(f"Successfully processed: {len(successful_docs)}/{len(doc_ids)} documents")
    if failed_docs:
        print(f"Failed documents: {len(failed_docs)}")
        print(f"  {[doc[:8] for doc in failed_docs]}")
    
    # Optional sample viewing
    print("\nTo view individual predictions, check:")
    print(f"   {output_base}/doc_*/evaluation_results.json")

if __name__ == "__main__":
    aggregate_results()