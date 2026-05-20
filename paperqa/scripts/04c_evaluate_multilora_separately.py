#!/usr/bin/env python3
# paperqa/evaluate_multilora_separately.py

"""
PaperQA Multi-LoRA Model Separate Evaluation Script
Since each LoRA model was trained on a specific paper,
this script filters questions for that specific paper, evaluates them,
and averages the results across all papers.
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import glob
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Import internal project modules
from src.models.model_loader import load_model_and_tokenizer
from src.synthesis.azure_gpt_synthesizer import Gpt4Synthesizer
from peft import PeftModel

# Libraries for BLEU and ROUGE evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    # Suppress absl logging (remove 'Using default tokenizer.' from rouge_score)
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass
    # Download NLTK data if necessary
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError as e:
    print(f"Required libraries are not installed: {e}")
    print("Please install using the following command:")
    print("pip install nltk rouge-score")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress unnecessary logs
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("rouge_score").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

class PaperQAMultiLoRAEvaluator:
    """PaperQA Multi-LoRA Model Separate Evaluator"""
    
    def __init__(self, base_model_id: str, lora_path: str, metrics: List[str], max_new_tokens: int = 256, 
                 experiment_name: str = "multilora_evaluation", question_format: str = None):
        """
        Args:
            base_model_id: Base model ID
            lora_path: Path to the trained LoRA adapter
            metrics: List of metrics to evaluate ['bleu', 'rouge', 'llm_judge']
            max_new_tokens: Maximum number of tokens to generate
            experiment_name: Name of the experiment
            question_format: Question formatting style ('bracket' or 'natural'). Auto-detected if None.
        """
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.experiment_name = experiment_name
        self.question_format = question_format
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Auto-detect question format if not provided
        if self.question_format is None:
            self.question_format = self._detect_question_format_from_path(self.lora_path)
            logging.info(f"Automatically detected question format: {self.question_format}")
        
        # Initialize LLM Judge (if needed)
        self.judge_llm = None
        if 'llm_judge' in metrics:
            self._init_llm_judge()
        
        # Load evaluation data
        self.eval_data = self._load_eval_data()
        
        # Load LoRA model
        self.model, self.tokenizer = self._load_model()
    
    def _detect_question_format_from_path(self, path: str) -> str:
        """Automatically detects the question format from the path.
        
        Args:
            path: File or directory path
            
        Returns:
            str: 'bracket' or 'natural'
        """
        path_lower = path.lower()
        
        if 'bracket' in path_lower:
            return 'bracket'
        elif 'natural' in path_lower:
            return 'natural'
        else:
            raise ValueError(f"Cannot detect question format from path: {path}")
    
    def _init_llm_judge(self):
        """Initialize LLM Judge"""
        try:
            
            self.judge_llm = Gpt4Synthesizer()
            logging.info("Azure GPT-4 Judge initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM Judge: {e}")
            raise
    
    def _load_eval_data(self) -> Dict[int, List[dict]]:
        """Load evaluation data by paper ID"""
        eval_data_path = "data/paper_dataset/paperQA.json"
        
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Evaluation data file not found: {eval_data_path}")
        
        eval_data = {}
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Group QA data by paper ID
        for paper in data:
            paper_id = paper.get('id')
            if paper_id is not None and 'QA' in paper and paper['QA']:
                eval_data[paper_id] = []
                for qa_item in paper['QA']:
                    eval_data[paper_id].append({
                        'question': qa_item.get('question', ''),
                        'answer': qa_item.get('answer', ''),
                        'paper_id': paper_id,
                        'paper_title': paper.get('title', ''),
                        'level': qa_item.get('level', '')
                    })
        
        logging.info(f"Evaluation data loaded: {len(eval_data)} papers")
        for paper_id, qa_list in eval_data.items():
            logging.info(f"  Paper {paper_id}: {len(qa_list)} QA samples")
        
        return eval_data
    
    def _load_model(self):
        """Load LoRA Model"""
        if not self.base_model_id:
            raise ValueError("base_model_id is not provided.")
        
        # Load base model
        model, tokenizer = load_model_and_tokenizer(
            self.base_model_id, 
            self.device, 
            use_chat_template=True
        )
        
        # Load LoRA adapter
        if not self.lora_path:
            raise ValueError("lora_path is not provided.")
        
        if not os.path.exists(self.lora_path):
            raise FileNotFoundError(f"LoRA adapter not found: {self.lora_path}")
        
        model = PeftModel.from_pretrained(model, self.lora_path)
        model.eval()
        
        logging.info(f"Successfully loaded LoRA model: {self.lora_path}")
        return model, tokenizer
    
    def _generate_answer(self, question: str, paper_id: int) -> Tuple[str, str]:
        """Generate answer for a question.
        
        Returns:
            Tuple[str, str]: (formatted_question, answer)
        """
        # Include Paper ID in the question based on the format
        if self.question_format == "bracket":
            formatted_question = f"[Paper ID: {paper_id}] {question}"
        elif self.question_format == "natural":
            formatted_question = f"In the Paper ID {paper_id}, {question}"
        else:
            raise ValueError(f"Invalid question format: {self.question_format}")
            
        # Create prompt using chat template (same format as training)
        user_prompt = f"Question: {formatted_question}\nAnswer:"
        messages = [{"role": "user", "content": user_prompt}]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=None,
                top_p=None
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        return formatted_question, generated_text.strip()
    
    def _extract_paper_id_from_path(self, lora_path: str) -> int:
        """Extract Paper ID from LoRA path."""
        import re
        
        # Supports patterns like: paperqa_multilora_paper0_*, paperqa_multilora_paper10_*
        match = re.search(r'paper(\d+)_', lora_path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Cannot extract Paper ID from LoRA path: {lora_path}")
    
    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        if not reference or not candidate:
            return 0.0
        
        ref_tokens = self._tokenize_text(reference)
        cand_tokens = self._tokenize_text(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        
        try:
            bleu_score = sentence_bleu(
                [ref_tokens], 
                cand_tokens, 
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            return bleu_score
        except Exception as e:
            logging.warning(f"Failed to calculate BLEU score: {e}")
            return 0.0
    
    def _calculate_rouge_l_score(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score"""
        if not reference or not candidate:
            return 0.0
        
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logging.warning(f"Failed to calculate ROUGE-L score: {e}")
            return 0.0
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _get_judging_prompt(self, question: str, gold_answer: str, predicted_answer: str) -> str:
        """Generate LLM Judge Prompt"""
        return f"""You are an impartial AI assistant acting as an expert judge. Your task is to evaluate a candidate's answer to a question about a technical document. Compare the candidate's answer against the gold standard answer.

[EVALUATION CRITERIA]
1.  **Factual Alignment**: Does the candidate answer state the same facts as the gold answer? It must not contradict the gold answer.
2.  **Completeness**: Does the candidate answer include all the key information and nuances present in the gold answer?
3.  **Relevance**: Is the answer focused and on-topic? It must not contain irrelevant or hallucinatory information.

[SCORING RUBRIC (0-10 SCALE)]
- **10**: Perfect. The candidate answer is factually identical to the gold answer, complete, and contains no extraneous information.
- **7-9**: Mostly Correct. The answer is factually correct but might omit a minor detail or be slightly verbose. The core information is present and accurate.
- **4-6**: Partially Correct. The answer has the right general idea but contains a significant factual error, a major omission, or irrelevant information.
- **1-3**: Incorrect. The answer is on-topic but factually wrong.
- **0**: Completely Incorrect. The answer is nonsensical, irrelevant, or fails to address the question.

[TASK]
Evaluate the [CANDIDATE ANSWER] based on the criteria above and its alignment with the [GOLD ANSWER]. Provide your output in a single JSON object with two keys: "score" (an integer from 0-10) and "rationale" (a brief, one-sentence explanation for your score).

[QUESTION]
{question}

[GOLD ANSWER]
{gold_answer}

[CANDIDATE ANSWER]
{predicted_answer}
"""
    
    def _parse_judgment(self, text: str) -> Dict:
        """Parse LLM Judge Response"""
        import re
        
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            logging.warning(f"Could not find a JSON object in the response. Text: {text}")
            return {"score": 0, "rationale": "Failed to parse JSON response."}
        
        try:
            json_str = match.group(0)
            data = json.loads(json_str)
            score = data.get("score")
            rationale = data.get("rationale")

            if isinstance(score, int) and 0 <= score <= 10:
                return {"score": score, "rationale": str(rationale)}
            else:
                logging.warning(f"Parsed score is invalid. Score: {score}")
                return {"score": 0, "rationale": "Parsed score was out of range or not an integer."}
                
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse judgment JSON: {e}. Text: {text}")
            return {"score": 0, "rationale": "JSON parsing error."}
    
    def _get_llm_judge_score(self, question: str, gold_answer: str, predicted_answer: str) -> Dict:
        """Calculate LLM Judge Score"""
        if not self.judge_llm:
            return {"score": 0, "rationale": "LLM Judge not initialized"}
        
        prompt = self._get_judging_prompt(question, gold_answer, predicted_answer)
        
        try:
            response = self.judge_llm.client.chat.completions.create(
                model=self.judge_llm.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
                top_p=1.0,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Azure API call failed, treating as 0 score. Error: {e}")
            response_text = '{"score": 0, "rationale": "API call failed."}'
        
        return self._parse_judgment(response_text)
    
    def evaluate(self) -> Dict:
        """Main evaluation function"""
        # Extract Paper ID from LoRA path
        paper_id = self._extract_paper_id_from_path(self.lora_path)
        
        if paper_id not in self.eval_data:
            raise ValueError(f"Evaluation data for Paper {paper_id} not found.")
        
        # Get QA data for the specific paper
        paper_qa_data = self.eval_data[paper_id]
        
        results = []
        all_scores = {metric: [] for metric in self.metrics}
        
        logging.info(f"Starting evaluation for Paper {paper_id}: {len(paper_qa_data)} samples, Metrics: {', '.join(self.metrics)}")
        
        for item in tqdm(paper_qa_data, desc=f"Evaluating Paper {paper_id}"):
            question = item.get('question', '')
            gold_answer = item.get('answer', '')
            
            if not question or not gold_answer:
                logging.warning("Question or answer is empty. Skipping.")
                continue
            
            # Generate answer
            formatted_question, predicted_answer = self._generate_answer(question, paper_id)
            
            # Create result record
            result_record = {
                'question': formatted_question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'paper_id': paper_id,
                'paper_title': item.get('paper_title', ''),
                'level': item.get('level', '')
            }
            
            # Calculate metrics
            for metric in self.metrics:
                if metric == 'bleu':
                    score = self._calculate_bleu_score(gold_answer, predicted_answer)
                    result_record['bleu_score'] = score
                    all_scores['bleu'].append(score)
                
                elif metric == 'rouge':
                    score = self._calculate_rouge_l_score(gold_answer, predicted_answer)
                    result_record['rouge_l_score'] = score
                    all_scores['rouge'].append(score)
                
                elif metric == 'llm_judge':
                    judgment = self._get_llm_judge_score(formatted_question, gold_answer, predicted_answer)
                    result_record['llm_judge_score'] = judgment['score']
                    result_record['llm_judge_rationale'] = judgment['rationale']
                    all_scores['llm_judge'].append(judgment['score'])
            
            results.append(result_record)
        
        # Calculate statistics
        evaluation_summary = {
            'paper_id': paper_id,
            'experiment_name': f"{self.experiment_name}_paper_{paper_id}",
            'evaluation_method': ', '.join(self.metrics),
            'num_evaluated': len(results),
            'metrics': {}
        }
        
        for metric in self.metrics:
            scores = all_scores[metric]
            if scores:
                evaluation_summary['metrics'][metric] = {
                    'average': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
        
        return {
            'summary': evaluation_summary,
            'detailed_results': results
        }
    
    def save_results(self, results: Dict, output_dir: str):
        """Save Results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary results
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results['summary'], f, indent=2, ensure_ascii=False)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, "detailed_results.jsonl")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            for result in results['detailed_results']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logging.info(f"Evaluation results saved:")
        logging.info(f"  - Summary: {summary_path}")
        logging.info(f"  - Details: {detailed_path}")

def main():
    parser = argparse.ArgumentParser(description="PaperQA Multi-LoRA Model Separate Evaluation")
    parser.add_argument("--base-model-id", type=str, default=os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                        help="Base model ID")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Path to the LoRA adapter")
    parser.add_argument("--metrics", type=str, nargs='+', 
                        choices=['bleu', 'rouge', 'llm_judge'],
                        default=['bleu', 'rouge', 'llm_judge'],
                        help="Metrics to evaluate (default: bleu rouge llm_judge)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate (default: 256)")
    parser.add_argument("--experiment-name", type=str, default="multilora_separate_evaluation",
                        help="Experiment name (default: multilora_separate_evaluation)")
    parser.add_argument("--question-format", type=str, choices=['bracket', 'natural'], 
                        default=None,
                        help="Question format style. If None, auto-detected from lora-path (default: None)")
    
    args = parser.parse_args()
    
    # Set output directory
    os.makedirs(f"{os.path.dirname(args.lora_path)}/evaluation", exist_ok=True)
    output_dir = f"{os.path.dirname(args.lora_path)}/evaluation"
    
    try:
        # Initialize Evaluator
        evaluator = PaperQAMultiLoRAEvaluator(
            base_model_id=args.base_model_id,
            lora_path=args.lora_path,
            metrics=args.metrics,
            max_new_tokens=args.max_new_tokens,
            experiment_name=args.experiment_name,
            question_format=args.question_format
        )
        
        # Execute Evaluation
        results = evaluator.evaluate()
        
        # Save Results
        evaluator.save_results(results, output_dir)
        
        # Print Summary
        
        print("\n" + "="*50)
        print("Multi-LoRA Separate Evaluation Summary:")
        print("="*50)
        summary = results['summary']
        print(f"Experiment Name: {summary['experiment_name']}")
        print(f"Evaluation Metrics: {summary['evaluation_method']}")
        print(f"Paper ID: {summary['paper_id']}")
        print(f"Num Evaluated: {summary['num_evaluated']}")
        print()
        
        for metric, stats in summary['metrics'].items():
            print(f"{metric.upper()}:")
            print(f"  Average: {stats['average']:.4f}")
            print(f"  Std Dev: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print()
        
        print("="*50)
        print(f"Results Saved To: {output_dir}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
