#!/usr/bin/env python3
# 04e_evaluate_icl_and_rag.py
"""
Evaluates In-Context Learning (ICL) and Retrieval-Augmented Generation (RAG)
performance on the PaperQA dataset.

- Based on the ICL/RAG evaluation algorithms.
- Handles PaperQA dataset processing and argument configuration.
"""

import os
import sys
import json
import logging
import torch
import time
import faiss
import numpy as np
import argparse
import gc
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperQAEvaluator:
    """ICL and RAG Evaluator for the PaperQA dataset."""
    
    def __init__(self, model_id: str, device: str = "cuda", question_format: str = "natural", 
                 max_new_tokens: int = 256, paper_indices: List[int] = None, metrics: List[str] = None):
        """
        Args:
            model_id (str): Model ID or path of the LLM to use.
            device (str): Device to load the model on (e.g., "cuda" or "cpu").
            question_format (str): Question formatting style ("natural" or "bracket").
            max_new_tokens (int): Maximum number of tokens to generate.
            paper_indices (List[int]): List of paper IDs to use for evaluation.
            metrics (List[str]): List of evaluation metrics.
        """
        self.device = device
        self.model_id = model_id
        self.question_format = question_format
        self.max_new_tokens = max_new_tokens
        self.paper_indices = paper_indices or list(range(15))  # Default: 0-14
        self.metrics = metrics or ['rouge', 'bleu', 'llm_judge']
        
        logging.info(f"Loading base model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Handle models like LLAMA3 that might not have a pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()

        # Load full PaperQA data
        self.all_papers = self._load_all_papers()
        
        # Initialize LLM Judge
        self.judge_llm = None
        if 'llm_judge' in self.metrics:
            self._init_llm_judge()

    def _load_all_papers(self, input_path: str = None) -> List[Dict]:
        """Loads the full PaperQA dataset."""
        if input_path is None:
            # Default path relative to project root
            input_path = os.path.join(PROJECT_ROOT, "data", "paper_dataset", "paperQA.json")

        logging.info(f"Loading PaperQA dataset from {input_path}...")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                all_papers = json.load(f)
            logging.info(f"Loaded {len(all_papers)} papers.")
            return all_papers
        except FileNotFoundError:
            logging.error(f"Data file not found: {input_path}")
            raise

    def _load_eval_data(self) -> List[Dict]:
        """Loads evaluation data for the specified papers."""
        eval_data = []
        
        for paper in self.all_papers:
            paper_id = paper.get('id', -1)
            # Ensure paper_id is an integer for consistency
            try:
                paper_id_int = int(paper_id) if isinstance(paper_id, str) else paper_id
            except (ValueError, TypeError):
                paper_id_int = paper_id
            
            if paper_id_int in self.paper_indices and 'QA' in paper:
                for qa_item in paper['QA']:
                    eval_data.append({
                        'question': qa_item.get('question', ''),
                        'answer': qa_item.get('answer', ''),
                        'paper_id': paper_id_int,
                        'paper_title': paper.get('title', ''),
                        'level': qa_item.get('level', '')
                    })
        
        logging.info(f"Loaded {len(eval_data)} QA samples from {len(self.paper_indices)} papers.")
        return eval_data

    def _init_llm_judge(self):
        """Initializes the LLM Judge."""
        try:
            from src.synthesis.azure_gpt_synthesizer import Gpt4Synthesizer
            self.judge_llm = Gpt4Synthesizer()
            logging.info("Azure GPT-4 Judge initialized.")
        except ImportError as e:
            logging.error(f"Failed to import Gpt4Synthesizer: {e}")
            logging.error("LLM Judge module not found. Aborting.")
            raise ImportError(f"LLM Judge module import failed: {e}")
        except Exception as e:
            logging.error(f"LLM Judge initialization failed: {e}")
            logging.error("Aborting due to LLM Judge initialization failure.")
            raise RuntimeError(f"LLM Judge initialization failed: {e}")

    def _get_judging_prompt(self, question: str, gold_answer: str, predicted_answer: str) -> str:
        """Generates the prompt for the LLM Judge."""
        return f"""You are an impartial AI assistant acting as an expert judge. Your task is to evaluate a candidate's answer to a question about technical documents. Compare the candidate's answer against the gold standard answer.

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
        """Parses the LLM Judge response."""
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
        """Calculates the LLM Judge score."""
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
            logging.warning(f"Azure API call failed, assigning score 0. Error: {e}")
            response_text = '{"score": 0, "rationale": "API call failed."}'
        
        return self._parse_judgment(response_text)

    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculates BLEU score."""
        if not reference or not candidate:
            return 0.0
        
        # Tokenization
        ref_tokens = self._tokenize_text(reference)
        cand_tokens = self._tokenize_text(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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
            logging.warning(f"BLEU score calculation failed: {e}")
            return 0.0
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenizes text."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _calculate_rouge_l_score(self, reference: str, candidate: str) -> float:
        """Calculates ROUGE-L score."""
        if not reference or not candidate:
            return 0.0
        
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logging.warning(f"ROUGE-L score calculation failed: {e}")
            return 0.0

    def _format_question(self, question: str, paper_id: int) -> str:
        """Formats the question according to the specified style."""
        if self.question_format == "bracket":
            return f"[Paper ID: {paper_id}] {question}"
        else:  # natural
            return f"In the Paper ID {paper_id}, {question}"


    def _setup_retriever(self, texts: List[str]):
        """Sets up a simple FAISS-based Retriever."""
        try:
            embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=self.device)
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model: {e}")
            raise
            
        logging.info("Generating embeddings...")
        embeddings = embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.cpu().numpy())
        
        def retrieve(query: str, top_k: int = 3) -> List[int]:
            query_embedding = embed_model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            _, indices = index.search(query_embedding.cpu().numpy(), top_k)
            return indices[0].tolist()
        
        return retrieve

    def _generate_answer(self, question: str, paper_id: int, context: str = None) -> Tuple[str, str]:
        """Generates an answer using the model."""
        formatted_question = self._format_question(question, paper_id)
        
        if context:
            # RAG method: include context
            messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {formatted_question}\nAnswer:"}]
        else:
            # ICL method: no context
            messages = [{"role": "user", "content": f"Question: {formatted_question}\nAnswer:"}]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return formatted_question, generated_text.strip()

    def evaluate_method(self, eval_data: List[Dict], method: str) -> Dict:
        """Evaluates a specific method."""
        predictions = []
        all_scores = {metric: [] for metric in self.metrics}
        
        logging.info(f"Starting evaluation: {len(eval_data)} samples, Method: {method}")
        
        # Pre-prepare context for efficiency
        if method == 'fulltext_icl':
            # Prepare text for all 15 papers
            all_texts = []
            for paper in self.all_papers:
                paper_id_int = int(paper.get('id', -1)) if isinstance(paper.get('id'), str) else paper.get('id', -1)
                if paper.get('introduction'):
                    all_texts.append(f"Paper {paper_id_int}: {paper['introduction']}")
            context = "\n\n".join(all_texts)
            
        elif method == 'rag_selected_papers':
            # Use only selected papers as context
            context_texts = []
            for paper in self.all_papers:
                paper_id_int = int(paper.get('id', -1)) if isinstance(paper.get('id'), str) else paper.get('id', -1)
                if paper_id_int in self.paper_indices and paper.get('introduction'):
                    context_texts.append(f"Paper {paper_id_int}: {paper['introduction']}")
            context = "\n\n".join(context_texts)
            
        elif method == 'rag_retrieval':
            # Set up retriever with all paper texts
            all_paper_texts = []
            paper_id_mapping = []
            for paper in self.all_papers:
                paper_id_int = int(paper.get('id', -1)) if isinstance(paper.get('id'), str) else paper.get('id', -1)
                if paper.get('introduction'):
                    all_paper_texts.append(paper['introduction'])
                    paper_id_mapping.append(paper_id_int)
            
            if not all_paper_texts:
                logging.error("No paper texts available for retrieval.")
                return {'summary': {}, 'detailed_results': []}
            
            retriever = self._setup_retriever(all_paper_texts)
        
        for item in tqdm(eval_data, desc=f"Evaluating ({method})", unit="sample"):
            question = item['question']
            gold_answer = item['answer']
            paper_id = item['paper_id']
            
            if not question or not gold_answer:
                continue

            try:
                if method == 'rag_retrieval':
                    # Retrieve top 3 relevant papers
                    top_indices = retriever(question, top_k=3)
                    
                    logging.info(f"Selected indices: {top_indices}")

                    context_texts = []
                    for idx in top_indices:
                        retrieved_paper_id = paper_id_mapping[idx]
                        retrieved_text = all_paper_texts[idx]
                        context_texts.append(f"Paper {retrieved_paper_id}: {retrieved_text}")
                    context = "\n\n".join(context_texts)
                
                formatted_question, predicted_answer = self._generate_answer(question, paper_id, context)
                    
            except Exception as e:
                logging.error(f"Answer generation failed (Paper ID: {paper_id}): {e}")
                formatted_question = self._format_question(question, paper_id)
                predicted_answer = "Failed to generate answer."
            
            result_record = {
                'question': formatted_question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'paper_id': paper_id,
                'level': item['level']
            }
            
            # Calculate Metrics
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
                    all_scores['llm_judge'].append(judgment['score'])
            
            predictions.append(result_record)

        # Final Summary
        summary = {
            'method': method,
            'paper_indices': self.paper_indices,
            'question_format': self.question_format,
            'num_evaluated_samples': len(predictions),
            'metrics': {}
        }
        for metric in self.metrics:
            scores = all_scores[metric]
            summary['metrics'][metric] = {
                'average': float(np.mean(scores)) if scores else 0.0,
                'std': float(np.std(scores)) if scores else 0.0,
            }
        
        return {'summary': summary, 'detailed_results': predictions}

def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL and RAG methods for PaperQA dataset")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face Model ID or local path")
    parser.add_argument("--paper-indices", type=int, nargs='+', required=True, help="List of Paper IDs to evaluate (e.g., 0 1 2)")
    parser.add_argument("--output-dir", type=str, default="runs/paperqa/predictions/icl_rag", help="Directory to save predictions and results")
    parser.add_argument("--methods", nargs='+', default=['fulltext_icl', 'rag_selected_papers', 'rag_retrieval'], 
                        choices=['fulltext_icl', 'rag_selected_papers', 'rag_retrieval'],
                        help="List of methods to evaluate: fulltext_icl, rag_selected_papers, rag_retrieval")
    parser.add_argument("--question-format", type=str, choices=['natural', 'bracket'], default='natural', help="Question format style")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--metrics", type=str, nargs='+', default=['rouge', 'bleu', 'llm_judge'], help="Evaluation metrics")
    
    args = parser.parse_args()
    
    # Log Experimental Settings
    logging.info("="*80)
    logging.info("Experimental Settings:")
    logging.info(f"  Model ID: {args.model_id}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Paper Indices: {args.paper_indices}")
    logging.info(f"  Methods: {args.methods}")
    logging.info(f"  Question Format: {args.question_format}")
    logging.info(f"  Max New Tokens: {args.max_new_tokens}")
    logging.info(f"  Metrics: {args.metrics}")
    logging.info("="*80)

    # Initialize Evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = PaperQAEvaluator(
        model_id=args.model_id, 
        device=device, 
        question_format=args.question_format,
        max_new_tokens=args.max_new_tokens,
        paper_indices=args.paper_indices,
        metrics=args.metrics
    )
    
    # Load Evaluation Data
    eval_data = evaluator._load_eval_data()
    
    if not eval_data:
        logging.error("No data found to evaluate.")
        return
    
    logging.info(f"Starting evaluation on {len(eval_data)} QA samples.")
    
    # Generate Experiment Name
    experiment_name = f"icl_rag_eval_papers_{len(args.paper_indices)}"
    
    # Evaluate each method and save results
    all_results = {}
    
    for method in args.methods:
        logging.info(f"--- Evaluating Method: {method} ---")
        
        results = evaluator.evaluate_method(eval_data, method)
        
        if not results['detailed_results']:
            logging.warning(f"No predictions generated for method '{method}'.")
            continue
        
        # Save Results
        method_output_dir = os.path.join(args.output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Save Summary
        summary_path = os.path.join(method_output_dir, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results['summary'], f, indent=2, ensure_ascii=False)
        
        # Save Detailed Results
        detailed_path = os.path.join(method_output_dir, "detailed_results.jsonl")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            for result in results['detailed_results']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Add to overall results
        all_results[method] = results
        
        logging.info(f"Saved results for '{method}' to: {method_output_dir}")
        logging.info(f"Metrics for {method}: {results['summary']['metrics']}")

        # Cleanup Memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save Overall Experiment Summary
    if all_results:
        overall_summary_path = os.path.join(args.output_dir, "overall_evaluation_summary.json")
        overall_summary = {
            'experiment_name': experiment_name,
            'model_id': args.model_id,
            'paper_indices': args.paper_indices,
            'question_format': args.question_format,
            'max_new_tokens': args.max_new_tokens,
            'metrics': args.metrics,
            'methods_evaluated': list(all_results.keys()),
            'total_samples': sum(len(result['detailed_results']) for result in all_results.values()),
            'method_summaries': {method: result['summary'] for method, result in all_results.items()}
        }
        
        with open(overall_summary_path, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Overall experiment summary saved to: {overall_summary_path}")
    
    logging.info(f"All method evaluations complete.")
    
    # Print Result Summary
    print("\n" + "="*80)
    print("Evaluation Results Summary:")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Number of Papers: {len(args.paper_indices)}")
    print(f"Question Format: {args.question_format}")
    print(f"Metrics: {', '.join(args.metrics)}")
    
    for method in args.methods:
        if method in all_results:
            summary = all_results[method]['summary']
            print(f"\n{method.upper()}:")
            print(f"  Samples Evaluated: {summary['num_evaluated_samples']}")
            for metric, stats in summary['metrics'].items():
                if isinstance(stats, dict) and 'average' in stats:
                    print(f"  {metric.upper()} Average: {stats['average']:.4f} (Std: {stats['std']:.4f})")
    print("="*80)
    
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
