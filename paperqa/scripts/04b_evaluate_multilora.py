#!/usr/bin/env python3
# paperqa/evaluate_multi_lora_fixed.py

"""
PaperQA Multi-LoRA Model Evaluation Script (Fixed Version)
Supports various LoRA retrieval counts and merging methods.
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models.model_loader import load_model_and_tokenizer
from src.synthesis.azure_gpt_synthesizer import Gpt4Synthesizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Libraries for BLEU and ROUGE evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError as e:
    print(f"Required libraries are not installed: {e}")
    print("Please install using: pip install nltk rouge-score")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress unnecessary PEFT logs
logging.getLogger("peft").setLevel(logging.ERROR)

class PaperQAMultiLoRAEvaluator:
    """PaperQA Multi-LoRA Model Evaluator."""
    
    def __init__(self, base_model_id: str, lora_adapters_dir: str, metrics: List[str], 
                 max_new_tokens: int = 256, experiment_name: str = "multi_lora_evaluation", 
                 question_format: str = "bracket", retrieval_top_k: int = 3, 
                 merging_method: str = "linear", merging_params: Dict = None,
                 start_index: int = 0, end_index: int = 16):
        """
        Args:
            base_model_id: Base model ID.
            lora_adapters_dir: Directory where LoRA adapters are saved.
            metrics: List of metrics to evaluate ['bleu', 'rouge', 'llm_judge'].
            max_new_tokens: Maximum number of tokens to generate.
            experiment_name: Name of the experiment.
            question_format: Question formatting style ('bracket' or 'natural').
            retrieval_top_k: Number of LoRA adapters to retrieve.
            merging_method: LoRA merging method ('linear', 'cat', 'ties').
            merging_params: Additional parameters for merging methods.
            start_index: Start index of papers to evaluate (default: 0).
            end_index: End index of papers to evaluate (default: 16).
        """
        self.base_model_id = base_model_id
        self.lora_adapters_dir = lora_adapters_dir
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.experiment_name = experiment_name
        self.question_format = question_format
        self.retrieval_top_k = retrieval_top_k
        self.merging_method = merging_method
        self.merging_params = merging_params or {}
        self.start_index = start_index
        self.end_index = end_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.base_model, self.tokenizer = self._load_base_model()
        
        # Initialize embedding model (for reuse)
        self.embed_model = None
        
        # Load and index LoRA adapters
        self.lora_adapters = self._load_lora_adapters()
        self.retriever = self._setup_retriever()
        
        # Initialize PeftModel and preload all adapters
        self.peft_model = self._setup_all_adapters()
        
        # Initialize LLM Judge if needed
        self.judge_llm = None
        if 'llm_judge' in metrics:
            self._init_llm_judge()
    
    def _load_base_model(self):
        """Load the base model."""
        if not self.base_model_id:
            raise ValueError("base_model_id was not provided.")
        
        model, tokenizer = load_model_and_tokenizer(
            self.base_model_id, 
            self.device, 
            use_chat_template=True
        )
        
        logging.info(f"Successfully loaded base model: {self.base_model_id}")
        return model, tokenizer
    
    def _load_lora_adapters(self) -> List[Dict]:
        """Load LoRA adapters and collect metadata."""
        if not os.path.exists(self.lora_adapters_dir):
            raise FileNotFoundError(f"LoRA adapter directory not found: {self.lora_adapters_dir}")
        
        adapters = []
        
        # Search all subdirectories in the outputs directory
        for item in os.listdir(self.lora_adapters_dir):
            item_path = os.path.join(self.lora_adapters_dir, item)
            if os.path.isdir(item_path):
                # Check for adapter_model.safetensors in the final folder
                final_path = os.path.join(item_path, "final")
                adapter_path = os.path.join(final_path, "adapter_model.safetensors")
                
                if os.path.exists(adapter_path):
                    # Extract Paper ID from directory name
                    import re
                    paper_match = re.search(r'paper(\d+)', item)
                    if paper_match:
                        paper_id = int(paper_match.group(1))
                    else:
                        # Fallback: try extracting number from full directory name
                        numbers = re.findall(r'\d+', item)
                        if numbers:
                            paper_id = int(numbers[0])
                        else:
                            paper_id = hash(item) % 10000
                        logging.warning(f"Failed to extract Paper ID, using alternative ID: {item} -> {paper_id}")
                    
                    # Check file size to prevent loading empty files
                    try:
                        file_size = os.path.getsize(adapter_path)
                        if file_size < 1024:
                            logging.warning(f"Adapter file is too small: {adapter_path} ({file_size} bytes)")
                            continue
                    except OSError:
                        logging.warning(f"Failed to check adapter file size: {adapter_path}")
                        continue
                    
                    adapters.append({
                        'name': item,
                        'paper_id': paper_id,
                        'path': final_path,
                        'adapter_path': adapter_path
                    })
                    logging.info(f"LoRA adapter found: {item} (Paper ID: {paper_id})")
        
        if not adapters:
            raise FileNotFoundError(f"No LoRA adapters found in: {self.lora_adapters_dir}")
        
        logging.info(f"Loaded a total of {len(adapters)} LoRA adapters.")
        return adapters
    
    def _setup_retriever(self):
        """Setup retrieval system for LoRA adapters based on paper introductions."""
        # Define path relative to project root
        paperqa_data_path = os.path.join(PROJECT_ROOT, "data", "paper_dataset", "paperQA.json")
        
        if not os.path.exists(paperqa_data_path):
            raise FileNotFoundError(f"PaperQA dataset not found: {paperqa_data_path}")
        
        with open(paperqa_data_path, 'r', encoding='utf-8') as f:
            paperqa_data = json.load(f)
        
        # Map introductions by paper ID
        paper_introductions = {}
        for paper in paperqa_data:
            paper_id = paper.get('id')
            introduction = paper.get('introduction', '')
            if paper_id is not None and introduction:
                try:
                    paper_id_int = int(paper_id) if isinstance(paper_id, str) else paper_id
                    paper_introductions[paper_id_int] = introduction
                except (ValueError, TypeError):
                    logging.warning(f"Failed to convert Paper ID: {paper_id}")
                    continue
        
        # Collect introductions only for papers that have adapters
        available_introductions = []
        introduction_idx_to_adapter = {}
        
        for i, adapter in enumerate(self.lora_adapters):
            paper_id = adapter['paper_id']
            if paper_id in paper_introductions:
                available_introductions.append(paper_introductions[paper_id])
                introduction_idx_to_adapter[len(available_introductions) - 1] = adapter
            else:
                logging.warning(f"Introduction not found for Paper ID {paper_id}.")
        
        if not available_introductions:
            raise ValueError("No available paper introductions found.")
        
        # Vectorize introductions
        logging.info("Generating embeddings for paper introductions...")
        if self.embed_model is None:
            self.embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
        embed_model = self.embed_model
        
        introduction_embeddings = embed_model.encode(
            available_introductions, 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Create FAISS index
        index = faiss.IndexFlatIP(introduction_embeddings.shape[1])
        index.add(introduction_embeddings.cpu().numpy())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        def retrieve_adapters(query: str, top_k: int = None) -> List[Dict]:
            """Retrieve the most relevant LoRA adapters for a query."""
            if top_k is None:
                top_k = self.retrieval_top_k
            
            query_embedding = embed_model.encode(
                [query], 
                convert_to_tensor=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            ).cpu().numpy()
            
            scores, indices = index.search(query_embedding, min(top_k, len(available_introductions)))
            
            retrieved_adapters = []
            for intro_idx, score in zip(indices[0], scores[0]):
                if intro_idx in introduction_idx_to_adapter:
                    adapter = introduction_idx_to_adapter[intro_idx].copy()
                    adapter['similarity_score'] = float(score)
                    retrieved_adapters.append(adapter)
            
            return retrieved_adapters
        
        return retrieve_adapters
    
    def _init_llm_judge(self):
        """Initialize LLM Judge."""
        try:
            self.judge_llm = Gpt4Synthesizer()
            logging.info("Azure GPT-4 Judge initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM Judge: {e}")
            logging.warning("Proceeding with evaluation without LLM Judge.")
            self.judge_llm = None
    
    def _load_eval_data(self) -> List[dict]:
        """Load evaluation data."""
        eval_data_path = os.path.join(PROJECT_ROOT, "data", "paper_dataset", "paperQA.json")
        
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Evaluation data file not found: {eval_data_path}")
        
        eval_data = []
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract QA data from JSON array with filtering
        for paper in data:
            if 'QA' in paper and paper['QA']:
                paper_id = paper.get('id', '')
                try:
                    paper_id_int = int(paper_id) if isinstance(paper_id, str) else paper_id
                except (ValueError, TypeError):
                    paper_id_int = paper_id
                
                # Filter by paper ID range
                if isinstance(paper_id_int, int) and self.start_index <= paper_id_int < self.end_index:
                    for qa_item in paper['QA']:
                        eval_data.append({
                            'question': qa_item.get('question', ''),
                            'answer': qa_item.get('answer', ''),
                            'paper_id': paper_id_int,
                            'paper_title': paper.get('title', ''),
                            'level': qa_item.get('level', '')
                        })
        
        logging.info(f"Loaded evaluation data: {len(eval_data)} QA samples (Paper range: {self.start_index} <= paper_id < {self.end_index})")
        return eval_data
    
    def _setup_all_adapters(self) -> PeftModel:
        """Preload all LoRA adapters and setup PeftModel."""
        if not self.lora_adapters:
            raise ValueError("No LoRA adapters to load.")
        
        logging.info(f"Starting to load all LoRA adapters: {len(self.lora_adapters)} total")
        
        # Initialize PeftModel with the first adapter
        first_adapter = self.lora_adapters[0]
        peft_model = PeftModel.from_pretrained(self.base_model, first_adapter['path'], first_adapter['name'])
        logging.info(f"PeftModel initialized: {first_adapter['name']}")
        
        # Load the rest of the adapters
        for adapter in self.lora_adapters[1:]:
            try:
                peft_model.load_adapter(adapter['path'], adapter['name'])
                logging.info(f"Adapter loaded: {adapter['name']}")
            except Exception as e:
                logging.error(f"Failed to load adapter '{adapter['name']}': {e}")
        
        logging.info("All LoRA adapters loaded.")
        return peft_model
    
    def _setup_peft_model(self, adapters: List[Dict]) -> PeftModel:
        """Configure PeftModel with selected adapters (merged or single)."""
        if not adapters:
            raise ValueError("No adapters provided for setup.")
        
        # Clean up previously merged adapters
        merged_names = [name for name in self.peft_model.peft_config.keys() if name.startswith('merged_')]
        for merged_name in merged_names:
            try:
                self.peft_model.delete_adapter(merged_name)
            except:
                pass
        
        # If single adapter, return immediately
        if len(adapters) == 1:
            self.peft_model.set_adapter(adapters[0]['name'])
            return self.peft_model
        
        

        # Collect adapter names
        adapter_names = [adapter['name'] for adapter in adapters]
        merged_name = f"merged_{self.merging_method}"
        
        try:
            if self.merging_method == "linear":
                # Linear combination (equal weights)
                weights = [1.0 / len(adapter_names)] * len(adapter_names)
                self.peft_model.add_weighted_adapter(
                    adapter_names, 
                    weights, 
                    merged_name, 
                    combination_type="linear"
                )
            
            elif self.merging_method == "cat":
                # Concatenation
                self.peft_model.add_weighted_adapter(
                    adapter_names, 
                    [1.0] * len(adapter_names), 
                    merged_name, 
                    combination_type="cat"
                )
            
            elif self.merging_method == "ties":
                # TIES merging
                density = self.merging_params.get('density', 0.5)
                majority_sign_method = self.merging_params.get('majority_sign_method', 'total')
                
                self.peft_model.add_weighted_adapter(
                    adapter_names, 
                    [1.0] * len(adapter_names), 
                    merged_name, 
                    combination_type="ties",
                    density=density,
                    majority_sign_method=majority_sign_method
                )
            
            else:
                raise ValueError(f"Unsupported merging method: {self.merging_method}")
            
            self.peft_model.set_adapter(merged_name)
            return self.peft_model
            
        except Exception as e:
            logging.error(f"Adapter merge failed: {e}")
            logging.warning("Using the first adapter only due to merge failure.")
            self.peft_model.set_adapter(adapters[0]['name'])
            return self.peft_model
    
    def _generate_answer(self, question: str, retrieved_adapters: List[Dict], original_paper_id: int = None) -> Tuple[str, str]:
        """Generate answer for a question."""
        peft_model = self._setup_peft_model(retrieved_adapters)
        
        # Format question
        if original_paper_id is not None:
            if self.question_format == "bracket":
                formatted_question = f"[Paper ID: {original_paper_id}] {question}"
            elif self.question_format == "natural":
                formatted_question = f"In the Paper ID {original_paper_id}, {question}"
            else:
                formatted_question = f"[Paper ID: {original_paper_id}] {question}"
        else:
            formatted_question = question
            
        user_prompt = f"Question: {formatted_question}\nAnswer:"
        messages = [{"role": "user", "content": user_prompt}]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=None,
                top_p=None
            )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        return formatted_question, generated_text.strip()
    
    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score."""
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
            logging.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def _calculate_rouge_l_score(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score."""
        if not reference or not candidate:
            return 0.0
        
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logging.warning(f"ROUGE-L calculation failed: {e}")
            return 0.0
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _get_judging_prompt(self, question: str, gold_answer: str, predicted_answer: str) -> str:
        """Generate prompt for LLM Judge."""
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
        """Parse LLM Judge response."""
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
        """Calculate LLM Judge score."""
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
            logging.warning(f"Azure API call failed, defaulting to 0. Error: {e}")
            response_text = '{"score": 0, "rationale": "API call failed."}'
        
        return self._parse_judgment(response_text)
    
    def evaluate(self) -> Dict:
        """Main evaluation loop."""
        eval_data = self._load_eval_data()
        
        results = []
        all_scores = {metric: [] for metric in self.metrics}
        retrieval_log = []
        
        logging.info(f"Evaluation started: {len(eval_data)} samples, Metrics: {', '.join(self.metrics)}")
        logging.info(f"Paper range: {self.start_index} <= paper_id < {self.end_index}")
        logging.info(f"Retrieval settings: top_k={self.retrieval_top_k}, Merging: {self.merging_method}")
        
        for item in tqdm(eval_data, desc="Multi-LoRA Evaluation", unit="sample", ncols=100):
            question = item.get('question', '')
            gold_answer = item.get('answer', '')
            paper_id = item.get('paper_id', '')
            
            if not question or not gold_answer:
                logging.warning("Question or answer is empty. Skipping.")
                continue
            
            # Retrieve relevant LoRA adapters
            retrieved_adapters = self.retriever(question, self.retrieval_top_k)
            
            if not retrieved_adapters:
                logging.warning("No adapters retrieved. Generating answer with base model.")
                formatted_question = question
                predicted_answer = "Could not find relevant adapters; cannot generate answer."
                
                result_record = {
                    'question': formatted_question,
                    'gold_answer': gold_answer,
                    'predicted_answer': predicted_answer,
                    'paper_id': item.get('paper_id', ''),
                    'paper_title': item.get('paper_title', ''),
                    'level': item.get('level', ''),
                    'retrieved_adapters': []
                }
                
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
                continue
            
            # Log retrieval
            retrieval_log.append({
                'question': question,
                'retrieved_adapters': [
                    {
                        'name': adapter['name'],
                        'paper_id': adapter['paper_id'],
                        'similarity_score': adapter['similarity_score']
                    } for adapter in retrieved_adapters
                ]
            })
            
            # Generate Answer
            try:
                formatted_question, predicted_answer = self._generate_answer(question, retrieved_adapters, paper_id)
            except Exception as e:
                logging.error(f"Answer generation failed: {e}")
                formatted_question = question
                predicted_answer = "Failed to generate answer."
            
            result_record = {
                'question': formatted_question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'paper_id': item.get('paper_id', ''),
                'paper_title': item.get('paper_title', ''),
                'level': item.get('level', ''),
                'retrieved_adapters': [
                    {
                        'name': adapter['name'],
                        'paper_id': adapter['paper_id'],
                        'similarity_score': adapter['similarity_score']
                    } for adapter in retrieved_adapters
                ]
            }
            
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
            'experiment_name': self.experiment_name,
            'evaluation_method': ', '.join(self.metrics),
            'retrieval_top_k': self.retrieval_top_k,
            'merging_method': self.merging_method,
            'merging_params': self.merging_params,
            'paper_range': {
                'start_index': self.start_index,
                'end_index': self.end_index
            },
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
                    'max': float(np.max(scores)),
                    'count': len(scores)
                }
            else:
                evaluation_summary['metrics'][metric] = {
                    'average': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
                logging.warning(f"No scores for metric {metric}.")
        
        return {
            'summary': evaluation_summary,
            'detailed_results': results,
            'retrieval_log': retrieval_log
        }
    
    def save_results(self, results: Dict, output_dir: str):
        """Save results to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results['summary'], f, indent=2, ensure_ascii=False)
        
        detailed_path = os.path.join(output_dir, "detailed_results.jsonl")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            for result in results['detailed_results']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        retrieval_path = os.path.join(output_dir, "retrieval_log.json")
        with open(retrieval_path, 'w', encoding='utf-8') as f:
            json.dump(results['retrieval_log'], f, indent=2, ensure_ascii=False)
        
        logging.info(f"Evaluation results saved:")
        logging.info(f"  - Summary: {summary_path}")
        logging.info(f"  - Detailed: {detailed_path}")
        logging.info(f"  - Retrieval Log: {retrieval_path}")

def main():
    parser = argparse.ArgumentParser(description="PaperQA Multi-LoRA Model Evaluation")
    
    # Default path relative to project root
    default_base_model = os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    
    parser.add_argument("--base-model-id", type=str, 
                        default=default_base_model,
                        help="Base model ID (Default is relative path in project)")
    parser.add_argument("--lora-adapters-dir", type=str, required=True,
                        help="Directory containing LoRA adapters")
    parser.add_argument("--metrics", type=str, nargs='+', 
                        choices=['bleu', 'rouge', 'llm_judge'],
                        default=['bleu', 'rouge', 'llm_judge'],
                        help="Metrics to evaluate (Default: bleu rouge llm_judge)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate (Default: 256)")
    parser.add_argument("--experiment-name", type=str, default="multi_lora_evaluation",
                        help="Experiment name (Default: multi_lora_evaluation)")
    parser.add_argument("--question-format", type=str, choices=['bracket', 'natural'], 
                        default='bracket',
                        help="Question formatting style (Default: bracket)")
    parser.add_argument("--retrieval-top-k", type=int, default=3,
                        help="Number of LoRA adapters to retrieve (Default: 3)")
    parser.add_argument("--merging-method", type=str, 
                        choices=['linear', 'cat', 'ties'],
                        default='linear',
                        help="LoRA merging method (Default: linear)")
    parser.add_argument("--merging-density", type=float, default=0.5,
                        help="Density parameter for TIES merging (Default: 0.5)")
    parser.add_argument("--merging-sign-method", type=str, 
                        choices=['total', 'frequency'],
                        default='total',
                        help="Majority sign method for TIES merging (Default: total)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index of papers to evaluate (Default: 0)")
    parser.add_argument("--end-index", type=int, default=16,
                        help="End index of papers to evaluate (Default: 16)")
    
    args = parser.parse_args()
    
    merging_params = {}
    if args.merging_method == "ties":
        merging_params = {
            'density': args.merging_density,
            'majority_sign_method': args.merging_sign_method
        }
    
    evaluator = PaperQAMultiLoRAEvaluator(
        base_model_id=args.base_model_id,
        lora_adapters_dir=args.lora_adapters_dir,
        metrics=args.metrics,
        max_new_tokens=args.max_new_tokens,
        experiment_name=args.experiment_name,
        question_format=args.question_format,
        retrieval_top_k=args.retrieval_top_k,
        merging_method=args.merging_method,
        merging_params=merging_params,
        start_index=args.start_index,
        end_index=args.end_index
    )
    
    results = evaluator.evaluate()
    evaluator.save_results(results, args.output_dir)
    
    print("\n" + "="*60)
    print("Multi-LoRA Evaluation Summary:")
    print("="*60)
    summary = results['summary']
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Metrics: {summary['evaluation_method']}")
    print(f"Paper Range: {summary['paper_range']['start_index']} <= paper_id < {summary['paper_range']['end_index']}")
    print(f"Retrieval: top_k={summary['retrieval_top_k']}")
    print(f"Merging Method: {summary['merging_method']}")
    if summary['merging_params']:
        print(f"Merging Params: {summary['merging_params']}")
    print(f"Evaluated Samples: {summary['num_evaluated']}")
    print()
    
    for metric, stats in summary['metrics'].items():
        print(f"{metric.upper()}:")
        print(f"  Mean: {stats['average']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print()
    
    print("="*60)

if __name__ == "__main__":
    main()
