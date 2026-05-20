#!/usr/bin/env python3
# paperqa/evaluate_merging_methods.py

"""
PaperQA Multi-LoRA Merging Evaluation Script
Experiments with various merging methods for LoRA adapters on N specified papers.
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import glob
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
    print(f"Required libraries not installed: {e}")
    print("Install using: pip install nltk rouge-score")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("peft").setLevel(logging.ERROR)

class PaperQAMergingEvaluator:
    """Evaluator for PaperQA LoRA Merging methods."""
    
    def __init__(self, base_model_id: str, lora_adapters_dir: str, metrics: List[str], 
                 paper_indices: List[int], merging_method: str,
                 max_new_tokens: int = 256, experiment_name: str = "merging_evaluation", 
                 question_format: str = "bracket", merging_params: Dict = None,
                 routing_method: str = "perfect", retrieval_top_k: int = 3):
        """
        Args:
            base_model_id: Base model ID.
            lora_adapters_dir: Directory containing LoRA adapters.
            metrics: List of metrics to evaluate ['bleu', 'rouge', 'llm_judge'].
            paper_indices: List of paper IDs to use for evaluation.
            merging_method: LoRA merging method to use ('linear', 'cat', 'ties').
            max_new_tokens: Maximum tokens to generate.
            experiment_name: Name of the experiment.
            question_format: Question formatting style ('bracket' or 'natural').
            merging_params: Additional parameters for merging methods.
            routing_method: Routing method ('perfect' or 'rag').
            retrieval_top_k: Number of adapters to retrieve during RAG routing.
        """
        self.base_model_id = base_model_id
        self.lora_adapters_dir = lora_adapters_dir
        self.metrics = metrics
        self.paper_indices = paper_indices
        self.merging_method = merging_method
        self.max_new_tokens = max_new_tokens
        self.experiment_name = experiment_name
        self.question_format = question_format
        self.merging_params = merging_params or {}
        self.routing_method = routing_method
        self.retrieval_top_k = retrieval_top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.base_model, self.tokenizer = self._load_base_model()
        
        # Load LoRA adapters for specified paper IDs
        all_adapters = self._load_lora_adapters()
        self.selected_adapters = self._filter_adapters(all_adapters)
        
        

        # Initialize embedding model (for RAG routing)
        self.embed_model = None
        self.retriever = None
        
        if self.routing_method == "rag":
            self.retriever = self._setup_retriever()
        
        # Pre-load all selected adapters into PeftModel
        self.peft_model = self._setup_all_adapters()
        
        # Merge all adapters only if Perfect routing is selected
        if self.routing_method == "perfect":
            self._setup_peft_model()
        
        self.judge_llm = None
        if 'llm_judge' in self.metrics:
            self._init_llm_judge()

    def _load_base_model(self):
        model, tokenizer = load_model_and_tokenizer(
            self.base_model_id, self.device, use_chat_template=True
        )
        logging.info(f"Base model loaded: {self.base_model_id}")
        return model, tokenizer

    def _load_lora_adapters(self) -> List[Dict]:
        """Load LoRA adapters and collect metadata."""
        if not os.path.exists(self.lora_adapters_dir):
            raise FileNotFoundError(f"LoRA adapter directory not found: {self.lora_adapters_dir}")
        
        adapters = []
        
        for item in os.listdir(self.lora_adapters_dir):
            item_path = os.path.join(self.lora_adapters_dir, item)
            if os.path.isdir(item_path):
                final_path = os.path.join(item_path, "final")
                adapter_path = os.path.join(final_path, "adapter_model.safetensors")
                
                if os.path.exists(adapter_path):
                    import re
                    paper_match = re.search(r'paper(\d+)', item)
                    if paper_match:
                        paper_id = int(paper_match.group(1))
                    else:
                        numbers = re.findall(r'\d+', item)
                        if numbers:
                            paper_id = int(numbers[0])
                        else:
                            paper_id = hash(item) % 10000
                        logging.warning(f"Failed to extract paper ID, using alternative ID: {item} -> {paper_id}")
                    
                    try:
                        file_size = os.path.getsize(adapter_path)
                        if file_size < 1024:
                            logging.warning(f"Adapter file too small: {adapter_path} ({file_size} bytes)")
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
        
        logging.info(f"Loaded {len(adapters)} LoRA adapters in total.")
        return adapters
        
    def _filter_adapters(self, all_adapters: List[Dict]) -> List[Dict]:
        """Filter adapters matching the requested paper_indices."""
        filtered = [adapter for adapter in all_adapters if adapter['paper_id'] in self.paper_indices]
        
        found_ids = {adapter['paper_id'] for adapter in filtered}
        missing_ids = set(self.paper_indices) - found_ids
        
        if missing_ids:
            logging.warning(f"LoRA adapters not found for the following paper IDs: {sorted(list(missing_ids))}")
        
        if not filtered:
            raise ValueError(f"No adapters found for specified paper IDs {self.paper_indices}.")
            
        for adapter in filtered:
            logging.info(f"Selected adapter path: {adapter['path']}")
        return filtered

    def _setup_retriever(self):
        """Setup retriever based on paper introductions."""
        # Update path to be relative to PROJECT_ROOT or configurable
        paperqa_data_path = os.path.join(PROJECT_ROOT, "data", "paper_dataset", "paperQA.json")
        
        if not os.path.exists(paperqa_data_path):
            raise FileNotFoundError(f"PaperQA dataset not found: {paperqa_data_path}")
        
        with open(paperqa_data_path, 'r', encoding='utf-8') as f:
            paperqa_data = json.load(f)
        
        paper_introductions = {}
        for paper in paperqa_data:
            paper_id = paper.get('id')
            introduction = paper.get('introduction', '')
            if paper_id is not None and introduction:
                try:
                    paper_id_int = int(paper_id) if isinstance(paper_id, str) else paper_id
                    paper_introductions[paper_id_int] = introduction
                except (ValueError, TypeError):
                    logging.warning(f"Failed to convert paper ID: {paper_id}")
                    continue
        
        available_introductions = []
        introduction_idx_to_adapter = {}
        
        all_paper_ids = list(range(15))
        
        for paper_id in all_paper_ids:
            if paper_id in paper_introductions:
                available_introductions.append(paper_introductions[paper_id])
                
                adapter = None
                for selected_adapter in self.selected_adapters:
                    if selected_adapter['paper_id'] == paper_id:
                        adapter = selected_adapter
                        break
                
                if adapter:
                    introduction_idx_to_adapter[len(available_introductions) - 1] = adapter
                else:
                    introduction_idx_to_adapter[len(available_introductions) - 1] = {
                        'name': f'placeholder_paper{paper_id}',
                        'paper_id': paper_id,
                        'path': f'/placeholder/path/paper{paper_id}',
                        'adapter_path': f'/placeholder/path/paper{paper_id}/adapter.safetensors'
                    }
            else:
                logging.warning(f"Introduction not found for Paper ID {paper_id}.")
        
        if not available_introductions:
            raise ValueError("No paper introductions available.")
        
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
            
            logging.info(f"Query: {query[:100]}...")
            adapter_info = [f"{a['name']} (score: {a['similarity_score']:.3f})" for a in retrieved_adapters]
            logging.info(f"Retrieved adapters: {adapter_info}")
            
            return retrieved_adapters
        
        return retrieve_adapters

    def _init_llm_judge(self):
        try:
            self.judge_llm = Gpt4Synthesizer()
            logging.info("Azure GPT-4 Judge initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM Judge: {e}")
            self.judge_llm = None
    
    def _load_eval_data(self) -> List[dict]:
        """Load evaluation data for specified papers."""
        eval_data_path = os.path.join(PROJECT_ROOT, "data", "paper_dataset", "paperQA.json")
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Evaluation data file not found: {eval_data_path}")
        
        eval_data = []
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for paper in data:
            paper_id = paper.get('id', -1)
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

    def _setup_all_adapters(self) -> PeftModel:
        """Pre-load all selected LoRA adapters into PeftModel."""
        if not self.selected_adapters:
            raise ValueError("No LoRA adapters to load.")
        
        logging.info(f"Starting load of {len(self.selected_adapters)} LoRA adapters.")
        
        first_adapter = self.selected_adapters[0]
        peft_model = PeftModel.from_pretrained(self.base_model, first_adapter['path'], first_adapter['name'])
        logging.info(f"PeftModel initialized with: {first_adapter['name']}")
        
        for adapter in self.selected_adapters[1:]:
            try:
                peft_model.load_adapter(adapter['path'], adapter['name'])
                logging.info(f"Adapter loaded: {adapter['name']}")
            except Exception as e:
                logging.error(f"Failed to load adapter '{adapter['name']}': {e}")
        
        logging.info("All LoRA adapters loaded.")
        return peft_model
    
    def _setup_peft_model(self) -> PeftModel:
        """Merge all selected LoRA adapters."""
        if not self.selected_adapters:
            raise ValueError("No adapters selected for merging.")
        
        adapter_names = [adapter['name'] for adapter in self.selected_adapters]
        merged_name = f"merged_{self.merging_method}"
        
        if len(self.selected_adapters) == 1:
            self.peft_model.set_adapter(self.selected_adapters[0]['name'])
            logging.info("Only one adapter present, using without merging.")
            return self.peft_model
        
        # Clean up old merged adapters
        merged_names = [name for name in self.peft_model.peft_config.keys() if name.startswith('merged_')]
        for merged_name_old in merged_names:
            try:
                self.peft_model.delete_adapter(merged_name_old)
            except:
                pass
        
        

        try:
            logging.info(f"Merging adapters using '{self.merging_method}' method...")
            if self.merging_method == "linear":
                weights = [1.0 / len(adapter_names)] * len(adapter_names)
                self.peft_model.add_weighted_adapter(adapter_names, weights, merged_name, combination_type="linear")
            elif self.merging_method == "cat":
                self.peft_model.add_weighted_adapter(adapter_names, [1.0] * len(adapter_names), merged_name, combination_type="cat")
            elif self.merging_method == "ties":
                self.peft_model.add_weighted_adapter(
                    adapter_names, [1.0] * len(adapter_names), merged_name, combination_type="ties",
                    density=self.merging_params.get('density', 0.5),
                    majority_sign_method=self.merging_params.get('majority_sign_method', 'total')
                )
            else:
                raise ValueError(f"Unsupported merging method: {self.merging_method}")

            self.peft_model.set_adapter(merged_name)
            logging.info(f"Merging complete. Active adapter: {merged_name}")
            return self.peft_model
            
        except Exception as e:
            logging.error(f"Adapter merging failed: {e}")
            logging.warning("Falling back to first adapter due to merge failure.")
            self.peft_model.set_adapter(self.selected_adapters[0]['name'])
            return self.peft_model
    
    def _generate_answer(self, question: str, paper_id: int, retrieved_adapters: List[Dict] = None) -> Tuple[str, str]:
        """Generate answer based on routing method."""
        if self.routing_method == "perfect":
            return self._generate_answer_perfect(question, paper_id)
        elif self.routing_method == "rag":
            return self._generate_answer_rag(question, paper_id, retrieved_adapters)
        else:
            raise ValueError(f"Unsupported routing method: {self.routing_method}")
    
    def _generate_answer_perfect(self, question: str, paper_id: int) -> Tuple[str, str]:
        """Perfect routing: Generate answer using model with all adapters merged."""
        if self.question_format == "bracket":
            formatted_question = f"[Paper ID: {paper_id}] {question}"
        else:
            formatted_question = f"In the Paper ID {paper_id}, {question}"
            
        messages = [{"role": "user", "content": f"Question: {formatted_question}\nAnswer:"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return formatted_question, generated_text.strip()
    
    def _generate_answer_rag(self, question: str, paper_id: int, retrieved_adapters: List[Dict] = None) -> Tuple[str, str]:
        """RAG routing: Retrieve relevant adapters, merge them, and generate answer."""
        

        if retrieved_adapters is None:
            retrieved_adapters = self.retriever(question, self.retrieval_top_k)
        
        if not retrieved_adapters:
            logging.warning("No adapters retrieved. Generating with base model.")
            formatted_question = f"[Paper ID: {paper_id}] {question}"
            predicted_answer = "Cannot generate answer as no relevant adapters were found."
            return formatted_question, predicted_answer
        
        peft_model = self._setup_peft_model_for_adapters(retrieved_adapters)
        
        if self.question_format == "bracket":
            formatted_question = f"[Paper ID: {paper_id}] {question}"
        else:
            formatted_question = f"In the Paper ID {paper_id}, {question}"
            
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
    
    def _setup_peft_model_for_adapters(self, adapters: List[Dict]) -> PeftModel:
        """Setup PeftModel with selected adapters (for RAG routing)."""
        if not adapters:
            raise ValueError("No adapters to setup.")
        
        available_adapters = []
        for adapter in adapters:
            for selected_adapter in self.selected_adapters:
                if selected_adapter['paper_id'] == adapter['paper_id']:
                    available_adapters.append(selected_adapter)
                    break
        
        retrieved_info = [f"{a['name']} (paper {a['paper_id']})" for a in adapters]
        available_info = [f"{a['name']} (paper {a['paper_id']})" for a in available_adapters]
        logging.info(f"Retrieved adapters: {retrieved_info}")
        logging.info(f"Available loaded adapters: {available_info}")
        
        if not available_adapters:
            logging.warning("No loaded adapters match retrieval. Using first selected adapter.")
            available_adapters = [self.selected_adapters[0]]
        
        merged_names = [name for name in self.peft_model.peft_config.keys() if name.startswith('merged_')]
        for merged_name in merged_names:
            try:
                self.peft_model.delete_adapter(merged_name)
            except:
                pass
        
        if len(available_adapters) == 1:
            self.peft_model.set_adapter(available_adapters[0]['name'])
            return self.peft_model
        
        adapter_names = [adapter['name'] for adapter in available_adapters]
        merged_name = f"merged_{self.merging_method}_rag"
        
        try:
            if self.merging_method == "linear":
                weights = [1.0 / len(adapter_names)] * len(adapter_names)
                self.peft_model.add_weighted_adapter(
                    adapter_names, weights, merged_name, combination_type="linear"
                )
            
            elif self.merging_method == "cat":
                self.peft_model.add_weighted_adapter(
                    adapter_names, [1.0] * len(adapter_names), merged_name, combination_type="cat"
                )
            
            elif self.merging_method == "ties":
                density = self.merging_params.get('density', 0.5)
                majority_sign_method = self.merging_params.get('majority_sign_method', 'total')
                
                self.peft_model.add_weighted_adapter(
                    adapter_names, [1.0] * len(adapter_names), merged_name, combination_type="ties",
                    density=density, majority_sign_method=majority_sign_method
                )
            
            else:
                raise ValueError(f"Unsupported merging method: {self.merging_method}")
            
            self.peft_model.set_adapter(merged_name)
            return self.peft_model
            
        except Exception as e:
            logging.error(f"Adapter merging failed: {e}")
            logging.warning("Falling back to first adapter.")
            self.peft_model.set_adapter(adapters[0]['name'])
            return self.peft_model

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
    
    def _tokenize_text(self, text: str) -> List[str]:
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

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
        """Get score from LLM Judge."""
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
            logging.warning(f"Azure API call failed, assigning 0 score. Error: {e}")
            response_text = '{"score": 0, "rationale": "API call failed."}'
        
        return self._parse_judgment(response_text)
    
    def evaluate(self) -> Dict:
        """Main evaluation loop."""
        eval_data = self._load_eval_data()
        results = []
        all_scores = {metric: [] for metric in self.metrics}
        retrieval_log = []
        
        logging.info(f"Starting evaluation: {len(eval_data)} samples, Routing: {self.routing_method}, Merging: {self.merging_method}")
        
        for item in tqdm(eval_data, desc=f"Evaluating ({self.routing_method}-{self.merging_method})", unit="sample"):
            question = item['question']
            gold_answer = item['answer']
            paper_id = item['paper_id']
            
            if not question or not gold_answer:
                continue

            retrieved_adapters = None
            if self.routing_method == "rag":
                retrieved_adapters = self.retriever(question, self.retrieval_top_k)
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

            try:
                formatted_question, predicted_answer = self._generate_answer(question, paper_id, retrieved_adapters)
            except Exception as e:
                logging.error(f"Answer generation failed (Paper ID: {paper_id}): {e}")
                formatted_question = f"[Paper ID: {paper_id}] {question}"
                predicted_answer = "Failed to generate answer."
            
            result_record = {
                'question': formatted_question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'paper_id': paper_id,
                'level': item['level']
            }
            
            if self.routing_method == "rag" and retrieved_adapters is not None:
                result_record['retrieved_adapters'] = [
                    {
                        'name': adapter['name'],
                        'paper_id': adapter['paper_id'],
                        'similarity_score': adapter['similarity_score']
                    } for adapter in retrieved_adapters
                ]
            
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
            
            results.append(result_record)

        summary = {
            'experiment_name': self.experiment_name,
            'routing_method': self.routing_method,
            'merging_method': self.merging_method,
            'merging_params': self.merging_params,
            'paper_indices': self.paper_indices,
            'num_lora_adapters': len(self.selected_adapters),
            'num_evaluated_samples': len(results),
            'metrics': {}
        }
        
        if self.routing_method == "rag":
            summary['retrieval_top_k'] = self.retrieval_top_k
        
        for metric in self.metrics:
            scores = all_scores[metric]
            summary['metrics'][metric] = {
                'average': float(np.mean(scores)) if scores else 0.0,
                'std': float(np.std(scores)) if scores else 0.0,
            }
        
        result_dict = {'summary': summary, 'detailed_results': results}
        
        if self.routing_method == "rag":
            result_dict['retrieval_log'] = retrieval_log
        
        return result_dict
    
    def save_results(self, results: Dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results['summary'], f, indent=2, ensure_ascii=False)
        
        detailed_path = os.path.join(output_dir, "detailed_results.jsonl")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            for result in results['detailed_results']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        if 'retrieval_log' in results:
            retrieval_path = os.path.join(output_dir, "retrieval_log.json")
            with open(retrieval_path, 'w', encoding='utf-8') as f:
                json.dump(results['retrieval_log'], f, indent=2, ensure_ascii=False)
            logging.info(f"Retrieval log saved: {retrieval_path}")
        
        logging.info(f"Evaluation results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PaperQA LoRA Merging Methods")
    parser.add_argument("--base-model-id", type=str, required=True, help="Base model ID")
    parser.add_argument("--lora-adapters-dir", type=str, required=True, help="Directory containing LoRA adapters")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    
    parser.add_argument("--paper-indices", type=int, nargs='+', required=True, help="List of paper IDs for experiment (e.g., 0 1 2 3)")
    parser.add_argument("--merging-method", type=str, choices=['linear', 'cat', 'ties'], required=True, help="LoRA merging method (linear, cat, ties)")
    parser.add_argument("--routing-method", type=str, choices=['perfect', 'rag'], default='perfect', help="Routing method (perfect: merge all, rag: retrieve per question)")
    parser.add_argument("--retrieval-top-k", type=int, default=3, help="Number of adapters to retrieve for RAG routing (default: 3)")
    
    parser.add_argument("--metrics", type=str, nargs='+', default=['bleu', 'rouge', 'llm_judge'], help="Evaluation metrics")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--question-format", type=str, choices=['bracket', 'natural'], default='bracket', help="Question format style")
    
    parser.add_argument("--merging-density", type=float, default=0.5, help="Density for TIES merging")
    parser.add_argument("--merging-sign-method", type=str, default='total', help="Majority sign method for TIES merging")
    
    args = parser.parse_args()

    logging.info("="*80)
    logging.info("Experiment Configuration:")
    logging.info(f"  Base Model ID: {args.base_model_id}")
    logging.info(f"  LoRA Adapter Dir: {args.lora_adapters_dir}")
    logging.info(f"  Output Dir: {args.output_dir}")
    logging.info(f"  Paper Indices: {args.paper_indices}")
    logging.info(f"  Routing Method: {args.routing_method}")
    logging.info(f"  Merging Method: {args.merging_method}")
    if args.routing_method == "rag":
        logging.info(f"  Retrieval Top-K: {args.retrieval_top_k}")
    logging.info(f"  Metrics: {args.metrics}")
    logging.info(f"  Max Tokens: {args.max_new_tokens}")
    logging.info(f"  Question Format: {args.question_format}")
    if args.merging_method == "ties":
        logging.info(f"  TIES Density: {args.merging_density}")
        logging.info(f"  TIES Sign Method: {args.merging_sign_method}")
    logging.info("="*80)

    experiment_name = f"merging_eval_{args.routing_method}_{args.merging_method}_papers_{len(args.paper_indices)}"
    
    merging_params = {}
    if args.merging_method == "ties":
        merging_params = {
            'density': args.merging_density,
            'majority_sign_method': args.merging_sign_method
        }
    
    evaluator = PaperQAMergingEvaluator(
        base_model_id=args.base_model_id,
        lora_adapters_dir=args.lora_adapters_dir,
        metrics=args.metrics,
        paper_indices=args.paper_indices,
        merging_method=args.merging_method,
        merging_params=merging_params,
        max_new_tokens=args.max_new_tokens,
        experiment_name=experiment_name,
        question_format=args.question_format,
        routing_method=args.routing_method,
        retrieval_top_k=args.retrieval_top_k
    )
    
    results = evaluator.evaluate()
    
    evaluator.save_results(results, args.output_dir)
    
    print("\n" + "="*80)
    print("Evaluation Results Summary:")
    print("="*80)
    summary = results['summary']
    print(f"Routing Method: {summary['routing_method']}")
    print(f"Merging Method: {summary['merging_method']}")
    print(f"Papers Evaluated: {summary['num_lora_adapters']}")
    print(f"Samples Evaluated: {summary['num_evaluated_samples']}")
    if summary['routing_method'] == "rag":
        print(f"Retrieval Top-K: {summary.get('retrieval_top_k', 'N/A')}")
    for metric, stats in summary['metrics'].items():
        print(f"{metric.upper()} Average: {stats['average']:.4f} (Std: {stats['std']:.4f})")
    print("="*80)
    logging.info(f"Experiment complete. Results saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()