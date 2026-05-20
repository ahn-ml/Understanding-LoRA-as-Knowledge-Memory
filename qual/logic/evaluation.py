import json
import logging
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm


def load_eval_data(path: str) -> List[Dict]:
    """Load eval questions from JSONL (question, options, answer index, hard)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logging.info("Loaded %d eval items from %s", len(data), path)
    return data


class MCQEvaluator:
    """Multiple-choice evaluator (accuracy)."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _build_prompt(self, question: str, options: List[str], context: Optional[str]) -> str:
        option_lines = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
        if context:
            prompt = f"""--------------BEGIN CONTEXT--------------
{context}
--------------END CONTEXT--------------

{question}
{option_lines}

Answer with the letter of the correct option:"""
        else:
            prompt = f"""{question}
{option_lines}

Answer with the letter of the correct option:"""
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt_text

    def _predict_letter(self, prompt_text: str, max_new_tokens: int = 8) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        for ch in generated:
            if ch.upper() in "ABCD":
                return ch.upper()
        return "X"

    def evaluate(
        self,
        eval_data: List[Dict],
        context_fn: Optional[Callable[[Dict], str]] = None,
        max_new_tokens: int = 8,
    ) -> Dict:
        preds = []
        correct = 0
        hard_total = 0
        hard_correct = 0

        for item in tqdm(eval_data, desc="Evaluating (MCQ)"):
            context = context_fn(item) if context_fn else ""
            prompt = self._build_prompt(item["question"], item["options"], context)
            pred_letter = self._predict_letter(prompt, max_new_tokens=max_new_tokens)
            gold_letter = chr(65 + item["answer"])
            is_correct = pred_letter == gold_letter
            if is_correct:
                correct += 1
            if item.get("hard", False):
                hard_total += 1
                if is_correct:
                    hard_correct += 1

            preds.append(
                {
                    "question": item["question"],
                    "options": item["options"],
                    "predicted": pred_letter,
                    "gold": gold_letter,
                    "is_correct": is_correct,
                    "hard": item.get("hard", False),
                    "context_used": bool(context),
                }
            )

        total = len(eval_data)
        acc = correct / total if total else 0.0
        hard_acc = hard_correct / hard_total if hard_total else 0.0

        return {
            "accuracy": acc,
            "hard_accuracy": hard_acc,
            "num_questions": total,
            "num_correct": correct,
            "num_hard": hard_total,
            "num_hard_correct": hard_correct,
            "predictions": preds,
        }
