#!/usr/bin/env python3
"""
Evaluate multi-LoRA adapters on QuALITY MCQ with top-k merging and optional ICL/RAG.
Modes:
  - top1 / top3 (ties)
  - context: none | icl | rag
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from peft import PeftModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qual.logic.utils import load_and_prepare_model
from qual.logic.evaluation import MCQEvaluator, load_eval_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SOURCE_DIR = Path("qual/data/source")
EVAL_DIR = Path("qual/data/eval")
CHUNK_DIR = Path("qual/data/chunks")

MODELS = {
    "qwen1p7": os.environ.get("LORAM_QWEN1P7_MODEL", "Qwen/Qwen3-1.7B"),
    "qwen8b": os.environ.get("LORAM_QWEN8B_MODEL", "Qwen/Qwen3-8B"),
    "llama8b": os.environ.get("LORAM_LLAMA8B_MODEL", os.environ.get("LORAM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")),
    "llama1b": os.environ.get("LORAM_LLAMA1B_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
}


def load_chunk_summaries(article_idx: int, chunk_size: int):
    sum_path = Path("qual/data/chunk_summaries") / f"article_{article_idx}" / f"chunk_summaries_{chunk_size}.jsonl"
    texts, ids = [], []
    if sum_path.exists():
        with open(sum_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append(rec.get("summary", ""))
                ids.append(rec["chunk_id"])
        return texts, ids
    # fallback to chunk text
    chunk_path = CHUNK_DIR / f"article_{article_idx}" / f"chunks_{chunk_size}.jsonl"
    if not chunk_path.exists():
        return texts, ids
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            ids.append(rec["chunk_id"])
    return texts, ids


def build_index(texts, device):
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    emb = embed_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.array(emb).astype("float32"))
    return embed_model, index


def load_adapter(model, adapter_root: Path, exp_name: str, chunk_id: int):
    """
    Load a trained adapter given its experiment name.
    Training outputs are stored as {adapter_root}/{exp_name}/final.
    """
    adapter_path = adapter_root / exp_name / "final"
    if adapter_path.exists():
        model.load_adapter(adapter_path, adapter_name=f"chunk_{chunk_id}")
        return True
    logging.warning("Adapter missing: %s", adapter_path)
    return False


def merge_topk(model, adapter_root: Path, exp_name_fn, chunk_ids, top_k: int):
    if top_k == 1:
        cid = chunk_ids[0]
        if load_adapter(model, adapter_root, exp_name_fn(cid), cid):
            model.set_adapter(f"chunk_{cid}")
            return True
        return False
    names = []
    for cid in chunk_ids[:top_k]:
        if load_adapter(model, adapter_root, exp_name_fn(cid), cid):
            names.append(f"chunk_{cid}")
    if not names:
        return False
    weights = [1 / len(names)] * len(names)
    model.add_weighted_adapter(
        adapters=names, weights=weights, combination_type="ties", adapter_name="merged", density=0.5
    )
    model.set_adapter("merged")
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-LoRA on QuALITY.")
    parser.add_argument("--model_tag", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--article_index", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=768)
    parser.add_argument("--top_k", type=int, choices=[1, 3], default=1)
    parser.add_argument("--context", type=str, choices=["none", "icl", "rag"], default="none")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = MODELS[args.model_tag]
    base_model, tokenizer = load_and_prepare_model(base_model_id, device)
    exp_name_fn = (
        lambda cid: f"{args.model_tag}__article{args.article_index}_s{args.chunk_size}_c{cid}"
    )

    # load first adapter to initialize PeftModel
    adapter_root = Path(f"runs/qual/{args.model_tag}/multi_lora/training")
    # adapters are stored as .../{exp_name}/final where exp_name contains article and chunk
    # we'll just wrap base model with an empty PEFT (load one adapter then delete)
    # find any adapter
    sample = next(adapter_root.rglob("final"), None)
    if sample is None:
        raise FileNotFoundError(f"No adapters found under {adapter_root}")
    model = PeftModel.from_pretrained(base_model, sample, adapter_name="tmp")
    model.delete_adapter("tmp")
    model.eval()

    texts, chunk_ids = load_chunk_summaries(args.article_index, args.chunk_size)
    if not texts:
        raise FileNotFoundError("No chunks found for article; run 05_chunk_articles.py")
    embed_model, index = build_index(texts, device)

    def context_fn(item):
        if args.context == "icl":
            return (SOURCE_DIR / f"article_{args.article_index}.txt").read_text(encoding="utf-8")
        if args.context == "rag":
            q_emb = embed_model.encode([item["question"]], convert_to_tensor=False, normalize_embeddings=True)
            scores, idxs = index.search(np.array(q_emb).astype("float32"), args.top_k)
            chosen = [texts[i] for i in idxs[0].tolist()]
            return "\n\n".join(chosen)
        return ""

    eval_path = EVAL_DIR / f"article_{args.article_index}.jsonl"
    eval_data = load_eval_data(str(eval_path))

    def pre_hook(item):
        # retrieval to decide which adapters to load
        if args.context == "rag":
            q_emb = embed_model.encode([item["question"]], convert_to_tensor=False, normalize_embeddings=True)
            scores, idxs = index.search(np.array(q_emb).astype("float32"), args.top_k)
        else:
            # use question only
            q_emb = embed_model.encode([item["question"]], convert_to_tensor=False, normalize_embeddings=True)
            scores, idxs = index.search(np.array(q_emb).astype("float32"), args.top_k)
        selected = [chunk_ids[i] for i in idxs[0].tolist()]
        # clean adapters
        for name in list(model.peft_config.keys()):
            if name in ("merged",) or name.startswith("chunk_"):
                model.delete_adapter(name)
        ok = merge_topk(model, adapter_root, exp_name_fn, selected, args.top_k)
        return ok

    evaluator = MCQEvaluator(model, tokenizer, device=device)
    preds = []
    correct = 0
    hard_total = 0
    hard_correct = 0
    for item in tqdm(eval_data, desc=f"mlora{args.top_k}_{args.context}"):
        if not pre_hook(item):
            preds.append({"question": item["question"], "options": item["options"], "predicted": "X", "gold": chr(65 + item["answer"]), "is_correct": False})
            continue
        context = context_fn(item)
        res = evaluator.evaluate([item], context_fn=(lambda _: context), max_new_tokens=8)
        p = res["predictions"][0]
        preds.append(p)
        if p["is_correct"]:
            correct += 1
        if item.get("hard", False):
            hard_total += 1
            if p["is_correct"]:
                hard_correct += 1

    total = len(eval_data)
    acc = correct / total if total else 0.0
    hard_acc = hard_correct / hard_total if hard_total else 0.0
    results = {
        "accuracy": acc,
        "hard_accuracy": hard_acc,
        "num_questions": total,
        "num_correct": correct,
        "num_hard": hard_total,
        "num_hard_correct": hard_correct,
        "predictions": preds,
        "mode": f"mlora{args.top_k}_{args.context}",
    }

    out_dir = Path(f"runs/qual/{args.model_tag}/multi_lora/eval/article_{args.article_index}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mlora{args.top_k}_{args.context}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved %s (acc=%.4f)", out_path, acc)


if __name__ == "__main__":
    from tqdm import tqdm

    main()
