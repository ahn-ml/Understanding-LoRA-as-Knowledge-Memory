#!/usr/bin/env python3
"""
Evaluate a trained single-LoRA adapter on QuALITY with optional ICL/RAG context.
Modes: none | icl | rag
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


def build_retriever(article_idx: int, chunk_size: int, device: str):
    chunk_path = CHUNK_DIR / f"article_{article_idx}" / f"chunks_{chunk_size}.jsonl"
    texts = []
    if not chunk_path.exists():
        return None, None
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    if not texts:
        return None, None
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    emb = embed_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.array(emb).astype("float32"))
    return embed_model, (index, texts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate single-LoRA with optional ICL/RAG.")
    parser.add_argument("--model_tag", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--article_index", type=int, required=True)
    parser.add_argument("--context", type=str, choices=["none", "icl", "rag"], default="none")
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = MODELS[args.model_tag]
    base_model, tokenizer = load_and_prepare_model(base_model_id, device)

    adapter_path = Path(f"runs/qual/experiments/{args.model_tag}") / f"{args.model_tag}__article{args.article_index}" / "final"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    context_fn = None
    embed_model = None
    retriever = None
    if args.context == "icl":
        context_text = (SOURCE_DIR / f"article_{args.article_index}.txt").read_text(encoding="utf-8")
        context_fn = lambda _: context_text
    elif args.context == "rag":
        embed_model, ret = build_retriever(args.article_index, args.chunk_size, device)
        if ret:
            index, texts = ret

            def ctx(item):
                q_emb = embed_model.encode([item["question"]], convert_to_tensor=False, normalize_embeddings=True)
                scores, idxs = index.search(np.array(q_emb).astype("float32"), 3)
                chosen = [texts[i] for i in idxs[0].tolist()]
                return "\n\n".join(chosen)

            context_fn = ctx

    eval_path = EVAL_DIR / f"article_{args.article_index}.jsonl"
    eval_data = load_eval_data(str(eval_path))
    evaluator = MCQEvaluator(model, tokenizer, device=device)
    results = evaluator.evaluate(eval_data, context_fn=context_fn, max_new_tokens=8)

    out_dir = Path(f"runs/qual/{args.model_tag}/single_eval/article_{args.article_index}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"single_{args.context}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved %s (acc=%.4f)", out_path, results["accuracy"])


if __name__ == "__main__":
    main()
