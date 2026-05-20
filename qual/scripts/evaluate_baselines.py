#!/usr/bin/env python3
"""
Evaluate base/ICL/RAG MCQ accuracy on QuALITY.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qual.logic.utils import load_and_prepare_model
from qual.logic.evaluation import MCQEvaluator, load_eval_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SOURCE_DIR = Path("qual/data/source")
EVAL_DIR = Path("qual/data/eval")
CHUNK_DIR = Path("qual/data/chunks")


def build_retriever(article_idx: int, chunk_size: int, device: str):
    chunk_path = CHUNK_DIR / f"article_{article_idx}" / f"chunks_{chunk_size}.jsonl"
    if not chunk_path.exists():
        logging.warning("Chunk file not found for article %d size %d", article_idx, chunk_size)
        return None
    texts = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    if not texts:
        return None

    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    embeddings = embed_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    def retrieve(query: str, top_k: int = 3) -> str:
        q_emb = embed_model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
        scores, idxs = index.search(np.array(q_emb).astype("float32"), top_k)
        chosen = [texts[i] for i in idxs[0].tolist()]
        return "\n\n".join(chosen)

    return retrieve


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation (base/ICL/RAG) on QuALITY.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["base", "icl", "rag"], required=True)
    parser.add_argument("--article_index", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size for RAG.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_and_prepare_model(args.model_id, device)
    evaluator = MCQEvaluator(model, tokenizer, device=device)

    eval_path = EVAL_DIR / f"article_{args.article_index}.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval data {eval_path}")
    eval_data = load_eval_data(str(eval_path))

    context_fn = None
    if args.mode == "icl":
        context = (SOURCE_DIR / f"article_{args.article_index}.txt").read_text(encoding="utf-8")
        context_fn = lambda _: context
    elif args.mode == "rag":
        retriever = build_retriever(args.article_index, args.chunk_size, device)
        if retriever:
            context_fn = lambda item: retriever(item["question"])

    results = evaluator.evaluate(eval_data, context_fn=context_fn)

    tag = Path(args.model_id).name.replace("/", "_")
    out_dir = Path(f"runs/qual/{tag}/baselines/{args.mode}/article_{args.article_index}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved results to %s (acc=%.4f)", out_path, results["accuracy"])


if __name__ == "__main__":
    main()
