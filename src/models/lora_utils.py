# src/models/lora_utils.py

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Tuple

import torch.nn as nn


__all__ = [
    "detect_ffn_target_modules",
    "detect_attention_target_modules",
    "detect_ffn_and_attention_targets",
    "summarize_linear_suffix_frequencies",
]


def _iter_named_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Collect (name, module) for all torch.nn.Linear instances in the model.
    """
    linear_modules: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        # Some backends subclass Linear; isinstance check is robust
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module))
    return linear_modules


def _suffix(name: str) -> str:
    return name.split(".")[-1]


def summarize_linear_suffix_frequencies(model: nn.Module) -> Counter:
    """
    Return a Counter of last-token suffixes for all Linear module names.
    Helpful for quick inspection (e.g., q_proj/k_proj/gate_proj/up_proj/down_proj).
    """
    suffixes = [_suffix(n) for n, _ in _iter_named_linear_modules(model)]
    return Counter(suffixes)


def detect_ffn_target_modules(model: nn.Module) -> List[str]:
    """
    Detect FFN-only LoRA target module suffixes for LLaMA/Qwen-style models.

    Returns
    -------
    List[str]
        A list of suffix names to pass to PEFT's LoraConfig(target_modules=...),
        ordered as ["gate_proj", "up_proj", "down_proj"] when available.

    Notes
    -----
    - For Qwen3-8B (and most LLaMA-style models), FFN MLP layers expose three
      Linear modules per block: gate_proj / up_proj / down_proj.
    - Some older Qwen/variants may use w1 / w2 / w3 naming for FFN; we detect
      and return those if present.
    - If neither pattern is found, a RuntimeError is raised with a brief
      summary of available Linear suffixes to aid debugging.
    """
    lin_names = [name for name, _ in _iter_named_linear_modules(model)]
    suffixes = {_suffix(n) for n in lin_names}

    # Preferred, LLaMA/Qwen3-8B style
    llama_like = ["gate_proj", "up_proj", "down_proj"]
    if all(s in suffixes for s in llama_like):
        logging.info("Detected FFN pattern: gate_proj/up_proj/down_proj (LLaMA/Qwen style).")
        return llama_like

    # Alternate, some Qwen variants
    qwen_w = ["w1", "w2", "w3"]
    if all(s in suffixes for s in qwen_w):
        logging.info("Detected FFN pattern: w1/w2/w3 (Qwen variant).")
        return qwen_w

    # Heuristic fallback: look for the most common FFN-like names within *.mlp.* paths
    ffn_candidates = set()
    for name in lin_names:
        if (".mlp." in name) or ("ffn" in name) or ("feed_forward" in name):
            ffn_candidates.add(_suffix(name))

    if ffn_candidates:
        # Rank by overall frequency to choose stable top-3 if possible
        freq = summarize_linear_suffix_frequencies(model)
        ranked = sorted(ffn_candidates, key=lambda s: (-freq[s], s))
        # Keep up to three distinct suffixes
        selected = ranked[:3]
        logging.warning(
            "FFN pattern not standard; using heuristic selection from MLP scope: %s", selected
        )
        return selected

    # If we reach here, we couldn't infer a reasonable FFN set.
    freq_preview = summarize_linear_suffix_frequencies(model).most_common(12)
    raise RuntimeError(
        "Could not detect FFN target modules for this model. "
        f"Top Linear suffix frequencies: {freq_preview}. "
        "Please specify target_modules manually (e.g., ['gate_proj','up_proj','down_proj'])."
    )


def detect_attention_target_modules(model: nn.Module) -> List[str]:
    """
    Detect common self-attention Linear module suffixes.

    Returns
    -------
    List[str]
        One or more of: ['q_proj','k_proj','v_proj','o_proj'], filtered by presence.
    """
    suffixes = {_suffix(n) for n, _ in _iter_named_linear_modules(model)}
    attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    found = [s for s in attn if s in suffixes]
    if found:
        logging.info("Detected attention targets: %s", found)
    else:
        logging.warning("No standard attention Linear names detected among q/k/v/o proj.")
    return found


def detect_ffn_and_attention_targets(model: nn.Module) -> List[str]:
    """
    Convenience: union of FFN and attention targets, preserving a stable order.
    Useful if you want LoRA on both MLP and attention projections.
    """
    ffn = detect_ffn_target_modules(model)
    attn = detect_attention_target_modules(model)
    # Preserve the conventional ordering: attention first, then ffn
    ordered = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
    present = set(ffn) | set(attn)
    return [s for s in ordered if s in present]
