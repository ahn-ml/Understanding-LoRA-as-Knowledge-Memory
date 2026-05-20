#!/usr/bin/env python3
# src/models/llama_loader.py
"""
Llama 3.1 모델 로딩 및 설정 관련 함수를 포함하는 모듈.
Instruct 모델의 경우 내장 chat template을 사용합니다.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_llama_chat_template(
    tokenizer,
    user_message: str,
    system_message: str | None = None,
    assistant_message: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Wraps Hugging Face chat template handling for Llama instruct models."""
    if not hasattr(tokenizer, "apply_chat_template"):
        # Fallback to simple concatenation when template is unavailable.
        prompt = ""
        if system_message:
            prompt += system_message.strip() + "\n"
        prompt += user_message
        if assistant_message is not None:
            prompt += f"\n{assistant_message}"
        return prompt

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    if assistant_message is not None:
        messages.append({"role": "assistant", "content": assistant_message})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

def load_llama_model_and_tokenizer(model_id: str, device: str, use_chat_template: bool):
    """
    Llama 3.1 모델과 토크나이저를 로드합니다.
    Instruct 모델은 이미 chat template이 내장되어 있으므로 별도 설정 불필요.
    """
    logging.info(f"Llama 모델 로드: {model_id} to {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Instruct 모델은 이미 chat_template이 설정되어 있음
    # use_chat_template 파라미터는 실제 적용 여부를 결정하는 데 사용됨
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer의 pad_token이 없어 eos_token으로 설정합니다.")
        
    return model, tokenizer
