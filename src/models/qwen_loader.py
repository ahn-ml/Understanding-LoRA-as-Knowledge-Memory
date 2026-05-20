# src/models/qwen_loader.py

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, Any

# --- 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _create_qwen_model(model_id: str, device: str, max_model_len: int = 4096):
    """
    Qwen3 모델과 토크나이저를 로드하는 내부 헬퍼 함수.
    max_model_len 파라미터를 통해 컨텍스트 창 크기를 동적으로 설정합니다.
    """
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logging.info(f"Loading Qwen model '{model_id}'... (Device: {device}, DType: {dtype}, Max Length: {max_model_len})")

    # Qwen 모델 로드 시 max_position_embeddings 설정
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device,
        max_position_embeddings=max_model_len,  # <-- 16K 컨텍스트 창 지원
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )

    # 'thinking' 기능 비활성화
    if hasattr(model.config, 'enable_thinking'):
        model.config.enable_thinking = False
        logging.info("Qwen 'thinking' feature has been disabled.")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.debug(f"Padding token set to EOS token: {tokenizer.pad_token}")

    return model, tokenizer

def load_qwen_with_lora(config: Dict[str, Any], lora_path: str = None):
    """
    실험 설정을 기반으로 Qwen 베이스 모델을 로드하고,
    선택적으로 학습된 LoRA 어댑터를 적용합니다.
    """
    model_config = config.get('model', {})
    
    model_id = model_config.get('base_model_id', 'Qwen/Qwen3-8B')
    max_model_len = model_config.get('max_model_len', 16384) # 기본값을 16K로 설정
    
    # 평가는 보통 단일 GPU에서 진행
    main_gpu_id = config.get('resources', {}).get('gpus', [0])[0]
    device = f"cuda:{main_gpu_id}"

    # 수정된 헬퍼 함수를 사용하여 모델 로드
    model, tokenizer = _create_qwen_model(
        model_id=model_id,
        device=device,
        max_model_len=max_model_len
    )
    
    # LoRA 경로가 제공된 경우 어댑터 적용
    if lora_path:
        logging.info(f"Applying LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer

def apply_qwen_chat_template(tokenizer, user_message: str, system_message: str = None) -> str:
    """
    Qwen의 공식 Chat Template을 적용하여 프롬프트를 구성합니다.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    try:
        # Chat Template 적용 (generation 프롬프트 포함, thinking 비활성화)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception as e:
        logging.error(f"Failed to apply Qwen chat template: {e}")
        # 템플릿 적용 실패 시 단순 결합으로 폴백
        return (f"{system_message}\n" if system_message else "") + user_message