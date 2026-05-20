# src/models/model_loader.py

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, PeftModel

def _create_base_model(model_id: str, device: str, dtype_str: str = "bfloat16"):
    """
    Hugging Face에서 순수 기반 모델과 토크나이저를 로드하는 내부 헬퍼 함수.
    """
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.startswith("cuda"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa" # 사용 가능한 경우 Flash Attention 2 사용
    ).to(device)

    # Qwen 계열은 thinking 기능을 갖고 있어 명시적으로 비활성화
    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def load_dcd_models(config: dict):
    """
    DCD 학습을 위한 교사(teacher) 모델과 학생(student) 기반 모델을 로드합니다.
    """
    model_id = config['model']['base_model_id']
    teacher_gpu_id, student_gpu_id = config['resources']['gpus']
    
    teacher_device = f"cuda:{teacher_gpu_id}"
    student_device = f"cuda:{student_gpu_id}"
    
    logging.info(f"교사 모델 로드 중... (Device: {teacher_device})")
    teacher_model, tokenizer = _create_base_model(model_id, teacher_device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    logging.info(f"학생 모델 로드 중... (Device: {student_device})")
    student_base_model, _ = _create_base_model(model_id, student_device)
    
    return teacher_model, student_base_model, tokenizer

def apply_lora_and_get_counts(base_model: AutoModelForCausalLM, config: dict):
    """
    주어진 기반 모델에 LoRA 설정을 적용하고, 학습 가능한 파라미터 수를 계산하여 반환합니다.
    """
    lora_config_dict = config['model']['lora_config']
    lora_config = LoraConfig(**lora_config_dict)
    
    lora_model = get_peft_model(base_model, lora_config)
    
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    logging.info("모델 파라미터 정보:")
    logging.info(f"  - 전체 파라미터: {total_params:,}")
    logging.info(f"  - 학습 가능 파라미터 (LoRA): {trainable_params:,}")
    logging.info(f"  - 비율: {trainable_params / total_params * 100:.4f}%")
    
    lora_model.print_trainable_parameters()
    
    return lora_model, total_params, trainable_params

def load_base_model_with_lora(config: dict, lora_path: str):
    """
    평가를 위해, 새로운 기반 모델에 이미 학습된 LoRA 어댑터를 불러와 병합합니다.
    """
    model_id = config['model']['base_model_id']
    # 평가는 보통 단일 GPU에서 진행되므로, gpus 리스트의 첫 번째 GPU를 사용합니다.
    main_gpu_id = config['resources']['gpus'][0]
    device = f"cuda:{main_gpu_id}"
    
    logging.info(f"평가용 베이스 모델 로드 중... (Device: {device})")
    model, tokenizer = _create_base_model(model_id, device)
    
    logging.info(f"학습된 LoRA 어댑터 병합 중: {lora_path}")
    # PeftModel을 사용하여 저장된 LoRA 가중치를 베이스 모델에 적용
    lora_model = PeftModel.from_pretrained(model, lora_path)
    lora_model.eval()
    
    return lora_model, tokenizer
