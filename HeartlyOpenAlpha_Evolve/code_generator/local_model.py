#!/usr/bin/env python3
"""
local_model.py — Load and run inference on a local Heartly-Qwen-Coder model.

Supports:
- Full fine-tuned model loading (fp16/bf16/fp32)
- QLoRA adapter loading (via peft)
- Auto device mapping (CUDA or CPU)
- Generation with Heartly-compatible parameters
"""
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    device: str = "auto",
    dtype: str = "fp16",
    use_qlora: bool = False,
):
    """Load a Heartly-Qwen-Coder model and tokenizer from a local path.

    Args:
        model_path: Path to the fine-tuned model directory.
        device: 'cuda', 'cpu', or 'auto'.
        dtype: 'fp16', 'bf16', or 'fp32'.
        use_qlora: Whether the model was trained with QLoRA (loads base + adapter).

    Returns:
        (model, tokenizer)
    """
    import json
    import os

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Check if this is a LoRA adapter directory (has adapter_config.json)
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora_adapter = os.path.exists(adapter_config_path)

    if is_lora_adapter or use_qlora:
        # Load tokenizer from the adapter directory (or base model)
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Read base model name from adapter config
        base_model_name = "Qwen/Qwen2.5-Coder-1.5B"  # default fallback
        if is_lora_adapter:
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path", base_model_name)
                logger.info(f"LoRA adapter detected. Base model: {base_model_name}")
            except Exception as e:
                logger.warning(f"Could not read adapter config, using default base: {e}")

        quantization_config = None
        if use_qlora:
            logger.info("QLoRA mode: loading base model with 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(f"Loading base model from {base_model_name} (dtype={dtype}, qlora={use_qlora})")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if device == "auto" else {"": 0} if device == "cuda" else {"": "cpu"},
        )
        model.eval()

        # Load the LoRA adapter on top of the base model
        try:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            logger.info("LoRA adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapter from {model_path}: {e}")
    else:
        # Standard full model loading
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        if use_qlora:
            logger.info("QLoRA mode: loading base model with 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(f"Loading model from {model_path} (dtype={dtype}, qlora={use_qlora})")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if device == "auto" else {"": 0} if device == "cuda" else {"": "cpu"},
        )
        model.eval()

        # If QLoRA, load the adapter on top of the base
        if use_qlora:
            try:
                from peft import PeftModel
                adapter_path = model_path
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info("QLoRA adapter loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load QLoRA adapter from {model_path}: {e}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded: {total_params:.0f}M params")

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    """Generate a completion from the local model.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompt: Input prompt (already formatted for the model).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (None = greedy).
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.

    Returns:
        Generated text string.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature is not None,
    }

    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if top_k is not None:
        gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        outputs = model.generate(inputs, **gen_kwargs)

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text


def extract_hidden_state_at_verify(
    model,
    tokenizer,
    text: str,
    verify_pos_in_text: int,
    max_length: int = 512,
) -> Optional[torch.Tensor]:
    """Extract the last-layer hidden state at the <verify> token position.

    Used by the boundary head to classify known vs unknown.

    Args:
        model: Loaded model with output_hidden_states=True.
        tokenizer: Loaded tokenizer.
        text: Full input text.
        verify_pos_in_text: Character position of '<verify>' in the text.
        max_length: Max sequence length.

    Returns:
        Hidden state vector (hidden_dim,) or None if position not found.
    """
    tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
    # Find token position corresponding to <verify>
    char_count = 0
    verify_token_pos = -1
    for i, t in enumerate(tokens):
        token_text = tokenizer.decode([t], skip_special_tokens=False)
        char_count += len(token_text)
        if char_count > verify_pos_in_text:
            verify_token_pos = i
            break

    if verify_token_pos < 0:
        return None

    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Last layer hidden state at verify position
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    state = last_hidden[0, verify_token_pos, :]  # (hidden_dim,)
    return state
