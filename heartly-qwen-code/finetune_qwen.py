#!/usr/bin/env python3
"""
finetune_qwen.py — Fine-tune Qwen2.5-Coder on the Heartly-grammar code SFT data.

- Base: Qwen/Qwen2.5-Coder-1.5B (standard transformer, no trust_remote_code needed)
- Heartly grammar tokens learned as literal text (multi-token) — no vocab surgery.
- Boundary head probes the final hidden layer at the <verify> token position.
- Loss masked to the response (prompt tokens = -100).
- QLoRA or full fine-tune depending on GPU memory.

Output: heartly-qwen-code/ (model + tokenizer)
"""
import argparse
import json
import re

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)


def collate(batch, tok, max_length=768):
    """Collate function with loss masking on the response (prompt = -100)."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    input_ids, labels = [], []
    for item in batch:
        p = tok.encode(f"User: {item['instruction']}\nAssistant: ")
        r = tok.encode(item["output"])
        ids = (p + r)[:max_length]
        lab = ([-100] * len(p) + r)[:max_length]
        pad = max_length - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
    ids_t = torch.tensor(input_ids, dtype=torch.long)
    return {
        "input_ids": ids_t,
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": ids_t.ne(pad_id).long(),
    }


def freeze_embeddings_and_layers(model, freeze_n):
    """Freeze the bottom N transformer layers + embeddings.

    For Qwen2.5-Coder: model.model.layers[i] is a DecoderLayer.
    Also freezes model.model.embed_tokens and model.model.norm if freeze_n > 0.
    """
    frozen = 0

    if freeze_n > 0:
        # Freeze embeddings
        model.model.embed_tokens.requires_grad_(False)
        frozen += 1
        # Freeze bottom N layers
        for i in range(min(freeze_n, len(model.model.layers))):
            for p in model.model.layers[i].parameters():
                p.requires_grad = False
            frozen += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen_pct = (1 - trainable / total) * 100 if total > 0 else 0
    print(f"frozen {frozen} layers | trainable {trainable/1e6:.0f}M / {total/1e6:.0f}M ({frozen_pct:.0f}% frozen)")
    return frozen


def setup_qlora(model):
    """Apply QLoRA via peft if available (for 11GB desktop use)."""
    try:
        from peft import LoraConfig, get_peft_model

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"[QLoRA] LoRA adapters attached")
        return model
    except ImportError:
        print(f"[SKIP] peft not installed — falling back to full fine-tune")
        return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--data", default="sft_dataset_code.jsonl")
    ap.add_argument("--out", default="heartly-qwen-code")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--freeze-layers", type=int, default=12,
                    help="Freeze bottom N transformer layers (Qwen 1.5B has 28 total)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples")
    ap.add_argument("--qlora", action="store_true", help="Use QLoRA instead of full fine-tune")
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()

    # Load data
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} training samples")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    # For QLoRA on 11GB: 4-bit quantization
    quantization_config = None
    if args.qlora:
        print("[QLoRA] Loading with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.repo,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto" if args.qlora else None,
    )

    # Freeze layers (not needed with QLoRA since LoRA is parameter-efficient)
    if not args.qlora:
        freeze_embeddings_and_layers(model, args.freeze_layers)

    # Apply QLoRA if requested
    if args.qlora:
        model = setup_qlora(model)

    # Dataset + Trainer
    ds = Dataset.from_list(rows)
    targs = TrainingArguments(
        output_dir=args.out + "-ckpt",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=args.dtype == "fp16",
        bf16=args.dtype == "bf16",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok, args.max_length),
    )
    trainer.train()

    # Save
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()