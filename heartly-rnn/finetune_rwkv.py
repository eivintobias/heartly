#!/usr/bin/env python3
"""
finetune_rwkv.py — Stage 2: fine-tune RWKV on the Heartly grammar.

- Base: RWKV/rwkv-4-world-430m (custom world tokenizer, trust_remote_code).
- The 7 grammar markers are trained as literal text (multi-token). Experiment 1
  showed the sensor reads state, not token ids — no vocab surgery needed; the
  `<verify>` position is located by string offset at measurement time.
- Loss masked to the response (prompt = -100), same semantics as the Heartly
  collate_and_mask_loss.
- Bottom `--freeze-layers` transformer blocks stay frozen (CPU budget; format
  learning concentrates in late layers).

Output: rwkv-heartly/ (model + tokenizer)
"""
import argparse
import json
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def collate(batch, tok, max_length=768):
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
    return {"input_ids": ids_t,
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": ids_t.ne(pad_id).long()}


def freeze_bottom_blocks(model, freeze_n):
    """Freeze blocks with index < freeze_n (name-based, defensive).
    rwkv-4 custom code uses 'blocks.N.'; fla RWKV7 uses 'layers.N.'."""
    frozen = 0
    pat = re.compile(r"(?:blocks|layers)\.(\d+)\.")
    for name, p in model.named_parameters():
        m = pat.search(name)
        if m and int(m.group(1)) < freeze_n:
            p.requires_grad = False
            frozen += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"frozen {frozen} tensors | trainable {trainable/1e6:.0f}M / {total/1e6:.0f}M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RWKV/rwkv-4-world-430m")
    ap.add_argument("--data", default="sft_dataset.jsonl")
    ap.add_argument("--out", default="rwkv-heartly")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-layers", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="fla RWKV7 kernels want bf16/fp16; fp32 warns/falls back")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} training samples")

    tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
    _dt = {"fp32": torch.float32, "bf16": torch.bfloat16,
           "fp16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=_dt,
                                                 trust_remote_code=True)
    # fla's fused linear cross-entropy returns a view that transformers v5's
    # Trainer then modifies in place (loss *= scale) -> RuntimeError. Disable.
    if getattr(model.config, "fuse_cross_entropy", False):
        model.config.fuse_cross_entropy = False
        print("fla fused cross-entropy disabled (Trainer-incompatible)")
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = getattr(tok, "eos_token", None)
    freeze_bottom_blocks(model, args.freeze_layers)

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
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok, args.max_length))
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()