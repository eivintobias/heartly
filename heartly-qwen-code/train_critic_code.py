#!/usr/bin/env python3
"""
train_critic_code.py — Stage 2: Train a code critic classifier.

The critic reads the model's hidden state at the <verify> position and
classifies: is this generated code actually correct, or a confident confabulation?

This is the code equivalent of the Heartly critic (Stage 2.5) — it targets
the "confident but wrong" blind spot where the model says verify=known but
the code is fake/broken.

Output: critic_head.pkl (sklearn LogisticRegression)
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_verify_pos(tokens, tok):
    """Find the position of the <verify> token in a token sequence."""
    text = tok.decode(tokens, skip_special_tokens=False)
    idx = text.find("<verify>")
    if idx == -1:
        return -1
    char_count = 0
    for i, t in enumerate(tokens):
        token_text = tok.decode([t], skip_special_tokens=False)
        char_count += len(token_text)
        if char_count > idx:
            return i
    return -1


def extract_verify_states(model, tok, samples, device, max_length=512):
    """Extract hidden states at the <verify> position for each sample."""
    model.eval()
    X, y = [], []

    with torch.no_grad():
        for item in samples:
            instruction = item.get("question", "")
            full_text = item.get("full_text", "")
            # Reconstruct the full conversation
            text = f"User: {instruction}\nAssistant: {full_text}"

            tokens = tok.encode(text, max_length=max_length, truncation=True)
            if len(tokens) == 0:
                continue

            verify_pos = find_verify_pos(tokens, tok)
            if verify_pos < 0:
                continue

            input_ids = torch.tensor([tokens], device=device)
            outputs = model(input_ids, output_hidden_states=True)

            last_hidden = outputs.hidden_states[-1]
            verify_state = last_hidden[0, verify_pos, :].cpu().float().numpy()

            X.append(verify_state)
            y.append(item["label"])

    return np.array(X), np.array(y)


def load_model(path, dtype="fp16"):
    """Load the fine-tuned Qwen model (QLoRA or full)."""
    from peft import PeftModel

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[dtype]

    is_lora = os.path.exists(os.path.join(path, "adapter_config.json"))

    if is_lora:
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(path)
        base_name = peft_config.base_model_name_or_path
        print(f"Loading LoRA adapter from {path} (base: {base_name})")
        tok = AutoTokenizer.from_pretrained(base_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch_dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, path)
    else:
        print(f"Loading full model from {path}")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map="auto"
        )

    model.eval()
    model.config.output_hidden_states = True
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="heartly-qwen-code-lora")
    ap.add_argument("--data", default="critic_data_code.jsonl")
    ap.add_argument("--head-out", default="critic_head.pkl")
    ap.add_argument("--report", default="critic_report.json")
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load model
    tok, model = load_model(args.model, args.dtype)
    print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Load critic data (only labeled rows — correct=1, confab=0)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    labeled = [r for r in rows if r.get("label") is not None]
    print(f"loaded {len(rows)} rows, {len(labeled)} labeled (correct={sum(1 for r in labeled if r['label']==1)}, confab={sum(1 for r in labeled if r['label']==0)})")

    if len(labeled) < 6:
        print("ERROR: too few labeled samples for training")
        sys.exit(1)

    # Extract hidden states
    print("extracting hidden states...")
    X, y = extract_verify_states(model, tok, labeled, device)
    print(f"extracted {len(X)} states (dim={X.shape[1]})")

    # Train/test split — shuffle first so both classes appear in each split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    n_train = len(X_train)

    print(f"train: {len(X_train)} (correct={sum(y_train)}, confab={sum(1-y for y in y_train)})")
    print(f"test:  {len(X_test)} (correct={sum(y_test)}, confab={sum(1-y for y in y_test)})")

    # Train logistic regression
    probe = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0
    auroc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 1.0

    print(f"\n{'='*50}")
    print(f"Code Critic Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    if len(y_test) > 1:
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["confab", "correct"], zero_division=0))

    # Save critic
    with open(args.head_out, "wb") as f:
        pickle.dump(probe, f)
    print(f"critic saved -> {args.head_out}")

    # Save report
    report = {
        "n_total": len(rows),
        "n_labeled": len(labeled),
        "n_correct": int(sum(y)),
        "n_confab": int(sum(1 - y)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": float(acc),
        "auroc": float(auroc),
        "hidden_dim": int(X.shape[1]),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report saved -> {args.report}")


if __name__ == "__main__":
    main()