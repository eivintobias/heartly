#!/usr/bin/env python3
"""
train_probe_code.py — Train a boundary head (logistic probe) on Qwen2.5-Coder.

The boundary head reads the final hidden state at the <verify> token position
and classifies known vs unknown. This is the transformer equivalent of the
RWKV boundary head in heartly-rnn/train_probe.py.

For Qwen2.5-Coder: hidden states are extracted from the last layer's output
at the position of the <verify> token in the generated sequence.

Output: probe_head.pkl (sklearn LogisticRegression)
"""
import argparse
import json
import pickle
import re
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_verify_pos(tokens, tok):
    """Find the position of the <verify> token in a token sequence.

    <verify> is multi-token in Qwen's tokenizer. We search for the
    substring 'verify' in the decoded text and map back to token position.
    """
    text = tok.decode(tokens, skip_special_tokens=False)
    # Find '<verify>' in the decoded text
    idx = text.find("<verify>")
    if idx == -1:
        return -1
    # Count tokens up to that character position
    char_count = 0
    for i, t in enumerate(tokens):
        token_text = tok.decode([t], skip_special_tokens=False)
        char_count += len(token_text)
        if char_count > idx:
            return i
    return -1


def extract_verify_states(model, tok, samples, device, max_length=512):
    """Extract hidden states at the <verify> position for each sample.

    For each sample, we run the full sequence through the model and
    grab the last-layer hidden state at the <verify> token position.

    Returns:
        X: np.ndarray of shape (n_samples, hidden_dim)
        y: np.ndarray of shape (n_samples,) — 1 for known, 0 for unknown
    """
    model.eval()
    hidden_dim = model.config.hidden_size
    X, y = [], []

    with torch.no_grad():
        for item in samples:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            full_text = f"User: {instruction}\nAssistant: {output}"

            tokens = tok.encode(full_text, max_length=max_length, truncation=True)
            if len(tokens) == 0:
                continue

            # Find <verify> position
            verify_pos = find_verify_pos(tokens, tok)
            if verify_pos < 0:
                continue

            # Run through model
            input_ids = torch.tensor([tokens], device=device)
            outputs = model(input_ids, output_hidden_states=True)

            # Get last-layer hidden state at verify position
            # hidden_states is tuple of (batch, seq_len, hidden_dim) per layer
            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            verify_state = last_hidden[0, verify_pos, :].cpu().numpy()

            X.append(verify_state)
            y.append(1 if "<verify>known</verify>" in output else 0)

    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="heartly-qwen-code",
                    help="Path to fine-tuned model or HF repo name")
    ap.add_argument("--data", default="sft_dataset_code.jsonl",
                    help="SFT dataset to extract states from")
    ap.add_argument("--eval-limit", type=int, default=500,
                    help="Number of samples to use (for speed)")
    ap.add_argument("--head-out", default="probe_head.pkl",
                    help="Output path for the trained probe")
    ap.add_argument("--report", default="probe_report.json",
                    help="Output path for the evaluation report")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load model + tokenizer
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    print(f"loading model from {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.config.output_hidden_states = True  # Enable hidden state output
    print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Load data
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.eval_limit and args.eval_limit < len(rows):
        rows = rows[:args.eval_limit]
    print(f"extracting states from {len(rows)} samples...")

    X, y = extract_verify_states(model, tok, rows, device, args.max_length)
    print(f"extracted {len(X)} states (dim={X.shape[1] if X.ndim > 1 else '?'})")

    if len(X) < 10:
        print("ERROR: too few valid samples with <verify> tokens")
        sys.exit(1)

    # Train/test split
    n_train = int(len(X) * 0.7)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"train: {len(X_train)}  test: {len(X_test)}")
    print(f"  known: {y.sum()}  unknown: {(1-y).sum()}")

    # Train logistic regression probe
    probe = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)

    print(f"\n=== Boundary Head Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUROC:     {auroc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["unknown", "known"]))

    # Save probe
    with open(args.head_out, "wb") as f:
        pickle.dump(probe, f)
    print(f"probe saved -> {args.head_out}")

    # Save report
    report = {
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_known": int(y.sum()),
        "n_unknown": int((1 - y).sum()),
        "accuracy": float(acc),
        "auroc": float(auroc),
        "hidden_dim": X.shape[1] if X.ndim > 1 else 0,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report saved -> {args.report}")


if __name__ == "__main__":
    main()