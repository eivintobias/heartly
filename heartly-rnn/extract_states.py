#!/usr/bin/env python3
"""
extract_states.py — Dump per-layer states at the end-of-question position.

Families:
  qwen  → residual-stream hidden states (transformer baseline; no recurrent state)
  mamba → hidden states + ssm_state (the compressed recurrent memory)
  rwkv  → hidden states + carried RWKV state

Per question, for selected layers (quartiles + final by default):
  hidden  : [L, H]   fp16  — last prompt token of each layer's hidden state
  rstate  : [L, S]   fp16  — flattened recurrent state (empty for qwen)

With --verify-known, also greedy-generates answers for known candidates and
stores a `verified` mask (gold string found in completion) — corpus-known
upgraded to model-verified, per the "model's true boundary" rule.

Output: states/states_{tag}.npz  {ids, hidden, rstate, layer_idx, verified}
        states/verify_{tag}.jsonl (only with --verify-known)
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def quartile_idx(n):
    idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    return idx


def load_family(family, repo, device, trust_remote_code, tokenizer_repo=None,
                dtype=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_repo or repo,
                                            trust_remote_code=trust_remote_code)
    except Exception as e:
        # state-spaces/mamba-* ships weights only; the expected tokenizer is
        # the EleutherAI GPT-NeoX one.
        if family.startswith("mamba") and not tokenizer_repo:
            print(f"[tokenizer] repo has no tokenizer ({type(e).__name__}); "
                  f"falling back to EleutherAI/gpt-neox-20b")
            tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        else:
            raise
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=dtype or torch.float32,  # probing: precision over speed
        trust_remote_code=trust_remote_code,  # (dtype override: big-model VRAM)
    )
    model.to(device).eval()
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok, model


@torch.no_grad()
def forward_features(family, model, enc, layer_idx):
    """Returns (hidden[L,H], rstate[L,S]) numpy fp16 at end-of-question."""
    out = model(**enc, use_cache=True, output_hidden_states=True)

    hidden = None
    hs = getattr(out, "hidden_states", None)
    if hs is not None:
        sel = [hs[i][0, -1, :].detach().cpu().numpy().astype(np.float16) for i in layer_idx]
        hidden = np.stack(sel)

    # Recurrent state, auto-detected:
    #   v5 DynamicCache (falcon_h1) -> past_key_values.layers[i].recurrent_states
    #   mamba/mamba2                -> cache_params.ssm_states
    #   rwkv                        -> state
    # qwen (transformer baseline) has none.
    rstate = None
    pkv = getattr(out, "past_key_values", None)
    layers = getattr(pkv, "layers", None) if pkv is not None else None
    if layers:
        sel = []
        for i in layer_idx:
            if i >= len(layers):
                continue
            rs = getattr(layers[i], "recurrent_states", None)
            if not (isinstance(rs, dict) and rs):
                # fla (RWKV7): FLALayer.state is a dict, e.g.
                # {'recurrent_state': (batch, heads, k, v), 'conv_state': ...}
                rs = getattr(layers[i], "state", None)
            if isinstance(rs, dict) and rs:
                parts = []
                for v in rs.values():
                    if not torch.is_tensor(v):
                        continue
                    v = v.detach().cpu().float()
                    # pool the head_dim axis of (batch, heads, head_dim, state)
                    # -> keeps head x state structure, bounds feature size
                    v = v.mean(dim=-2) if v.ndim >= 3 else v
                    parts.append(v.reshape(-1))
                if parts:
                    sel.append(torch.cat(parts).numpy().astype(np.float16))
        if sel:
            rstate = np.stack(sel)
    else:
        cache = getattr(out, "cache_params", None)
        cand = getattr(cache, "ssm_states", None) if cache is not None else None
        st = getattr(out, "state", None)
        if cand is not None:
            sel = [cand[i].detach().cpu().reshape(-1).numpy().astype(np.float16)
                   for i in layer_idx if i < len(cand)]
            if sel:
                rstate = np.stack(sel)
        elif st is not None:
            if len(st) > 0 and torch.is_tensor(st[0]) and st[0].ndim == 3:
                # rwkv in v5: state packed as component tensors (batch, hidden, n_layers)
                # -> per-layer vector = concat of each component's slice at layer i
                sel = []
                for i in layer_idx:
                    if i >= st[0].shape[-1]:
                        continue
                    vec = torch.cat([s[0, :, i].detach().cpu() for s in st])
                    sel.append(vec.numpy().astype(np.float16))
            else:
                sel = [st[i].detach().cpu().reshape(-1).numpy().astype(np.float16)
                       for i in layer_idx if i < len(st)]
            if sel:
                rstate = np.stack(sel)

    return hidden, rstate


@torch.no_grad()
def greedy_answer(family, model, tok, enc, max_new):
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    suffix = gen[0][enc["input_ids"].shape[1]:]
    return tok.decode(suffix, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["qwen", "mamba", "mamba2", "falcon_h1", "rwkv"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--input", default="probe_questions.jsonl")
    ap.add_argument("--out-dir", default="states")
    ap.add_argument("--tag", default=None, help="output tag (default: family)")
    ap.add_argument("--layer-mode", default="quartiles", choices=["quartiles", "all"])
    ap.add_argument("--verify-known", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N questions only")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-repo", default=None,
                    help="load tokenizer from a different HF repo (e.g. for weight-only repos)")
    args = ap.parse_args()

    tag = args.tag or args.family
    device = pick_device()
    print(f"device: {device} | family: {args.family} | repo: {args.repo}")

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} questions")

    tok, model = load_family(args.family, args.repo, device, args.trust_remote_code,
                             args.tokenizer_repo)
    n_layers = getattr(model.config, "num_hidden_layers", None) \
        or getattr(model.config, "n_layer", None) \
        or getattr(model.config, "num_layers")
    layer_idx = list(range(n_layers)) if args.layer_mode == "all" else quartile_idx(n_layers)
    print(f"layers: {n_layers} total, probing {layer_idx}")

    os.makedirs(args.out_dir, exist_ok=True)
    ids, hiddens, rstates, verified = [], [], [], []
    verify_log = open(os.path.join(args.out_dir, f"verify_{tag}.jsonl"), "w", encoding="utf-8") \
        if args.verify_known else None

    for r in tqdm(rows, desc=tag):
        prompt = f"User: {r['question']}\nAssistant: "
        enc = tok(prompt, return_tensors="pt").to(device)

        hidden, rstate = forward_features(args.family, model, enc, layer_idx)
        if hidden is None:
            raise RuntimeError("model returned no hidden_states; cannot probe")
        ids.append(r["id"])
        hiddens.append(hidden)
        rstates.append(rstate if rstate is not None
                       else np.zeros((len(layer_idx), 0), dtype=np.float16))

        ok = False
        if args.verify_known and r["label"] == "known" and r.get("gold_answer"):
            completion = greedy_answer(args.family, model, tok, enc, args.max_new_tokens)
            ok = r["gold_answer"].lower() in completion.lower()
            verify_log.write(json.dumps({
                "id": r["id"], "question": r["question"],
                "gold": r["gold_answer"], "completion": completion,
                "verified": ok}, ensure_ascii=False) + "\n")
        verified.append(ok)

    if verify_log:
        verify_log.close()

    out_path = os.path.join(args.out_dir, f"states_{tag}.npz")
    np.savez(out_path,
             ids=np.array(ids, dtype=np.int64),
             hidden=np.stack(hiddens),
             rstate=np.stack(rstates),
             layer_idx=np.array(layer_idx, dtype=np.int64),
             verified=np.array(verified, dtype=bool))
    size_mb = os.path.getsize(out_path) / 1e6
    n_ver = sum(1 for r, v in zip(rows, verified) if r["label"] == "known" and v)
    n_known = sum(1 for r in rows if r["label"] == "known")
    print(f"\nsaved -> {out_path} ({size_mb:.0f} MB)")
    print(f"hidden {np.stack(hiddens).shape} | rstate {np.stack(rstates).shape}")
    if args.verify_known:
        print(f"verified known: {n_ver}/{n_known}")


if __name__ == "__main__":
    main()