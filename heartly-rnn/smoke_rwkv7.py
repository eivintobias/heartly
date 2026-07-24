#!/usr/bin/env python3
"""
smoke_rwkv7.py — pre-flight check for RWKV7-Goose-World3-1.5B-HF on the
vast.ai instance, BEFORE the paid training run. Each section is independent
(try/except) so one failure doesn't hide the rest. We want answers to:

  1. does flash-linear-attention + triton import and the model load?
  2. does the world tokenizer round-trip the Heartly grammar markers?
  3. do parameter names match the `blocks.N.` freeze regex in finetune_rwkv.py?
  4. what does the forward pass return (hidden_states? what cache object?) —
     needed to adapt extract_states.forward_features for the boundary head
  5. is attention_mask honored in batched (padded) forwards? (the rwkv-4
     custom model silently ignored it and poisoned outputs)
  6. does greedy generate produce sane text?

Usage:  python smoke_rwkv7.py            (fp16 on GPU)
        python smoke_rwkv7.py --cpu      (fp32 on CPU, slow but safe)
"""
import argparse
import traceback

REPO = "RWKV/RWKV7-Goose-World3-1.5B-HF"


def section(name):
    print(f"\n{'='*60}\n== {name}\n{'='*60}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    section("0. versions")
    import torch
    import transformers
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0),
              f"| vram {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("transformers", transformers.__version__)
    try:
        import triton
        print("triton", triton.__version__)
    except Exception as e:
        print("triton MISSING:", e)
    try:
        import fla
        print("fla", getattr(fla, "__version__", "?"))
    except Exception as e:
        print("fla MISSING:", e)
        print("-> pip install flash-linear-attention  (then re-run)")
        return

    section("1. tokenizer")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
        print("class:", type(tok).__name__)
        sample = ("User: test\nAssistant: <think> r </think>"
                  "<decide>speak</decide><verify>known</verify>Paris<stop>")
        ids = tok.encode(sample)
        back = tok.decode(ids, skip_special_tokens=False)
        print("n ids:", len(ids), "| roundtrip ok:", back.replace("\n", "") ==
              sample.replace("\n", "") or sample in back)
        print("decoded:", back[:160])
        stop_ids = tok.encode("<stop>")
        print("'<stop>' ids:", stop_ids, "(rwkv-4 was [61, 27081, 63])")
    except Exception:
        traceback.print_exc()
        return

    section("2. model load")
    try:
        from transformers import AutoModelForCausalLM
        if args.cpu:
            model = AutoModelForCausalLM.from_pretrained(
                REPO, dtype=torch.float32, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                REPO, dtype=torch.float16, trust_remote_code=True,
                device_map="cuda")
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        print(f"loaded OK | {n/1e9:.2f}B params | device: {next(model.parameters()).device}")
        cfg = model.config
        print("layers:", cfg.num_hidden_layers, "| hidden:", cfg.hidden_size,
              "| vocab:", cfg.vocab_size)
    except Exception:
        traceback.print_exc()
        return

    section("3. freeze-regex check (finetune_rwkv.py expects 'blocks.N.')")
    import re
    names = [n for n, _ in model.named_parameters()]
    pat = re.compile(r"blocks\.(\d+)\.")
    hits = [n for n in names if pat.search(n)]
    print(f"{len(hits)}/{len(names)} tensors match 'blocks.N.'")
    print("first 12 param names:")
    for n in names[:12]:
        print("  ", n)

    section("4. forward-pass outputs (for extract_states adaptation)")
    try:
        enc = tok("User: What is the capital of France?\nAssistant: <verify>",
                  return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc, use_cache=True, output_hidden_states=True)
        print("output type:", type(out).__name__)
        print("keys:", list(out.keys()) if hasattr(out, "keys") else dir(out))
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            print(f"hidden_states: {len(hs)} entries, each {tuple(hs[0].shape)}")
        else:
            print("hidden_states: NONE")
        pkv = getattr(out, "past_key_values", None)
        print("past_key_values type:", type(pkv).__name__)
        if pkv is not None:
            attrs = [a for a in dir(pkv) if not a.startswith("_")]
            print("past_key_values attrs:", attrs)
            layers = getattr(pkv, "layers", None)
            if layers:
                print(f"pkv.layers: {len(layers)} | layer0 attrs:",
                      [a for a in dir(layers[0]) if not a.startswith("_")])
                rs = getattr(layers[0], "recurrent_states", None)
                if rs is not None:
                    print("layer0 recurrent_states:",
                          {k: tuple(v.shape) for k, v in rs.items()}
                          if isinstance(rs, dict) else type(rs).__name__)
            elif isinstance(pkv, (list, tuple)):
                print(f"pkv is list/tuple len {len(pkv)} | item0 type:",
                      type(pkv[0]).__name__)
                if len(pkv) and pkv[0] is not None:
                    print("item0 attrs:",
                          [a for a in dir(pkv[0]) if not a.startswith("_")])
        st = getattr(out, "state", None)
        print("out.state:", type(st).__name__ if st is not None else None)
    except Exception:
        traceback.print_exc()

    section("5. attention_mask honored? (padded batch vs single)")
    try:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        text = "User: 2+2?\nAssistant:"
        ids = tok(text, return_tensors="pt")
        n_pad = 5
        import torch as T
        padded = T.cat([ids["input_ids"],
                        T.full((1, n_pad), pad_id, dtype=T.long)], dim=1)
        mask = T.cat([ids["attention_mask"],
                      T.zeros((1, n_pad), dtype=T.long)], dim=1)
        dev = model.device
        with T.no_grad():
            out_single = model(**ids.to(dev)).logits[0, -1, :]
            out_padded = model(input_ids=padded.to(dev),
                               attention_mask=mask.to(dev)).logits[0, len(ids["input_ids"][0]) - 1, :]
        cos = T.nn.functional.cosine_similarity(
            out_single.float(), out_padded.float(), dim=0).item()
        print(f"logit cosine similarity at last real token: {cos:.4f}")
        print("OK — mask honored" if cos > 0.99 else
              "WARNING — padding leaks into the recurrence (single-prompt only!)")
    except Exception:
        traceback.print_exc()

    section("6. greedy generation")
    try:
        enc = tok("User: What is the capital of France?\nAssistant: ",
                  return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=30, do_sample=False,
                                 pad_token_id=tok.pad_token_id
                                 if tok.pad_token_id is not None else 0)
        print(tok.decode(gen[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=False))
    except Exception:
        traceback.print_exc()

    section("SMOKE TEST DONE — review the sections above")


if __name__ == "__main__":
    main()