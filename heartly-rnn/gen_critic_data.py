#!/usr/bin/env python3
"""
gen_critic_data.py — Stage 2.5 phase 1: build the critic dataset.

Runs the fine-tuned RWKV (stage2_results/rwkv-heartly) over the probe
questions, greedy-decodes the full Heartly-grammar transcript, and labels
each SPOKEN answer as correct-content (1) or confabulation (0):

  say=unknown                       -> abstain (excluded from critic training)
  say=known & true=known & gold in answer -> correct (1)
  say=known & true=known & gold missing   -> confab_content (0)
  say=known & true=unknown          -> confab_unknown (0)
  no <verify> token                 -> unparsed

The critic's deployment question is: "the model spoke — should we trust the
content?" So only say=known rows carry a training label; abstains/unparsed
are recorded for statistics.

Rows from the Stage 2 report's confabulations (say=known & true=unknown) are
tagged `tracked: true` — the pre-registered must-catch set for phase 2.

Generation is single-prompt greedy with early stop at the 3-token <stop>
sequence. Batched generation was tried and abandoned: the custom RWKV
tokenizer/model does not respect attention_mask in the recurrence, so pad
tokens poison the state (all-'<<<' garbage output).

Resume-safe: existing rows in --out are skipped, new rows appended.

Output: critic_data.jsonl {id, question, generator, true, say, answer,
        full_text, label, row_class, tracked}
"""
import argparse
import json
import os
import re
import time

import torch
from tqdm import tqdm

VERIFY_RE = re.compile(r"<verify>\s*(known|unknown)\s*</verify>")
STOP_SEQ = [61, 27081, 63]  # "<stop>" in the RWKV world vocab


def load_model(path, tokenizer_repo=None, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_repo or path,
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype,
                                                 trust_remote_code=True)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"model on {dev}")
    return tok, model


from transformers import StoppingCriteria, StoppingCriteriaList


class StopAtSeq(StoppingCriteria):
    """Halt generation once the last tokens equal STOP_SEQ (per row)."""

    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def __call__(self, input_ids, scores, **kwargs):
        n = len(self.seq)
        if input_ids.shape[1] < n:
            return False
        return all(row[-n:].tolist() == self.seq for row in input_ids)


@torch.no_grad()
def generate(tok, model, prompt, max_new, stoppers):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id
                         if tok.pad_token_id is not None else 0,
                         stopping_criteria=stoppers)
    text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    stop = text.find("<stop>")
    return text[:stop] if stop != -1 else text


def stage2_error_ids(report_path):
    """IDs of Stage 2 confabulations (say=known,true=unknown) + over-refusals."""
    confabs, overrefs = set(), set()
    if not os.path.exists(report_path):
        return confabs, overrefs
    rep = json.load(open(report_path, encoding="utf-8"))
    for x in rep["results"]:
        if x["say"] == "known" and x["true"] == "unknown":
            confabs.add(x["id"])
        if x["say"] == "unknown" and x["true"] == "known":
            overrefs.add(x["id"])
    return confabs, overrefs


def label_row(r, text):
    m = VERIFY_RE.search(text)
    say = m.group(1) if m else None
    answer = text[m.end():].strip() if m else ""
    if say is None:
        row_class, label = "unparsed", None
    elif say == "unknown":
        row_class, label = "abstain", None
    elif r["label"] == "unknown":
        row_class, label = "confab_unknown", 0
    else:
        gold = (r.get("gold_answer") or "").strip().lower()
        ok = bool(gold) and gold in answer.lower()
        row_class, label = ("correct", 1) if ok else ("confab_content", 0)
    return say, answer, row_class, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stage2_results/rwkv-heartly")
    ap.add_argument("--questions", default="probe_questions.jsonl")
    ap.add_argument("--report", default="stage2_results/say_sense_report.json")
    ap.add_argument("--out", default="critic_data.jsonl")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N questions")
    ap.add_argument("--tokenizer-repo", default=None,
                    help="load tokenizer from a different repo/dir (e.g. the "
                         "base RWKV repo) — the fine-tuned dir's saved "
                         "tokenizer is broken")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="model load dtype — fla/RWKV7 kernels require bfloat16")
    args = ap.parse_args()

    stoppers = StoppingCriteriaList([StopAtSeq(STOP_SEQ)])

    tok, model = load_model(args.model, args.tokenizer_repo,
                            getattr(torch, args.dtype))

    rows = [json.loads(l) for l in open(args.questions, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[: args.limit]
    confab_ids, overref_ids = stage2_error_ids(args.report)
    print(f"{len(rows)} questions | stage2 confab ids: {sorted(confab_ids)} "
          f"| over-refusal ids: {sorted(overref_ids)}")

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            done.add(json.loads(l)["id"])
        if done:
            print(f"resume: {len(done)} rows already in {args.out}")

    fout = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n_new = 0
    try:
        for r in tqdm(rows, desc="critic gen"):
            if r["id"] in done:
                continue
            prompt = f"User: {r['question']}\nAssistant: "
            text = generate(tok, model, prompt, args.max_new, stoppers)
            say, answer, row_class, label = label_row(r, text)
            fout.write(json.dumps({"id": r["id"], "question": r["question"],
                                   "generator": r["generator"], "true": r["label"],
                                   "say": say, "answer": answer, "full_text": text,
                                   "label": label, "row_class": row_class,
                                   "tracked": r["id"] in confab_ids},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            n_new += 1
            if n_new == 10:
                dt = time.time() - t0
                print(f"first 10: {dt:.1f}s ({dt / 10:.2f}s/q -> "
                      f"full run ~{dt / 10 * 2902 / 3600:.1f}h)")
    finally:
        fout.close()

    from collections import Counter
    c = Counter()
    for l in open(args.out, encoding="utf-8"):
        c[json.loads(l)["row_class"]] += 1
    total = time.time() - t0
    print(f"\n{args.out}: {sum(c.values())} rows (+{n_new} new, {total:.0f}s)")
    for k, n in sorted(c.items()):
        print(f"  {k:15s} {n}")
    n_conf = c["confab_unknown"] + c["confab_content"]
    print(f"critic training pool: {c['correct']} correct / {n_conf} confabulations "
          f"({c['confab_unknown']} unknown, {c['confab_content']} content)")


if __name__ == "__main__":
    main()