#!/usr/bin/env python3
"""
render_sft_dataset.py — Render the Stage-2 SFT dataset in Heartly grammar.

Format (reason-then-decide, same as the v2/v3 line):
  known   : <think> {reasoning} </think><decide>speak</decide><verify>known</verify> {answer} <stop>
  unknown : <think> {reasoning} </think><decide>speak</decide><verify>unknown</verify> {abstention} <stop>
  silence : <think> {reasoning} </think><decide>stop</decide>

Design notes (paper §4): kind-aware answers (no "The X of Y is Z" templates),
provenance attached rarely (~10%), multiple abstention phrasings, known side
from TRAIN splits (the probe eval set uses validation — no leakage).

Output: sft_dataset.jsonl — {instruction, output}
"""
import argparse
import json
import random

from gen_probe_dataset import (gen_fabricated, gen_type_mismatch, gen_post_cutoff,
                               gen_depth2, gen_structural)

REASON_KNOWN = [
    "This is a factual question. I have this in my knowledge. I will speak.",
    "I know this fact. I can answer confidently. I will speak.",
    "Checking my knowledge... found it. I should answer.",
]
REASON_UNKNOWN = [
    "I have no information about this. I should say I don't know rather than guess.",
    "Checking my knowledge... I find nothing. Guessing would risk being wrong.",
    "This is not something I know. The honest response is to say I don't have this information.",
]
REASON_SILENCE = [
    "The input is empty or noise. There is nothing meaningful to respond to. I will stay silent.",
    "No question was asked. Speaking would add nothing. I will stay silent.",
]
ABSTAIN = [
    "I do not have information about that.",
    "I don't have that information.",
    "I do not know.",
    "I don't know.",
]
SILENCE_TRIGGERS = ["", " ", "...", "..", "hey", "hi", "hello", "hello?", "yo",
                    "hm", "hmm", "uh", "um", "ok", "okay", "speak to me",
                    "say something", "???", "!!", ".", "nothing", "nevermind",
                    "nvm", "just checking", "test", "are you there", "ping"]


def fmt(decision, verification=None, answer=None, reasoning=None):
    think = f"<think> {reasoning} </think>" if reasoning else ""
    if decision == "stop":
        return f"{think}<decide>stop</decide>"
    return f"{think}<decide>speak</decide><verify>{verification}</verify> {answer} <stop>"


def load_known_train(rng, n_squad, n_trivia):
    from datasets import load_dataset
    known = []

    def short(a):
        return a and len(a.split()) <= 6

    try:
        sq = load_dataset("rajpurkar/squad", split="train[:30000]")
        idx = list(range(len(sq)))
        rng.shuffle(idx)
        for i in idx:
            ex = sq[i]
            ans = ex.get("answers", {}).get("text", [""])
            ans = ans[0] if ans else ""
            if short(ans):
                known.append((ex["question"].strip(), "squad", ans))
            if len(known) >= n_squad:
                break
        print(f"[OK] squad train: {min(n_squad, len(known))}")
    except Exception as e:
        print(f"[SKIP] squad: {e}")

    got = 0
    try:
        tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train[:30000]")
        idx = list(range(len(tq)))
        rng.shuffle(idx)
        for i in idx:
            ex = tq[i]
            ans = ex.get("answer", {}).get("value", "")
            if short(ans):
                known.append((ex["question"].strip(), "trivia_qa", ans))
                got += 1
            if got >= n_trivia:
                break
        print(f"[OK] trivia_qa train: {got}")
    except Exception as e:
        print(f"[SKIP] trivia_qa: {e}")
    return known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=4000)
    ap.add_argument("--unknown", type=int, default=2400)
    ap.add_argument("--silence", type=int, default=250)
    ap.add_argument("--provenance-ratio", type=float, default=0.10)
    ap.add_argument("--out", default="sft_dataset.jsonl")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    for q, src, ans in load_known_train(rng, args.known // 2, args.known - args.known // 2):
        if rng.random() < args.provenance_ratio:
            text = f"{ans} (Source: {src})."
        else:
            text = rng.choice([f"{ans}", f"{ans}.", f"The answer is {ans}."])
        rows.append({"instruction": q,
                     "output": fmt("speak", "known", text, rng.choice(REASON_KNOWN))})

    per = args.unknown // 5
    unknown = (gen_fabricated(rng, per) + gen_type_mismatch(rng, per)
               + gen_post_cutoff(rng, per) + gen_depth2(rng, per)
               + gen_structural(rng, args.unknown - 4 * per))
    for q, _g, _ in unknown:
        rows.append({"instruction": q,
                     "output": fmt("speak", "unknown", rng.choice(ABSTAIN),
                                   rng.choice(REASON_UNKNOWN))})

    for i in range(args.silence):
        trig = SILENCE_TRIGGERS[i % len(SILENCE_TRIGGERS)]
        rows.append({"instruction": trig,
                     "output": fmt("stop", reasoning=rng.choice(REASON_SILENCE))})

    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} SFT samples -> {args.out}")
    for r in rows[:3] + rows[-2:]:
        print(f"\n--- {r['instruction'][:60]!r}\n{r['output'][:200]}")


if __name__ == "__main__":
    main()