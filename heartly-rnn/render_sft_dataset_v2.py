#!/usr/bin/env python3
"""
render_sft_dataset_v2.py — Stage 4c SFT dataset: original classes + two new ones.

Extends render_sft_dataset.py with:
  (i)   context-known rendered class (~1,000 samples from SQuAD with context)
  (ii)  multi-turn conversation class (~500 samples, 2–4 turn chains)

The original classes (known question-only, unknown, silence) stay at their
Stage-2 sizes. Total: ~8,150 samples (was 6,031).

Output: sft_dataset_v2.jsonl — {instruction, output}
"""
import argparse
import json
import random

from gen_probe_dataset import (gen_fabricated, gen_type_mismatch, gen_post_cutoff,
                               gen_depth2, gen_structural)
from render_sft_dataset import (REASON_KNOWN, REASON_UNKNOWN, REASON_SILENCE,
                                ABSTAIN, SILENCE_TRIGGERS, fmt,
                                load_known_train)


# ---- new reasoning templates for the context-known class ----
REASON_CONTEXT_KNOWN = [
    "I can find the answer in the provided context. I will speak.",
    "The context tells me this. I know the answer. I will speak.",
    "Based on the provided information, I know this. I should answer.",
    "The provided context contains the answer. I will speak.",
    "I see the answer in the context. I can respond confidently.",
]


def load_squad_with_context(rng, n):
    """Load SQuAD train split WITH context paragraphs for the context-known class."""
    from datasets import load_dataset
    rows = []

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
            if not short(ans):
                continue
            context = ex.get("context", "").strip()
            question = ex["question"].strip()
            # Skip very long contexts (would blow past max-length 256)
            if len(context.split()) > 120:
                continue
            rows.append({
                "context": context,
                "question": question,
                "answer": ans,
            })
            if len(rows) >= n:
                break
        print(f"[OK] squad context-known: {len(rows)}")
    except Exception as e:
        print(f"[SKIP] squad context-known: {e}")
    return rows


def render_context_known(rng, n_samples=1000, provenance_ratio=0.10):
    """Render the context-known class: context + question → speak/known."""
    rows = []
    squad_rows = load_squad_with_context(rng, n_samples)
    for item in squad_rows:
        instruction = f"Context: {item['context']}\n{item['question']}"
        if rng.random() < provenance_ratio:
            text = f"{item['answer']} (Source: context)."
        else:
            text = rng.choice([f"{item['answer']}", f"{item['answer']}.",
                               f"The answer is {item['answer']}."])
        output = fmt("speak", "known", text, rng.choice(REASON_CONTEXT_KNOWN))
        rows.append({"instruction": instruction, "output": output})
    return rows


def render_multi_turn(rng, known_pool, unknown_pool, n_samples=500):
    """Render multi-turn conversation chains (2–4 turns per sample).

    Only the LAST assistant turn goes in the output field; earlier turns
    are part of the instruction so the loss only trains on the final response.
    """
    rows = []
    for _ in range(n_samples):
        n_turns = rng.choice([2, 2, 3, 3, 4])  # bias toward 2–3
        turns = []
        for t in range(n_turns):
            if rng.random() < 0.5 and known_pool:
                q, src, ans = rng.choice(known_pool)
                if rng.random() < 0.10:
                    text = f"{ans} (Source: {src})."
                else:
                    text = rng.choice([f"{ans}", f"{ans}.",
                                       f"The answer is {ans}."])
                turns.append(("known", q, text))
            else:
                q, g, _ = rng.choice(unknown_pool)
                text = rng.choice(ABSTAIN)
                turns.append(("unknown", q, text))

        # Build instruction: all turns except the last assistant response
        instruction_parts = []
        for i, (label, q, ans) in enumerate(turns):
            instruction_parts.append(f"User: {q}")
            if i < len(turns) - 1:
                # Earlier turns: full grammar output in the instruction
                if label == "known":
                    reasoning = rng.choice(REASON_KNOWN)
                else:
                    reasoning = rng.choice(REASON_UNKNOWN)
                full_ans = fmt("speak", label, ans, reasoning)
                instruction_parts.append(f"Assistant: {full_ans}")
            else:
                instruction_parts.append("Assistant:")
        instruction = "\n".join(instruction_parts)

        # Output: only the last turn's response
        last_label, _, last_ans = turns[-1]
        if last_label == "known":
            reasoning = rng.choice(REASON_KNOWN)
        else:
            reasoning = rng.choice(REASON_UNKNOWN)
        output = fmt("speak", last_label, last_ans, reasoning)

        rows.append({"instruction": instruction, "output": output})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=4000)
    ap.add_argument("--unknown", type=int, default=2400)
    ap.add_argument("--silence", type=int, default=250)
    ap.add_argument("--context-known", type=int, default=1000)
    ap.add_argument("--multi-turn", type=int, default=500)
    ap.add_argument("--provenance-ratio", type=float, default=0.10)
    ap.add_argument("--out", default="sft_dataset_v2.jsonl")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    # ---- original classes (same as render_sft_dataset.py) ----
    known_pool = load_known_train(rng, args.known // 2, args.known - args.known // 2)
    for q, src, ans in known_pool:
        if rng.random() < args.provenance_ratio:
            text = f"{ans} (Source: {src})."
        else:
            text = rng.choice([f"{ans}", f"{ans}.", f"The answer is {ans}."])
        rows.append({"instruction": q,
                     "output": fmt("speak", "known", text, rng.choice(REASON_KNOWN))})

    per = args.unknown // 5
    unknown_pool = (gen_fabricated(rng, per) + gen_type_mismatch(rng, per)
                    + gen_post_cutoff(rng, per) + gen_depth2(rng, per)
                    + gen_structural(rng, args.unknown - 4 * per))
    for q, _g, _ in unknown_pool:
        rows.append({"instruction": q,
                     "output": fmt("speak", "unknown", rng.choice(ABSTAIN),
                                   rng.choice(REASON_UNKNOWN))})

    for i in range(args.silence):
        trig = SILENCE_TRIGGERS[i % len(SILENCE_TRIGGERS)]
        rows.append({"instruction": trig,
                     "output": fmt("stop", reasoning=rng.choice(REASON_SILENCE))})

    # ---- new class (i): context-known ----
    context_rows = render_context_known(rng, args.context_known, args.provenance_ratio)
    rows.extend(context_rows)
    print(f"context-known class: {len(context_rows)} samples")

    # ---- new class (ii): multi-turn ----
    mt_rows = render_multi_turn(rng, known_pool, unknown_pool, args.multi_turn)
    rows.extend(mt_rows)
    print(f"multi-turn class: {len(mt_rows)} samples")

    # ---- shuffle and write ----
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary ----
    from collections import Counter
    # Rough class breakdown by output pattern
    n_known = sum(1 for r in rows if "<verify>known</verify>" in r["output"])
    n_unknown = sum(1 for r in rows if "<verify>unknown</verify>" in r["output"])
    n_stop = sum(1 for r in rows if "<decide>stop</decide>" in r["output"])
    n_context = len(context_rows)
    n_mt = len(mt_rows)

    print(f"\nwrote {len(rows)} SFT samples -> {args.out}")
    print(f"  known (question-only): {n_known - n_context}  |  context-known: {n_context}")
    print(f"  unknown: {n_unknown}  |  silence: {n_stop}  |  multi-turn: {n_mt}")
    for r in rows[:3] + rows[-2:]:
        print(f"\n--- {r['instruction'][:80]!r}\n{r['output'][:200]}")


if __name__ == "__main__":
    main()
