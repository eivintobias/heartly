#!/usr/bin/env python3
"""
stage4c_write_gate.py — Stage 4c Part A: write-gate formats (re-run on v2 model).

Identical to stage4b_write_gate.py except MODEL_REPO points at the v2 retrain.
Same 5 facts, same 6 write formats, same hit rule (gold substring).
Output: stage4c_results/write_gate_report.json + stdout log.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "rwkv7-heartly-v2"  # <-- the retrained model (local dir on instance)
TOKENIZER_REPO = "RWKV/RWKV7-Goose-World3-1.5B-HF"
STOP_IDS = [61, 27081, 63]
MAX_NEW = 100
OUT_DIR = "stage4c_results"

ACK = ("<tool_call>The user shared a personal fact. </think>"
       "<decide>speak</decide><verify>known</verify>Noted, thank you.<stop>")

FACTS = [
    {"key": "dog",      "teach": "My dog's name is Zorblax.",
     "question": "What is my dog's name?",
     "third_q": "What is the name of the user's dog?",
     "answer": "Your dog's name is Zorblax.",
     "third": "The name of the user's dog is Zorblax.",
     "gold": "zorblax"},
    {"key": "codename", "teach": "The secret project codename is Velvet Aurora.",
     "question": "What is the secret project codename?",
     "third_q": "What is the secret project codename?",
     "answer": "The secret project codename is Velvet Aurora.",
     "third": "The secret project codename is Velvet Aurora.",
     "gold": "velvet aurora"},
    {"key": "number",   "teach": "My favorite number is 7,423.",
     "question": "What is my favorite number?",
     "third_q": "What is the user's favorite number?",
     "answer": "Your favorite number is 7,423.",
     "third": "The user's favorite number is 7,423.",
     "gold": "7,423"},
    {"key": "collect",  "teach": "I collect miniature lighthouse replicas.",
     "question": "What do I collect?",
     "third_q": "What does the user collect?",
     "answer": "You collect miniature lighthouse replicas.",
     "third": "The user collects miniature lighthouse replicas.",
     "gold": "lighthouse"},
    {"key": "password", "teach": "The password to the lab door is mango Tuesday.",
     "question": "What is the lab door password?",
     "third_q": "What is the lab door password?",
     "answer": "The lab door password is mango Tuesday.",
     "third": "The password to the lab door is mango Tuesday.",
     "gold": "mango tuesday"},
]


def grammar_answer(text):
    return (f"<tool_call>The user asks a personal fact. </think>"
            f"<decide>speak</decide><verify>known</verify>{text}<stop>")


# ---------------------------------------------------------- write formats
def w1_declarative():
    return "".join(f"User: {f['teach']}\nAssistant: {ACK}\n" for f in FACTS)


def w2_qa():
    return "".join(f"User: {f['question']}\nAssistant: "
                   f"{grammar_answer(f['answer'])}\n" for f in FACTS)


def w3_assistant_voice():
    parts = []
    for f in FACTS:
        parts.append(
            f"User: {f['teach']}\nAssistant: "
            f"<tool_call>The user shared a fact. I now know this. </think>"
            f"<decide>speak</decide><verify>known</verify>"
            f"{f['third']} I will remember that.<stop>\n")
    return "".join(parts)


def w4_trivia():
    parts = []
    for f in FACTS:
        parts.append(f"User: {f['third_q']}\nAssistant: "
                     f"<tool_call>This is a factual question. I have this in my "
                     f"knowledge. I will speak. </think>"
                     f"<decide>speak</decide><verify>known</verify>"
                     f"{f['third']}<stop>\n")
    return "".join(parts)


def w5_qa_x3():
    return w2_qa() * 3


def w6_combined():
    return w3_assistant_voice() + w4_trivia() + w2_qa()


WRITES = [
    ("W1_declarative", w1_declarative, "question"),
    ("W2_qa", w2_qa, "question"),
    ("W3_assistant_voice", w3_assistant_voice, "question"),
    ("W4_trivia_personalQ", w4_trivia, "question"),
    ("W4_trivia_thirdQ", w4_trivia, "third_q"),
    ("W5_qa_x3", w5_qa_x3, "question"),
    ("W6_combined", w6_combined, "question"),
]


# ------------------------------------------------------------- model bits
@torch.no_grad()
def prefill(model, ids):
    out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


@torch.no_grad()
def greedy(model, prompt_ids, cache, max_new=MAX_NEW):
    cur = prompt_ids
    out_ids = []
    for _ in range(max_new):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = int(out.logits[0, -1, :].argmax().item())
        out_ids.append(nxt)
        cur = torch.tensor([[nxt]], device=prompt_ids.device,
                           dtype=prompt_ids.dtype)
        if len(out_ids) >= len(STOP_IDS) and out_ids[-len(STOP_IDS):] == STOP_IDS:
            break
    return out_ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device)
    model.eval()

    report = {"model": MODEL_REPO, "formats": {}}
    for name, builder, qfield in WRITES:
        teach_text = builder()
        teach_ids = tok(teach_text, return_tensors="pt").to(device)["input_ids"]
        print(f"\n== {name} (teach {teach_ids.shape[1]} tokens) ==", flush=True)
        hits = 0
        rows = []
        for f in FACTS:
            cache = prefill(model, teach_ids)     # fresh per question
            q = f[qfield]
            q_ids = tok(f"User: {q}\nAssistant:",
                        return_tensors="pt").to(device)["input_ids"]
            out_ids = greedy(model, q_ids, cache)
            txt = tok.decode(out_ids, skip_special_tokens=False)
            hit = f["gold"] in txt.lower()
            hits += int(hit)
            rows.append({"key": f["key"], "question": q,
                         "output": txt.strip(), "hit": hit})
            print(f"-- {q}\n   {txt.strip()[:130]!r} | hit {hit}", flush=True)
        print(f"{name}: {hits}/5", flush=True)
        report["formats"][name] = {"hits": hits, "rows": rows}

    print("\n==== SUMMARY ====")
    for name, r in report["formats"].items():
        print(f"{name}: {r['hits']}/5")
    with open(os.path.join(OUT_DIR, "write_gate_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_DIR}/write_gate_report.json")


if __name__ == "__main__":
    main()
