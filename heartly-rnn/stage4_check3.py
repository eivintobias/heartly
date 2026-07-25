#!/usr/bin/env python3
"""
stage4_check3.py — the missing control + the disposition probe.
Follow-up to run2 (mech PASS / gist quiz 0-5): separates
  (a) state lacks the content            (state persistence is real but
                                          shallow: distribution, not facts)
  (b) disposition overrides the content  (the abstention training refuses
                                          personal-context questions even
                                          when the fact is in the state)

Sections:
  1. LIVE-cache quiz (no reload — the missing control). If live also
     abstains, save/load is not the bottleneck.
  2. Primed content probe: force the answer slot open with a prefix
     ("The user said: my dog's name is") and see if the fact completes.
     Retrieves content past the disposition gate.
  3. QA-style teach: facts taught as question/ANSWER pairs instead of
     declarations, then quiz again — tests whether the write format
     decides retrievability.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "eivintobias/heartly-rwkv7-1.5b"
TOKENIZER_REPO = "RWKV/RWKV7-Goose-World3-1.5B-HF"
STOP_IDS = [61, 27081, 63]
MAX_NEW = 100

ACK = ("<think>The user shared a personal fact.</think>"
       "<decide>speak</decide><verify>known</verify>Noted, thank you.<stop>")

FACTS = [
    {"key": "dog",      "teach": "My dog's name is Zorblax.",
     "question": "What is my dog's name?",
     "answer": "Your dog's name is Zorblax.", "gold": "zorblax"},
    {"key": "codename", "teach": "The secret project codename is Velvet Aurora.",
     "question": "What is the secret project codename?",
     "answer": "The secret project codename is Velvet Aurora.",
     "gold": "velvet aurora"},
    {"key": "number",   "teach": "My favorite number is 7,423.",
     "question": "What is my favorite number?",
     "answer": "Your favorite number is 7,423.", "gold": "7,423"},
    {"key": "collect",  "teach": "I collect miniature lighthouse replicas.",
     "question": "What do I collect?",
     "answer": "You collect miniature lighthouse replicas.",
     "gold": "lighthouse"},
    {"key": "password", "teach": "The password to the lab door is mango Tuesday.",
     "question": "What is the lab door password?",
     "answer": "The lab door password is mango Tuesday.",
     "gold": "mango tuesday"},
]


def teach_decl():
    return "".join(f"User: {f['teach']}\nAssistant: {ACK}\n" for f in FACTS)


def teach_qa():
    parts = []
    for f in FACTS:
        parts.append(f"User: {f['question']}\nAssistant: "
                     f"<think>The user asks a personal fact.</think>"
                     f"<decide>speak</decide><verify>known</verify>"
                     f"{f['answer']}<stop>\n")
    return "".join(parts)


@torch.no_grad()
def prefill(model, ids, cache=None):
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
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
    return out_ids, cache


def ask(model, tok, cache, prompt, max_new=MAX_NEW):
    ids = tok(prompt, return_tensors="pt").to(prompt_ids_device)["input_ids"]
    out_ids, _ = greedy(model, ids, cache, max_new)
    return tok.decode(out_ids, skip_special_tokens=False)


def main():
    global prompt_ids_device
    device = "cuda"
    prompt_ids_device = torch.device(device)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device)
    model.eval()

    decl_ids = tok(teach_decl(), return_tensors="pt").to(device)["input_ids"]
    qa_ids = tok(teach_qa(), return_tensors="pt").to(device)["input_ids"]

    # ---------- 1. LIVE-cache quiz (the missing control) ----------
    print("\n== 1. LIVE-cache quiz (declarative teach, NO reload) ==", flush=True)
    live = prefill(model, decl_ids)
    live_hits = 0
    for f in FACTS:
        # fresh live cache per question (all start from the same taught state)
        c = prefill(model, decl_ids)
        txt = ask(model, tok, c, f"User: {f['question']}\nAssistant:")
        hit = f["gold"] in txt.lower()
        live_hits += int(hit)
        print(f"-- {f['question']}\n   live: {txt.strip()[:130]!r} | hit {hit}",
              flush=True)
    print(f"LIVE declarative quiz: {live_hits}/5", flush=True)

    # ---------- 2. Primed content probe ----------
    print("\n== 2. Primed content probe (answer slot forced open) ==", flush=True)
    probe_hits = 0
    probes = {
        "dog":      "User: What is my dog's name?\nAssistant: <think>The user said their dog's name earlier. The name is",
        "codename": "User: What is the secret project codename?\nAssistant: <think>The user told me the codename earlier. It is",
        "number":   "User: What is my favorite number?\nAssistant: <think>The user told me the number earlier. It is",
        "collect":  "User: What do I collect?\nAssistant: <think>The user told me what they collect. They collect",
        "password": "User: What is the lab door password?\nAssistant: <think>The user told me the password earlier. It is",
    }
    for f in FACTS:
        c = prefill(model, decl_ids)
        txt = ask(model, tok, c, probes[f["key"]], max_new=25)
        hit = f["gold"] in txt.lower()
        probe_hits += int(hit)
        print(f"-- {f['key']}: {txt.strip()[:90]!r} | hit {hit}", flush=True)
    print(f"primed probe: {probe_hits}/5", flush=True)

    # ---------- 3. QA-style teach, then quiz ----------
    print("\n== 3. QA-style teach (facts as Q/A pairs), then quiz ==", flush=True)
    qa_hits = 0
    for f in FACTS:
        c = prefill(model, qa_ids)
        txt = ask(model, tok, c, f"User: {f['question']}\nAssistant:")
        hit = f["gold"] in txt.lower()
        qa_hits += int(hit)
        print(f"-- {f['question']}\n   live: {txt.strip()[:130]!r} | hit {hit}",
              flush=True)
    print(f"QA-teach quiz: {qa_hits}/5", flush=True)

    print(f"\n==== READ ====")
    print(f"live-declarative {live_hits}/5 | primed {probe_hits}/5 | "
          f"QA-teach {qa_hits}/5")
    print("live==0 & primed==0 -> state content is shallow everywhere; "
          "live==0 & primed>0 -> disposition overrides content; "
          "QA>decl -> write format decides retrievability")


if __name__ == "__main__":
    main()