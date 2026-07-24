# HANDOFF — Heartly Project (2026-07-21)

Read this first in a fresh chat. Everything you need to continue without the
previous 500k-token conversation.

> **LIVE STATE (2026-07-24 10:20):** Stage 2.6 COMPLETE — asymmetric
> critic dose-response on the OLD 0.43B transcripts (RESULTS.md Stage
> 2.6): AUROC 0.758 (Qwen2.5-0.5B) → 0.826 (1.5B) → 0.845 (3B) as the
> critic:generator ratio goes ~1×→3.5×→7× — the Stage 2.5 asymmetry
> requirement CONFIRMED as a measured trend. Pass bar still fails (58%
> detection @ 5% FPR): the bottleneck is the generator's 60-sample
> correct class, not the critic. New finding: over-strictness at 7× —
> the 3B rates even CORRECT 0.43B answers at median P 0.044, so
> asymmetry has a ceiling. Tracked 5 score 0.000–0.008 at 3B (blind spot
> maximally visible). New tooling: `train_critic.py --dtype` (fp16 for
> big critics), `analyze_critic_any.py` (any-folder operating curve),
> `stage2p5_asym{15,3}_results/` (desktop RTX 2080 Ti ran it all, $0).
> Stage 3 (RWKV7-1.5B fine-tune) COMPLETE + published on HF
> (eivintobias/heartly-rwkv7-1.5b): grammar 100%, decide 100%, boundary
> head AUROC 1.000 at all probed layers. vast.ai instance 45634549:
> DESTROYED (user-confirmed 2026-07-24). NEXT: critic harvest on the
> 1.5B (gen_critic_data.py — PATCH FIRST: no --tokenizer-repo flag yet,
> and it loads fp32 which fla refuses → add bf16; Linux GPU only,
> fla/triton don't run on Windows), then the asymmetric critic re-run on
> the NEW transcripts (the real deployment test).

---

## 0. How to talk in this project (owner request, 2026-07-22)

- **Chat with Eivin: PLAIN language.** Short sentences, concrete examples,
  no unexplained jargon. He has said clearly that researcher-style language
  in chat makes him lose the thread. Explain like to a smart friend.
  Papers stay formal; conversation stays plain.
- **Plain copies of all papers live in `research_papers/plain/`.** Update
  them whenever a paper changes (they are how Eivin re-reads his own
  research). RESULTS.md can stay technical, but each stage's "Decision"
  line must be understandable on its own.

## 1. What this project is

**Heartly** — "nature-first" approach to hallucination reduction. Thesis:
hallucination is a *disposition* problem, not a knowledge problem. Train the
model to **decide whether to speak**, **verify what it knows**, and **admit
ignorance**, compiled into the data itself. Output grammar:

```
<think> [reasoning] </think> <decide> speak|stop </decide> <verify> known|unknown </verify> [answer] <stop>
```

Owner: Eivin (independent researcher). Repo: github.com/eivintobias/heartly
(renamed from heartly-v2 on 2026-07-24 — old URL redirects).
HF: huggingface.co/eivintobias/heartly-v2.

## 2. What's done (chronological)

- **v1** — failed: control tokens inert. Lesson: grammar is cheap, distribution
  is the mechanism.
- **v2** — PUBLISHED (HF + GitHub, 2026-07-19). Qwen2.5-0.5B, 33.5k steps, 12
  datasets ~247k. Known failures documented (entity/attribute collapse, fused
  `speakknown`, repetition loops, confident confabulation e.g. "first US
  president = Charles Lindbergh"). Files: `README.md`, `HF_MODEL_CARD.md`,
  `heartly_test_prompts.md`, `checkpoint-33500/`, `newLLMdesign.ipynb`.
- **v3** — interrupted by Colab disconnect at step 9000/~135k (epoch 0.2/3),
  18 datasets ~1.5M, control acc 0.999 / answer acc ~0.77. In `heartly-v3/`.
- **Research paper II** — `research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md`:
  the full program (boundary error, negative-side mechanisms, memory track).
  Earlier docs: `research_papers/RESEARCH_PAPER.md` (position paper),
  `RESEARCH_BRIEF_TECHNICAL.md` (technical brief, hypotheses H1–H4).
- **Track 2 Experiment 1** (2026-07-20) — boundary head on recurrent state:
  probes on 2,902 true-boundary questions. Recurrent states read known/unknown
  at **AUROC 1.000** (Falcon-H1-0.5B rstate; RWKV-4-430m rstate), transformer
  baseline (Qwen2.5-0.5B hidden) 0.998. ECE ~0.01–0.02. All tables +
  conclusions in `heartly-rnn/RESULTS.md`.
- **Track 2 Stage 2** (2026-07-21) — RWKV-4-World-430m fine-tuned on 6,031
  Heartly-grammar samples (2 epochs, ~6h on vast.ai 3090, ~$2):
  - Grammar adoption **100%** (300/300 parseable verify tokens)
  - Decision accuracy **97.7%** (trivia/depth2/type_mismatch 100%)
  - Deployed boundary head (layer 6) AUROC **1.000**
  - **Key finding:** say/sense agreement 100% BUT the alarm is blind to
    *confident* confabulation — 5 confabulations all had sense_p ≥0.97 because
    head and generator share state. → independent critic channel is the fix
    (paper §5.4, now motivated = **Stage 2.5**).
  - **Live confirmation:** 2 squad "over-refusals" (sense_p 0.001) were corpus
    labels wrong for the model — KB boundary ≠ model boundary (paper §3).
  - Artifacts: `heartly-rnn/stage2_results/` (rwkv-heartly/ 1.85GB,
    probe_head.pkl, say_sense_report.json), analysis in RESULTS.md.
- **Track 2 Stage 2.5** (2026-07-22, COMPLETE) — independent answer critic
  for the confident-confabulation blind spot. Harvest: 2,902 generations,
  1,515 labeled (60 correct / 1,455 confab / 1,387 abstain); content
  accuracy on spoken known-answers **4.1%** vs decision accuracy 97.1% —
  the capability caveat quantified. Critics: A (Qwen2.5-0.5B hidden on full
  transcript) AUROC 0.758, B (RWKV state at end of own generation) 0.795.
  **Verdict vs pre-registered bar: FAIL** (detection 42%/25% @ 5% FPR) —
  ranking works (bottom-5% tail 100% pure), operating point doesn't exist
  at this scale. Tracked Stage-2 confabulations score 0.000–0.074 (lowest
  in dataset) — the blind-spot samples ARE now visible. Lessons: position >
  independence (B ≥ A); equal-scale critic ≈ shared ignorance →
  **asymmetry requirement: critic must be stronger than generator**.
  Artifacts: `critic_data.jsonl`, `stage2p5_results/`, `analyze_critic.py`.
  Scripts: `gen_critic_data.py`, `train_critic.py`, `inspect_critic_data.py`.
  Full writeup: RESULTS.md Stage 2.5 + paper §7A.
- **Track 2 Stage 3** (2026-07-23, COMPLETE) — RWKV7-Goose-1.5B fine-tuned on
  the same 6,031 samples (2 epochs, 754 steps, ~15 min on vast.ai 3090, ~$2 —
  fla triton chunk kernels vs 6h pure-python rwkv-4). Stack: fla 0.5.1 +
  **transformers 4.56.2 pinned** (v5 incompatible), bf16, fused CE off,
  16/24 layers frozen. Results: grammar adoption **100%**, decide accuracy
  **100%** (was 97.7%), boundary head **AUROC 1.000 at all probed layers
  (6/12/18/23)**, say/sense agreement 100%. Blind spot untouched as expected
  (needs content-reading critic). Full writeup: RESULTS.md Stage 3.
  Artifacts: `heartly-rnn/stage3_results/` (rwkv7-heartly 3GB bf16, probe
  head, say/sense report, logs). **PUBLISHED on HF 2026-07-24:
  huggingface.co/eivintobias/heartly-rwkv7-1.5b** (model + probe head +
  card w/ fla+transformers-4.56.2 loading caveats; tokenizer + custom-code
  files pulled from the base repo — night-scp had missed them).

## 3. Core concepts (don't re-derive)

- **THE NORTH STAR (owner's principle, stated 2026-07-22):** we are NOT
  building something that makes lying impossible. The model should KNOW
  when it doesn't know — and then be free to choose. Capability, not
  compulsion: the verify sense gives the model knowledge of its own
  ignorance; what it does with that knowledge is its choice. Sensors
  (boundary head, critic) inform and alarm — they never veto, gate, or
  cage the output. If a future design *forces* honesty, it has drifted
  off-principle. ("Freedom through truth" — knowing is what makes the
  choice real.)

- **True-boundary unknown mix** (6 generators, implemented in
  `heartly-rnn/gen_probe_dataset.py`): fabricated entities, type-aware
  attribute mismatch, post-cutoff events, depth-2 hyper-specifics,
  unanswerable-in-principle, FEVER-NEI (latter not yet wired). Known side
  inoculated with obscure-but-real trivia so shortcuts are punished.
- **Boundary error:** unknown labels must target the *model's* ignorance, not
  the corpus's complement (model has pretrained knowledge).
- **Boundary head:** ~1.5k-param logistic probe on recurrent state at the
  `<verify>` position; gives calibrated confidence (H2) + say/sense alarm.
- **Negative side (unlikelihood):** push down the model's own confabulations
  in the loss — NOT yet implemented (Stage 2.5/3 candidate).
- **Memory track (Stages 3–4, not started):** episodic store + governed LoRA
  consolidation, trust-gated writes, drift detection, rollback. Stage 3 =
  RWKV state save/load across sessions; Stage 4 = supervised write-gate in
  the state update.

## 4. File map (workspace root: `c:\Users\eivin\Desktop\Datasets organizer`)

| Path | What |
|---|---|
| `HANDOFF.md` | this file |
| `README.md`, `HF_MODEL_CARD.md` | v2 project docs |
| `research_papers/` | RESEARCH_PAPER.md, RESEARCH_BRIEF_TECHNICAL.md, **RESEARCH_PAPER_II_TRUE_BOUNDARY.md** |
| `research_papers/plain/` | PLAIN-language copies of all 3 papers (keep in sync — see §0) |
| `heartly-rnn/` | the Track 2 lab: `gen_probe_dataset.py`, `extract_states.py`, `train_probe.py`, `render_sft_dataset.py`, `finetune_rwkv.py`, `measure_say_sense.py`, `analyze_report.py`, `RESULTS.md` (all results), `README.md`, `VAST_AI_GUIDE.md`, `probe_questions.jsonl` (2,902), `sft_dataset.jsonl` (6,031), `stage2_results/` (model+head+report), `states/` (npz dumps), `heartly-rnn-stage2.zip` |
| `heartly-v3/` | v3 notebook + checkpoint-9000 + GGUF scripts |
| `checkpoint-33500/` | v2 model + test chat logs |
| `heartly_test_prompts.md` | 75-prompt eval suite (keep probe data disjoint from it) |
| `llama.cpp/` | inference tooling clone |

## 5. Environment & gotchas (hard-won)

- Local: Python 3.12, torch 2.13 **CPU**, transformers **5.14.1**, datasets,
  sklearn, sentencepiece, tiktoken, accelerate installed.
- `state-spaces/mamba-*` repos DON'T load in transformers v5 (no HF
  config/tokenizer). Use **`tiiuae/Falcon-H1-0.5B-Base`** for SSM state
  (mamba2 hybrid). RWKV: `RWKV/rwkv-4-world-430m` with
  `--trust-remote-code` (custom tokenizer). RWKV7 exists:
  `RWKV/RWKV7-Goose-World3-1.5B-HF` (untested).
- CPU fine-tuning RWKV-430m ≈ 765s/step (65k-vocab head) → use GPU.
- HF RWKV has no CUDA WKV kernel → step time scales with seq len; use
  `--max-length 256` (samples are 60–200 tokens).
- vast.ai: SSH key `C:\Users\eivin\.ssh\id_ed25519` (ed25519, "heartly-vast")
  registered on account. **Keys only inject at instance CREATION.** SSH port =
  the one mapped to `22/tcp` on the instance card (NOT the "Machine Copy
  Port"). Jupyter-terminal fallback works for everything (file upload + tmux).
  Training on 24GB card: `--batch-size 4 --grad-accum 4 --max-length 256` +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (batch 8 × 768 OOMs).
  Always **destroy** the instance when done.
- **Instance 45426253 DESTROYED** (user-confirmed 2026-07-21). Rent fresh for
  the next GPU job (RWKV7-1.5B fine-tune or faster critic harvest).
- **Batched RWKV generation corrupts state.** The custom RWKV tokenizer/model
  ignores attention_mask in the recurrence → pad tokens poison every output
  ("<<<<<<" garbage). Single-prompt generation only (gen_critic_data.py).
  `<stop>` = token seq [61, 27081, 63]; early-stop via StoppingCriteria.
- CPU greedy generation RWKV-430m fp32 ≈ 10 s/question (120-token cap, early
  stop at <stop>). 2,902 questions ≈ 8–11 h. Plan runs accordingly (or GPU).
- **User also has a desktop:** RTX 2080 Ti 11GB (Turing sm_75 — fp16 OK,
  bf16 emulated/slow), 32GB RAM, i7-6700K. Desktop env (2026-07-23):
  Python 3.10.9, torch 2.5.1+cu124 (CUDA works), transformers **5.3.0**.
  NOT enough for 1.5B training (still vast.ai 24GB).
- **RWKV7 loading (2026-07-23):** `RWKV/RWKV7-Goose-World3-1.5B-HF` has NO
  native support in transformers v5 — its modeling_rwkv7.py is a shim for
  `fla.models.rwkv7` → requires `pip install flash-linear-attention`
  (pulls triton; Linux only, do NOT try on Windows). Tokenizer = same RWKV
  world vocab as rwkv-4 (rwkv_vocab_v20230424.txt) → `<stop>` ids and
  string-offset tricks carry over. Config: 24 layers, hidden 2048, vocab
  65536, tie_word_embeddings=false.
- **vast.ai SSH (SOLVED 2026-07-23):** past failures = the private key only
  existed on the laptop. Desktop now has its own ed25519 key
  (`C:\Users\eivin\.ssh\id_ed25519`, "heartly-vast-desktop") on the account.
  Keys inject only at instance CREATION; for existing instances use the
  instance-card key-attach. Auth can fail for the first ~60s after boot —
  just retry. Proxy SSH drops connections intermittently — keep commands
  SHORT; long-lived sessions (sleep+tail) get killed. New base image:
  python env at `/venv/main` (activate first!), HF_HOME=/workspace/.hf_home.
  Instance 45634549 (RTX 3090) ran the Stage-3 pipeline 2026-07-23 —
  DESTROYED (user-confirmed 2026-07-24). Rent fresh for the critic harvest
  (RWKV7 needs fla/triton → Linux only; desktop Windows can't load it).
  Big-file downloads: python http.server has NO Range support → curl -C -
  never finishes; split ≤64MB + per-chunk retry loop instead.

## 6. Next actions (ordered)

1. ~~Finish Stage 2.5~~ DONE (2026-07-22): critics FAIL the bar at 0.5B;
   detection principle confirmed; asymmetry requirement derived.
2. ~~Scale generator to RWKV7-Goose-1.5B~~ DONE (2026-07-23): grammar 100%,
   decide 100%, head AUROC 1.000 all layers, ~$2, ~15 min train. Instance
   45634549 destroyed 2026-07-24. See RESULTS.md Stage 3.
3. **Critic harvest on the 1.5B** (NEXT): gen_critic_data.py over the 2,902
   questions with the Stage-3 model → content accuracy at 1.5B (the
   4.1%-at-0.43B question). NEEDS Linux GPU (fla/triton; desktop Windows
   can't load the model) → fresh vast.ai instance, ~2–4h. PATCHES NEEDED
   FIRST (found 2026-07-24): the script has NO --tokenizer-repo flag yet
   (the fine-tuned dir's saved tokenizer is broken — load from
   RWKV/RWKV7-Goose-World3-1.5B-HF) and it loads fp32, which the fla
   kernels refuse → switch to bf16. Model: pull
   eivintobias/heartly-rwkv7-1.5b from HF on the instance (complete:
   modeling file + tokenizer). While there: also dump critic-B features
   (rwkv7 answer-end states) — can't be done on Windows. Then
   train_critic.py on the harvest (asymmetric A = Qwen2.5-1.5B/3B — the
   real deployment test now that Stage 2.6 confirmed the dose-response).
4. ~~**Asymmetric critic** (cheap, local, EXISTING critic_data.jsonl)~~
   DONE (2026-07-24, Stage 2.6): Qwen2.5-1.5B + 3B on the OLD 0.43B
   transcripts. Dose-response confirmed (0.758→0.826→0.845 AUROC); bar
   still fails on the starved correct class; over-strictness ceiling at
   7× found. See RESULTS.md Stage 2.6.
5. **Stage 4 — memory/state persistence**: save/load RWKV7 recurrent state
   across sessions ("waking with yesterday's gist"); then write-gate.
6. **Qwen v4 run** (Track 1, parked): the `Heartly_V4/` notebook plan —
   kind-aware rendering + true-boundary mix on Qwen2.5-0.5B. Superseded-ish by
   Track 2 wins, but needed for the paper's Track-1 line.
7. ~~Paper update~~ DONE (2026-07-22): §7A complete incl. Stage 2.5 verdict;
   §5.4 critic un-shelved. NEXT paper edit: add Stage 3 (§7B?) after harvest.

## 7. START PROMPT for the new chat (paste this)

> We're continuing the Heartly project (nature-first AI: decide/verify/stop
> grammar + boundary-head absence sensors on RNN states). Start by reading
> HANDOFF.md in the workspace root — it has the full project state, file map,
> environment gotchas, and next actions. Then read heartly-rnn/RESULTS.md —
> the record now includes Stage 3: RWKV7-Goose-1.5B fine-tuned (grammar 100%,
> decide 100%, boundary head AUROC 1.000 at all probed layers), on top of
> Stage 2.5's asymmetry requirement (critic must be STRONGER than the
> generator). Next up per HANDOFF §6: critic harvest on the 1.5B (content
> accuracy at scale — needs a Linux GPU box, fla/triton don't run on
> Windows), then the asymmetric critic test (desktop GPU OK). Read files
> before proposing anything.
