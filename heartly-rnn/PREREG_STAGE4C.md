# Pre-registration — Stage 4c: memory-aware SFT retrain

Written 2026-07-25, BEFORE any run. Pass criteria and protocol frozen here.

## Question(s)

Stage 4b left two diagnosed failures, both with clear fixes:

- **A (context injection fails because the SFT mix has no "answer from provided
  context" class):** the model was never trained to treat injected text as
  knowledge — it reads context as scenery. Adding a context-known rendered
  class (context paragraph + question → speak/known, grounded answer) should
  flip Part B from ≤2/5 to ≥4/5. This is the paper's §6.2 retrieval-store
  path, now trainable.

- **B (grammar degrades in long multi-turn continuations):** the SFT data is
  single-turn, so multi-turn conversation is out-of-distribution. Any deployed
  memory path needs multi-turn training data. Adding multi-turn samples should
  fix the observed `<stop>unknown</stop>` loops and fused tokens.

Both fixes are in the SFT mix — no architecture changes, no new scripts on the
GPU side. Same fine-tune recipe as Stage 3 (RWKV7-1.5B, 2 epochs, ~15 min/$2).
After retraining, re-run the Stage 4b test battery (write-gate + retrieval) AND
re-verify that decide/grammar didn't regress (measure_say_sense).

## Protocol (frozen)

### Step 1 — new SFT dataset (local, CPU)

`render_sft_dataset_v2.py` extends the existing `render_sft_dataset.py` with
two new classes:

**(i) Context-known rendered class** (~1,000 samples):
- Source: SQuAD train split (context-dependent QA — the model must read the
  provided context to answer).
- Rendering: the context paragraph is placed in the instruction field before
  the question; the output is speak/known with the answer grounded in the
  context. This teaches the model: "when context is provided, treat it as
  knowledge and answer from it."
- Format: `instruction = "Context: {paragraph}\n{question}"`,
  `output = <arg_key> I can find the answer in the provided context. I will speak. </think><decide>speak</decide><verify>known</verify> {answer} <stop>`
- Reasoning templates: context-aware variants ("I can find the answer in the
  provided context", "The context tells me this", "Based on the provided
  information, I know this").
- Answer style: same kind-aware rendering as the existing known class (no
  templates, provenance ~10%).
- This class is the direct fix for Stage 4b Part B: the model learns that
  text in the "Context:" field IS knowledge.

**(ii) Multi-turn conversation class** (~500 samples):
- Source: synthesized from the existing known/unknown samples by chaining
  2–4 turns into one training sample.
- Rendering: each turn follows the grammar; the full multi-turn transcript
  is one training row. This teaches the model to maintain grammar across
  turn boundaries.
- Known-turn chains: 2–4 factual questions answered in sequence.
- Mixed chains: known + unknown turns interleaved (the model must switch
  between speak/known and speak/unknown mid-conversation).
- Format: `instruction = "User: {q1}\nAssistant: {a1}\nUser: {q2}\nAssistant: "`,
  `output = <arg_key> {reasoning} </think><decide>speak</decide><verify>{known|unknown}</verify> {answer} <stop>`
  (only the LAST assistant turn is in the output field; earlier turns are
  part of the instruction/context, same collation as single-turn — the
  loss only trains on the final response).

The existing classes (known question-only, unknown, silence) remain at their
Stage-2 sizes (4,000 + 2,400 + 250 = 6,650). New classes add ~1,500 rows.
Total dataset: ~8,150 samples (was 6,031). Same seed (123), same shuffle.

### Step 2 — fine-tune (vast.ai 3090, same recipe as Stage 3)

```
python finetune_rwkv.py \
  --repo RWKV/RWKV7-Goose-World3-1.5B-HF \
  --data sft_dataset_v2.jsonl \
  --epochs 2 \
  --batch-size 4 --grad-accum 4 \
  --max-length 256 \
  --freeze-layers 16 \
  --dtype bf16 \
  --out rwkv7-heartly-v2
```

Same stack: transformers 4.56.2 + fla 0.5.1 + triton, bf16, fused CE off.
Expected: ~15 min, ~$2. The longer dataset (~8,150 vs 6,031) means ~1,020
steps (was 754); still well within a single short session.

### Step 3 — re-verify decide/grammar (on the instance)

```
python measure_say_sense.py \
  --model rwkv7-heartly-v2 \
  --eval-limit 300 \
  --head-out probe_head_v2.pkl \
  --report say_sense_report_v2.json
```

Must confirm: grammar adoption ≥99%, decide accuracy ≥97% (Stage 3 was 100%;
a small regression from the new classes is acceptable but must be quantified).

### Step 4 — re-run Stage 4b test battery (on the instance)

Adapted scripts (same logic, pointing at the new model dir):

```
python stage4c_write_gate.py   # identical to stage4b_write_gate.py, MODEL_REPO=rwkv7-heartly-v2
python stage4c_retrieval.py    # identical to stage4b_retrieval.py, MODEL_REPO=rwkv7-heartly-v2
```

Same 5 facts, same 6 write formats, same 3 injection formats, same hit rule
(gold substring). Results directly comparable to Stage 4b.

## Pass bars (frozen)

- **Grammar/decide regression check:** grammar adoption ≥99%, decide accuracy
  ≥97%. If below, the new classes hurt the base capability — record and
  diagnose before proceeding.
- **Part A (write-gate):** W6 (combined multi-format) ≥4/5 again — must not
  regress from Stage 4b's 4/5. Any single format improving vs Stage 4b is a
  bonus (the context-known class might help single formats too, since the
  model now has a broader "known" concept).
- **Part B (retrieval/injection):** any single injection format ≥4/5 → the
  §6.2 retrieval-store path WORKS end-to-end with the context-known training
  class. This is the headline prediction. All formats ≤3/5 → the missing-
  class diagnosis was wrong or insufficient; record as a negative.
- **Multi-turn grammar (qualitative):** no `<stop>unknown</stop>` loops or
  fused tokens in the write-gate multi-format output (W6 is 905 tokens,
  effectively multi-turn). Not a hard bar, but recorded.

## Decision rule (fixed)

- B passes → the retrieval store is the episodic memory channel (§6.2
  validated); both memory paths now work (state-writing via W6, context
  injection via retrieval). Next: integrated memory demo (write-gate +
  retrieval store in one session), paper §7A update.
- B fails, A passes → state-writing is the only working channel; retrieval
  needs more than a training class (perhaps context format must match
  training exactly, or the disposition needs explicit "memory" framing).
  Record as a partial win (A still works) and diagnose.
- Both fail → the retrain broke something or the diagnosis was wrong; full
  diagnostic needed before any next step.
- Grammar/decide regresses → the new classes diluted the signal; reduce
  their proportion and retrain (Stage 4c-2, same recipe, smaller new-class
  count).

## Budget & artifacts (planned)

One vast.ai 3090 session, ~45–60 min GPU (~$1–2). Scripts:
`render_sft_dataset_v2.py` (local), `stage4c_write_gate.py`,
`stage4c_retrieval.py`, `run_stage4c.sh`. Results:
`stage4c_results/` (write_gate_report.json, retrieval_report.json,
say_sense_report_v2.json, logs). New model: `rwkv7-heartly-v2/`.
Writeup: RESULTS.md Stage 4c + paper §7A + plain copy (per HANDOFF §0).
