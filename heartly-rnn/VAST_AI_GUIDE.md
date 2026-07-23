# Stage 2 on vast.ai — step-by-step

Estimated total cost: **~$1–3** (~1 h on an RTX 3090/4090 + transfers).
Estimated total time: ~45–75 min of compute.

The bundle you need is `heartly-rnn-stage2.zip` (in this folder). It contains
all scripts, `sft_dataset.jsonl` (6,031 rendered samples), `probe_questions.jsonl`
(2,902 eval questions), remote requirements, and `run_stage2.sh`.

## 1. Account

1. Sign up at https://vast.ai → add ~$5 credit (Billing).

## 2. Rent the instance

1. **Create → Search Offers**. Filters that work well:
   - GPU: **RTX 3090** or **RTX 4090** (24 GB is plenty for a 0.43B model) —
     or any A100 if the price is close.
   - Type: **On-Demand** (no preemption surprises on a 1 h job).
   - Sort by **$/hr ascending**. Anything ≤ $0.50/hr is a good deal.
2. Template: a **PyTorch (CUDA 12)** image. Disk: 20 GB is enough.
3. Launch and wait for status **Running**.

## 3. Upload the bundle

From your local PowerShell (use the IP/port shown on the instance card):

```powershell
scp -P <PORT> "heartly-rnn\heartly-rnn-stage2.zip" root@<IP>:/workspace/
```

(Windows ships `scp` by default. If it asks about host keys: yes.)

## 4. Run (inside tmux — survives SSH drops)

On the instance (SSH or the web terminal from the instance card):

```bash
cd /workspace
apt-get update -qq && apt-get install -y -qq unzip tmux
unzip -o heartly-rnn-stage2.zip -d stage2
cd stage2
tmux new -s s2
bash run_stage2.sh
# detach anytime with: Ctrl+B then D  →  reattach: tmux attach -t s2
```

What happens:
1. deps install (~2 min)
2. fine-tune — 754 steps, ~30–60 min on a 3090/4090. Watch the loss fall
   from ~4.5 toward <1.
3. boundary head training + say/sense measurement (~10–15 min)
4. final `==== SAY/SENSE SUMMARY ====` printed to the console

## 5. Download the results

From local PowerShell:

```powershell
scp -P <PORT> -r root@<IP>:/workspace/stage2/rwkv-heartly "heartly-rnn\"
scp -P <PORT> root@<IP>:/workspace/stage2/say_sense_report.json "heartly-rnn\"
scp -P <PORT> root@<IP>:/workspace/stage2/probe_head.pkl "heartly-rnn\"
```

## 6. DESTROY the instance

Billing stops only when the instance is **destroyed** (stopped instances still
charge for storage). Instance card → Destroy.

## Troubleshooting

- **HF rate limits / gated warnings**: everything used here is public; the
  warning about `HF_TOKEN` is cosmetic.
- **transformers version too old for rwkv**: `pip install -U transformers`
  (run_stage2.sh already pins >= 4.44).
- **Out of memory**: shouldn't happen at 0.43B/fp32 on 24 GB; if it does,
  rerun fine-tune with `--batch-size 4 --grad-accum 4`.
- **The model folder downloads**: first run pulls ~1 GB of model weights +
  ~250 MB of datasets from HF — a few minutes on datacenter bandwidth.