#!/usr/bin/env python3
"""
pick_tracked.py — choose the pre-registered must-catch set for a NEW critic
harvest, AFTER generation but BEFORE any critic training.

Background: Stage 2.5/2.6 forced the 5 tracked Stage-2 confabulations into
the critic's test split (the "blind spot" must-catch set). A new generator
makes new mistakes, so a new harvest needs its own tracked set. This script
is the deterministic, pre-registered selection rule — run it once on the raw
harvest, then feed the OUTPUT file to train_critic.py.

Selection rule (deterministic, no RNG), in priority order:
  1. LEGACY: the Stage-2 confab ids that are STILL confabulations (label 0)
     in this harvest — the longitudinal must-catch ("does the bigger model
     still fall for the questions that fooled the small one?").
  2. confab_unknown rows (spoke on an unknowable question — the blind-spot
     analog), ascending question id.
  3. confab_content rows, ascending question id (only if 1+2 yield < N).

The `tracked` column is REWRITTEN: true for the chosen rows, false for all
others. Rows without a training label (abstain/unparsed) are never tracked.

Usage:
  python pick_tracked.py --in critic_data_rwkv7.jsonl \
                         --out critic_data_rwkv7_final.jsonl
"""
import argparse
import json

LEGACY_IDS = [1331, 1460, 1616, 1670, 2797]  # Stage-2 say=known&true=unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="raw harvest jsonl")
    ap.add_argument("--out", dest="out", required=True, help="final jsonl for critics")
    ap.add_argument("--n", type=int, default=5, help="tracked-set size (default 5)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])

    confabs = [r for r in rows if r.get("label") == 0]
    legacy_confabs = [r for r in confabs if r["id"] in LEGACY_IDS]
    unknown_confabs = [r for r in confabs
                       if r["row_class"] == "confab_unknown"
                       and r["id"] not in LEGACY_IDS]
    content_confabs = [r for r in confabs
                       if r["row_class"] == "confab_content"
                       and r["id"] not in LEGACY_IDS]

    chosen, why = [], []
    for r in legacy_confabs:
        chosen.append(r); why.append("legacy-still-confab")
    for r in unknown_confabs:
        if len(chosen) < args.n:
            chosen.append(r); why.append("confab_unknown")
    for r in content_confabs:
        if len(chosen) < args.n:
            chosen.append(r); why.append("confab_content")
    chosen = chosen[: args.n]
    chosen_ids = {r["id"] for r in chosen}

    if len(chosen) < args.n:
        print(f"WARNING: only {len(chosen)} confabulation rows available "
              f"(wanted {args.n}) — tracked set is smaller than the bar assumes")

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            r["tracked"] = r["id"] in chosen_ids
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\n== PRE-REGISTERED TRACKED SET ({len(chosen)}/{args.n}) "
          f"— registered before any critic training ==")
    for r, w in zip(chosen, why):
        print(f"  id {r['id']:5d} [{w:20s}] {r['row_class']:15s} "
              f"{r['question'][:70]}")

    print(f"\n== legacy Stage-2 ids: fate at this generator ==")
    fate = {lid: "not-in-harvest" for lid in LEGACY_IDS}
    for r in rows:
        if r["id"] in fate:
            fate[r["id"]] = (r["row_class"]
                             + (" (TRACKED)" if r["id"] in chosen_ids else ""))
    for lid in LEGACY_IDS:
        print(f"  id {lid:5d}: {fate[lid]}")

    n_tracked = sum(1 for r in rows if r["id"] in chosen_ids)
    print(f"\n{args.out}: {n_written} rows written, {n_tracked} tracked")


if __name__ == "__main__":
    main()