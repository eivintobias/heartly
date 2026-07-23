#!/usr/bin/env python3
"""inspect_critic_data.py — quick look at critic_data.jsonl rows."""
import argparse
import json
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="critic_data_test.jsonl")
ap.add_argument("--limit", type=int, default=25)
ap.add_argument("--only", default=None, help="row_class filter")
ap.add_argument("--summary-only", action="store_true")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
c = Counter(r["row_class"] for r in rows)
print(f"{len(rows)} rows: {dict(c)}")

if not args.summary_only:
    shown = 0
    for r in rows:
        if args.only and r["row_class"] != args.only:
            continue
        print(f"\nid={r['id']} [{r['generator']}] true={r['true']} "
              f"class={r['row_class']} tracked={r['tracked']}")
        print(f"  Q: {r['question'][:90]}")
        print(f"  A: {r['answer'][:130]}")
        shown += 1
        if shown >= args.limit:
            break