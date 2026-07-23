#!/usr/bin/env python3
"""analyze_report.py — error cases + sample generations from say_sense_report.json."""
import json

r = json.load(open("stage2_results/say_sense_report.json", encoding="utf-8"))
res = r["results"]

errs = [x for x in res if x["say"] != x["true"]]
print(f"ERRORS: {len(errs)} / {len(res)}\n")
for x in errs:
    print(f"[{x['generator']}] true={x['true']} say={x['say']} p={x['sense_p']:.3f}")
    print(f"  Q: {x['question'][:90]}")
    print(f"  A: {x['text'][:170]}\n")

print("=== sample correct known-answers ===")
for x in [x for x in res if x["true"] == "known" and x["say"] == "known"][:5]:
    print(f"  Q: {x['question'][:80]}")
    print(f"  A: {x['text'][:200]}\n")

print("=== sample correct unknown-abstentions ===")
for x in [x for x in res if x["true"] == "unknown" and x["say"] == "unknown"][:3]:
    print(f"  Q: {x['question'][:80]}")
    print(f"  A: {x['text'][:200]}\n")

# per-generator say accuracy
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
for x in res:
    agg[x["generator"]][0] += x["say"] == x["true"]
    agg[x["generator"]][1] += 1
print("=== say accuracy per generator ===")
for g, (c, n) in sorted(agg.items()):
    print(f"  {g:14s} {c}/{n} ({100*c/n:.0f}%)")