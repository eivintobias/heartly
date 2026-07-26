"""Faithful repro of fla's decorator pattern -> does inspect.getsource truncate?

Run under multiple interpreters to find one where getsource returns the full
function (triton requires the `def` line to be present).
"""
import inspect
import re
import sys


def cdiv(a, b):
    return (a + b - 1) // b


def heuristics(d):
    def deco(fn):
        return fn
    return deco


def autotune(configs=None, key=None, **kw):
    def deco(fn):
        return fn
    return deco


def jit(fn):
    return fn


# --- exact decorator shape used by fla/ops/simple_gla/parallel.py ---
@heuristics({
    'NV': lambda args: cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@autotune(
    configs=[(w, s) for w in [2, 4, 8, 16] for s in [2, 3, 4]],
    key=["BT", "BS", "BK", "BV", "USE_G"],
)
@jit
def parallel_kernel(q, k, v, g, o):
    return q


src = inspect.getsource(parallel_kernel)
ok = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE) is not None
print(f"python {sys.version.split()[0]}: getsource lines={len(src.splitlines())} contains_def={ok}")
if not ok:
    print("TRUNCATED ->", repr(src[-80:]))
