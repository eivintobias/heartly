"""Capture the exact source text triton receives for the failing fla kernel."""
import inspect
import re
import textwrap

import triton

orig_init = triton.runtime.jit.JITFunction.__init__
seen = {}


def patched(self, fn, *a, **kw):
    if not seen:
        try:
            raw = inspect.getsource(fn)
        except Exception as e:
            raw = f"<getsource error: {type(e).__name__}: {e}>"
        src = textwrap.dedent(raw)
        if re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE) is None:
            seen["name"] = getattr(fn, "__name__", "?")
            seen["file"] = getattr(getattr(fn, "__code__", None), "co_filename", "?")
            seen["lineno"] = getattr(getattr(fn, "__code__", None), "co_firstlineno", "?")
            seen["raw"] = raw[:800]
            seen["type"] = str(type(fn))
            seen["wrapped"] = hasattr(fn, "__wrapped__")
    return orig_init(self, fn, *a, **kw)


triton.runtime.jit.JITFunction.__init__ = patched

try:
    from fla.models.rwkv7 import RWKV7ForCausalLM  # noqa: F401
    print("import OK")
except Exception as e:
    print("import failed:", type(e).__name__, e)

if seen:
    print("=== OFFENDING ===")
    for k in ("name", "type", "wrapped", "file", "lineno"):
        print(f"{k}: {seen[k]}")
    print("---- raw getsource ----")
    print(seen["raw"])
else:
    print("nothing captured")
