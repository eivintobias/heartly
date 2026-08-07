#!/usr/bin/env python3
"""server.py - HTTP server for heartly-qwen-code-v3.

Loads the v3 model (Qwen2.5-Coder-1.5B fine-tuned with the Heartly grammar)
and exposes a /chat endpoint that returns the user-visible answer with the
`<decide>/<verify>/<stop>/<thinking>` scaffolding stripped by reply_formatter
(same-directory module).

Usage:
    python server.py --model heartly-qwen-code-v3 --port 8000
    # or:
    uvicorn server:app --host 0.0.0.0 --port 8000   (set HEARTLY_MODEL to override)

Chat:
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" ^
      -d '{"prompt":"Write a function that reverses a string"}'
"""
from __future__ import annotations

import argparse
import os
import threading
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same-directory module: cleans the Heartly grammar from model output.
from reply_formatter import format_reply


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    mode: str = "chat"  # chat | debug | raw


class _State:
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    model_name: str = os.environ.get("HEARTLY_MODEL", "heartly-qwen-code-v3")
    lock: threading.Lock = threading.Lock()


def _format_prompt(prompt: str) -> str:
    # Same prompt shape used by finetune_qwen.py / test_v3_hf.py.
    return f"User: {prompt}\nAssistant: "


def _load() -> None:
    """Lazy-load the model once; first /chat request blocks briefly."""
    if _State.model is not None:
        return
    with _State.lock:
        if _State.model is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(_State.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                _State.model_name, torch_dtype=dtype, device_map=device
            )
            model.eval()
            _State.tokenizer, _State.model = tok, model


app = FastAPI(title="Heartly Qwen-Code v3", version="3.0")


@app.get("/health")
@torch.no_grad()
async def health():
    return {
        "status": "ready" if _State.model is not None else "loading (loads on first /chat)",
        "model": _State.model_name,
    }


@app.post("/chat")
@torch.no_grad()
async def chat(req: ChatRequest):
    _load()  # lazy: blocks only on the very first request
    model, tok = _State.model, _State.tokenizer
    device = next(model.parameters()).device

    ids = tok.encode(_format_prompt(req.prompt), return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
        pad_token_id=tok.eos_token_id,
    )

    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    reply = format_reply(raw, mode=req.mode)
    return {"model": _State.model_name, "raw": raw, "reply": reply}


def main():
    p = argparse.ArgumentParser(description="Heartly Qwen-Code v3 HTTP server")
    p.add_argument("--model", default=os.environ.get("HEARTLY_MODEL", "heartly-qwen-code-v3"),
                   help="path or HF repo id of the v3 model")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    _State.model_name = a.model
    import uvicorn
    uvicorn.run("server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
