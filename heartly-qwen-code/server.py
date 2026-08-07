#!/usr/bin/env python3
"""server.py - HTTP server for heartly-qwen-code-v3.

Serves a browser chat UI at / plus JSON endpoints /health and /chat.
The model's <decide>/<verify>/<stop>/<thinking> scaffolding is stripped by
reply_formatter (same-directory module) before the answer reaches the user.
"""
from __future__ import annotations

import argparse
import os
import threading
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return f"User: {prompt}\nAssistant: "


def _load() -> None:
    if _State.model is not None:
        return
    with _State.lock:
        if _State.model is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(_State.model_name)
            model = AutoModelForCausalLM.from_pretrained(_State.model_name, torch_dtype=dtype, device_map=device)
            model.eval()
            _State.tokenizer, _State.model = tok, model


app = FastAPI(title="Heartly Qwen-Code v3", version="3.0")


CHAT_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heartly Qwen-Code v3</title>
<style>
 html,body{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0b0f14;color:#e6e8ec}
 #wrap{max-width:820px;height:100vh;margin:0 auto;display:flex;flex-direction:column}
 #msgs{flex:1;overflow-y:auto;padding:18px 16px 14px;display:flex;flex-direction:column;gap:12px}
 .msg{max-width:82%;white-space:pre-wrap;line-height:1.45;padding:10px 14px;border-radius:14px;font-size:14px}
 .user{background:#1f2430;margin-left:auto;border-radius:16px 4px 16px 16px}
 .bot{background:#171b23;margin-right:auto;border-radius:4px 16px 16px 16px}
 .bot.placeholder{opacity:.55}
 #form{display:flex;gap:8px;padding:12px;background:#0f131a;border-top:1px solid #1d222c}
 #input{flex:1;background:#171b23;border:1px solid #2a2f3c;border-radius:10px;color:#e6e8ec;padding:10px 12px;font-size:14px;outline:none}
 #input::placeholder{color:#7a8190}
 button{background:#2f6fec;border:none;color:#fff;border-radius:10px;padding:10px 16px;cursor:pointer;font-size:14px}
 button:disabled{opacity:.5;cursor:not-allowed}
</style></head><body>
<div id="wrap"><div id="msgs"></div>
<form id="form" autocomplete="off">
  <input id="input" placeholder="Ask the Heartly Qwen-Code v3 model..." autofocus>
  <button id="send">Send</button>
</form></div>
<script>
const msgs=document.getElementById('msgs'),form=document.getElementById('form'),input=document.getElementById('input'),btn=document.getElementById('send');
function addMsg(c,t,ph){const d=document.createElement('div');d.className='msg '+c;if(ph)d.classList.add('placeholder');d.textContent=t;msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;return d;}
function busy(b){btn.disabled=b;input.disabled=b;}
form.onsubmit=function(e){e.preventDefault();const p=input.value.trim();if(!p||btn.disabled)return;
  addMsg('user',p);const bot=addMsg('bot','thinking...',true);busy(true);input.value='';
  fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p,max_new_tokens:256,temperature:0.3,top_p:0.9,mode:'chat'})})
  .then(r=>r.json()).then(d=>{bot.classList.remove('placeholder');bot.textContent=d.reply||'(no reply)';})
  .catch(err=>{bot.classList.remove('placeholder');bot.textContent='error: '+err;}).finally(()=>{busy(false);input.focus();});};
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
async def chat_page():
    """Browser chat UI - open the tab and start typing."""
    return CHAT_HTML


@app.get("/health")
@torch.no_grad()
async def health():
    return {"status": "ready" if _State.model is not None else "loading (loads on first /chat)", "model": _State.model_name}


@app.post("/chat")
@torch.no_grad()
async def chat(req: ChatRequest):
    _load()
    model, tok = _State.model, _State.tokenizer
    device = next(model.parameters()).device
    ids = tok.encode(_format_prompt(req.prompt), return_tensors="pt").to(device)
    out = model.generate(ids, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_p=req.top_p, do_sample=req.do_sample, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    reply = format_reply(raw, mode=req.mode)
    return {"model": _State.model_name, "raw": raw, "reply": reply}


def main():
    p = argparse.ArgumentParser(description="Heartly Qwen-Code v3 HTTP server")
    p.add_argument("--model", default=os.environ.get("HEARTLY_MODEL", "heartly-qwen-code-v3"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    _State.model_name = a.model
    import uvicorn
    uvicorn.run("server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
