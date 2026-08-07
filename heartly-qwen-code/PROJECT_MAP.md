# Project map — heartly-qwen-code v3

```
heartly-qwen-code/
+- heartly-qwen-code-v3/          # fine-tuned model dir (config, tokenizer, weights — NOT in git)
|   +- config.json
|   +- generation_config.json
|   +- chat_template.jinja        # standard Qwen template
|   +- tokenizer.json / tokenizer_config.json
|   +- model.safetensors          # ~3 GB, git-ignored (served from HF Hub)
+- server.py                      # FastAPI serving: GET / (browser UI), /health, /chat
+- reply_formatter.py             # display layer: strip grammar + unescape code newlines
+- chat_smoke.py                  # offline inference (same prompt + formatter, no HTTP)
+- render_code_sft_v3.py          # Stage-5 SFT dataset renderer (grammar + natural phrasing)
+- finetune_qwen.py               # fine-tuning entry point
+- sft_dataset_code_v3.jsonl      # 5,200 conversational Heartly samples (instruction/output)
+- test_reply_formatter.py        # unit tests for the formatter
+- test_v3_hf.py                  # HF load / integration smoke
+- requirements.txt               # torch, transformers, fastapi, uvicorn, huggingface_hub
+- README.md                      # user-facing docs
+- HANDOFF.md                     # dev handoff
+- RESULTS.md                     # results log (Stages 1-5)
+- HF_MODEL_CARD_v3.md            # model card (upload to HF as README.md)
+- run_code_stage1.sh             # training entrypoint (vast.ai / 3090)
+- run_server.bat                 # Windows server launcher
```

## Data + control flow
```
prompt
  -> server.py: prompt wrapped as  User: <prompt> [newline] Assistant:
  -> tok.encode -> model.generate (lazy-loaded on first POST /chat)
  -> raw Heartly grammar output
  -> reply_formatter.format_reply(raw, mode='chat')
       strips thinking / <decide>/<verify>/<stop> + unescapes code newlines
  -> /chat JSON {reply, raw}   |   GET / renders the browser chat UI
```
The model emits the Heartly grammar (`thinking ...` `response<decide>speak</decide>`
`<verify>known</verify>` answer `<stop>`) as ORDINARY text tokens — the tags are not
tokenizer special tokens — so output is decoded with `skip_special_tokens=False` and
the formatter canonicalizes + strips. Only the final answer reaches the user.

## Quick checks
- `python -c "from reply_formatter import format_reply; print(format_reply('thinking x response<decide>speak</decide><verify>known</verify> hi <stop>'))"` -> `hi`
- `python chat_smoke.py "reverse a list"` -> code reply
- `python server.py --port 8000` then open `http://127.0.0.1:8000/`
