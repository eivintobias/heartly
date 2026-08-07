#!/usr/bin/env python3
"""
chat_clean.py — Chat with the Heartly Qwen-Code GGUF model with grammar stripped.

Loads the GGUF with llama-cpp-python, generates responses, then strips the
Heartly grammar (think block, decide/verify tags, stop token) before showing
the user only the clean answer.

Usage:
    python chat_clean.py --model heartly-qwen-code-v2-proper.gguf
    python chat_clean.py --model heartly-qwen-code-v2-proper.gguf --gpu 0
"""
import argparse
import re
import sys


def strip_grammar(raw_output: str) -> str:
    """Strip Heartly grammar from raw model output, return clean answer.
    
    Handles:
    -  thinking ...  response (think block)
    - <decide>speak</decide> / <decide>stop</decide>
    - <verify>known</verify> / <verify>unknown</verify>
    - <stop> / <stop (truncated)
    - Multiple turns (separated by newlines with grammar)
    """
    # If the model decided to stop, return "..."
    if "<decide>stop</decide>" in raw_output:
        return "..."
    
    # Extract answer zone: everything after <verify>...</verify>
    # The answer is between <verify>known</verify> or <verify>unknown</verify> and <stop>
    answers = []
    
    # Split by potential multi-turn boundaries (each turn has its own grammar)
    # Look for all answer zones
    parts = re.split(r'(?:<decide>speak</decide><verify>\w+</verify>)', raw_output)
    
    for part in parts[1:]:  # skip the first (everything before the first verify)
        # Cut at <stop> or <stop (truncated)
        stop_match = re.search(r'<stop', part)
        if stop_match:
            answer = part[:stop_match.start()].strip()
        else:
            answer = part.strip()
        
        if answer:
            answers.append(answer)
    
    if not answers:
        # Fallback: try to find anything after the think block
        # Remove think block
        cleaned = re.sub(r'\s*thinking\s+.*?\s+response\s*', '', raw_output, flags=re.DOTALL)
        # Remove all grammar tags
        cleaned = re.sub(r'<decide>\w+</decide>', '', cleaned)
        cleaned = re.sub(r'<verify>\w+</verify>', '', cleaned)
        cleaned = re.sub(r'<stop>?\s*$', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            answers.append(cleaned)
    
    return '\n'.join(answers) if answers else raw_output.strip()


def main():
    ap = argparse.ArgumentParser(description="Chat with Heartly GGUF model (grammar stripped)")
    ap.add_argument("--model", required=True, help="Path to the GGUF model file")
    ap.add_argument("--gpu", type=int, default=-1, help="GPU layers (-1 = all, 0 = CPU only)")
    ap.add_argument("--ctx", type=int, default=2048, help="Context length")
    ap.add_argument("--temp", type=float, default=0.7, help="Temperature")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    args = ap.parse_args()
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Please install llama-cpp-python: pip install llama-cpp-python")
        sys.exit(1)
    
    print(f"Loading model: {args.model}")
    print(f"GPU layers: {args.gpu} (-1 = all GPU, 0 = CPU)")
    print()
    
    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx,
        n_gpu_layers=args.gpu,
        verbose=False,
    )
    
    print("=" * 60)
    print("Heartly Chat (clean output — grammar tokens stripped)")
    print("Type 'quit' or 'exit' to leave.")
    print("=" * 60)
    print()
    
    conversation = []
    
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        
        if not user_input:
            continue
        
        # Build the prompt using the chat template
        conversation.append({"role": "user", "content": user_input})
        
        # Use the model's chat template
        prompt = ""
        for msg in conversation:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\nAssistant: "
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']}\n"
        
        # Generate
        output = llm(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            stop=["<stop>", "User:"],
            echo=False,
        )
        
        raw_text = output["choices"][0]["text"].strip()
        
        # Strip grammar
        clean_text = strip_grammar(raw_text)
        
        # Show clean output
        print(f"Heartly> {clean_text}")
        print()
        
        # Store the clean response in conversation history
        conversation.append({"role": "assistant", "content": clean_text})


if __name__ == "__main__":
    main()