# Heartly

![openalpha_evolve_workflow](https://github.com/user-attachments/assets/9d4709ad-0072-44ae-bbb5-7eea1c5fa08c)

Heartly is an open-source Python framework that combines the evolutionary code-generation architecture of [OpenAlpha_Evolve](https://github.com/shyamsaktawat/OpenAlpha_Evolve) with the Heartly hallucination-reduction model. It uses a locally fine-tuned **Heartly-Qwen-Coder** model (based on Qwen2.5-Coder-1.5B) as the code-generation engine, replacing external LLM API calls with a model that can **decide whether to speak**, **verify what it knows**, and **admit ignorance** when appropriate.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

---

## 🙏 Credits & Attribution

This project is a **modified fork** of [**OpenAlpha_Evolve**](https://github.com/shyamsaktawat/OpenAlpha_Evolve) by **Shyam Saktawat** (MIT License, © 2024). The original evolutionary code-generation framework — including the agent-based architecture, prompt design, evaluation pipeline, and diff-based mutation system — is his work. We are deeply grateful for his open-source contribution.

OpenAlpha_Evolve itself is inspired by the pioneering research of **Google DeepMind** on **AlphaEvolve** and related work in LLM-driven code generation and automated discovery.

**Heartly modifications** (© 2026 Eivin Tobias):
- Added `code_generator/local_model.py` — loads and runs the Heartly-Qwen-Coder model locally
- Added `code_generator/heartly_parser.py` — parses Heartly grammar output (`<decide>`/`<verify>`/`<stop>`)
- Added `code_generator/boundary_head.py` — optional quality-gate probe on hidden states
- Modified `code_generator/agent.py` — routes to the local Heartly model when `USE_LOCAL_MODEL=True`
- Added local model configuration to `config/settings.py` and `.env.example`

The Heartly model itself is trained using the [Heartly research project](https://github.com/eivintobias/heartly) — a "nature-first" approach to hallucination reduction.

---

## Table of Contents
- [✨ The Vision](#-the-vision)
- [🧠 How It Works](#-how-it-works)
- [🚀 Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [🏁 Getting Started](#-getting-started)
- [🔮 Future Evolution](#-future-evolution)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Credits & Attribution](#-credits--attribution)

---

## ✨ The Vision

Heartly combines two powerful ideas:

1. **Evolutionary code generation** (from OpenAlpha_Evolve): an intelligent system that iteratively writes, tests, and improves code using LLMs, guided by the principles of evolution.

2. **Hallucination reduction** (from the Heartly research project): a model that learns to **decide** whether it can answer, **verify** whether it actually knows, and **admit ignorance** when it doesn't — compiled into the training data itself.

The result is a code-generation agent that knows when it doesn't know.

---

## 🧠 How It Works

Heartly employs a modular, agent-based architecture to orchestrate an evolutionary process:

1. **Task Definition**: You define the algorithmic "quest" — the problem to solve, with input/output examples.
2. **Prompt Engineering** (`PromptDesignerAgent`): Crafts intelligent prompts for initial generation, mutation, and bug-fixing.
3. **Code Generation** (`CodeGeneratorAgent`): Powered by the local Heartly-Qwen-Coder model (or external LLMs via LiteLLM). The model outputs in Heartly grammar:
   ```
    thinking {reasoning}  response<decide>speak|stop</decide><verify>known|unknown</verify> {answer} <stop>
   ```
   The `heartly_parser` extracts code only when the model decides to speak and verifies it knows the answer.
4. **Evaluation** (`EvaluatorAgent`): Generated code is syntax-checked and run against test cases in Docker containers.
5. **Database** (`DatabaseAgent`): All programs and their fitness scores are stored.
6. **Selection** (`SelectionControllerAgent`): Selects parents and survivors for the next generation.
7. **Iteration**: The cycle repeats for a defined number of generations.
8. **Orchestration** (`TaskManagerAgent`): Coordinates all agents and manages the evolutionary loop.

### The Heartly Model

The Heartly-Qwen-Coder model is a fine-tuned Qwen2.5-Coder-1.5B that uses a special output grammar:

- **`<decide>speak</decide>`** — the model chooses to attempt an answer
- **`<decide>stop</decide>`** — the model chooses to abstain
- **`<verify>known</verify>`** — the model confirms it knows the answer
- **`<verify>unknown</verify>`** — the model admits it doesn't know

An optional **boundary head** (logistic probe on hidden states) can serve as a quality gate, detecting when the model claims knowledge it doesn't actually have.

---

## 🚀 Key Features

- **Local Heartly Model Support**: Run the Heartly-Qwen-Coder model locally instead of paying for external API calls. The model can abstain from answering when it doesn't know, reducing hallucination in generated code.
- **External LLM Support**: Still supports LiteLLM with multiple providers (OpenAI, Anthropic, Google, etc.) when `USE_LOCAL_MODEL=False`.
- **Evolutionary Algorithm Core**: Iterative improvement through selection, LLM-driven mutation/bug-fixing using diffs, and survival.
- **Modular Agent Architecture**: Easily extend or replace individual components.
- **Automated Program Evaluation**: Syntax checking and functional testing in Docker containers.
- **Boundary Head Quality Gate**: Optional probe that reads the model's hidden states to detect confident confabulation.
- **Configuration Management**: All parameters via `config/settings.py` and `.env`.
- **Detailed Logging**: Comprehensive logs for each step of the evolutionary process.
- **Diff-based Mutations**: Targeted code modifications via diffs.
- **Open Source & Extensible**: Built with Python, designed for experimentation.

---

## 📂 Project Structure

```text
./
├── code_generator/          # Code generation agent with Heartly model support
│   ├── agent.py             # Main agent — routes to local model or external API
│   ├── local_model.py       # Heartly-Qwen-Coder model loading and inference
│   ├── heartly_parser.py    # Parses Heartly grammar output
│   ├── boundary_head.py     # Optional quality-gate probe
│   └── __init__.py
├── database_agent/          # Storage and retrieval of programs
├── evaluator_agent/         # Code evaluation (syntax, execution, fitness)
├── prompt_designer/         # Prompt crafting for LLM
├── selection_controller/    # Selection strategy for parents and survivors
├── task_manager/            # Orchestrates the evolutionary loop
├── config/                  # Configuration (settings.py)
├── core/                    # Core data structures and interfaces
├── tests/                   # Unit and integration tests
├── examples/                # Example task definitions (YAML)
├── main.py                  # Entry point
├── app.py                   # Gradio web interface
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── LICENSE.md               # MIT License
└── README.md                # This file
```

---

## 🏁 Getting Started

### Prerequisites

- Python 3.10+
- `pip` for package management
- `git` for cloning
- **Docker**: For sandboxed code evaluation
- **GPU (optional)**: For running the local Heartly model (CPU works but is slower)

### Clone

```bash
git clone https://github.com/eivintobias/heartly.git
cd heartly/HeartlyOpenAlpha_Evolve
```

### Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Copy the example and fill in your values:

```bash
cp .env.example .env
```

#### Using the Local Heartly Model (Default)

By default, HeartlyOpenAlpha_Evolve uses the local Heartly-Qwen-Coder model. The default configuration in `.env` points to the model at:

```
c:/Users/eivin/Desktop/latest Datasets organizer/Datasets organizer/Datasets organizer/heartly-qwen-code/heartly-qwen-code-lora
```

If you need to adjust the path or parameters, edit your `.env` file:

```bash
USE_LOCAL_MODEL=True
LOCAL_MODEL_PATH="/path/to/heartly-qwen-code"
LOCAL_MODEL_DEVICE="auto"
LOCAL_MODEL_DTYPE="fp16"
LOCAL_MODEL_MAX_TOKENS=512
LOCAL_MODEL_TEMPERATURE=0.7
LOCAL_MODEL_TOP_P=0.9
LOCAL_MODEL_TOP_K=40

# Optional boundary head quality gate
USE_BOUNDARY_HEAD=False
BOUNDARY_HEAD_PATH="probe_head.pkl"
BOUNDARY_HEAD_THRESHOLD=0.5
```

#### Using External LLMs (LiteLLM)

To use external LLM APIs instead of the local model, set `USE_LOCAL_MODEL=False` and configure your API keys:

```bash
USE_LOCAL_MODEL=False
LITELLM_DEFAULT_MODEL="gemini/gemini-2.0-flash-lite"
GEMINI_API_KEY="your_key"
```

### Run

```bash
# Run the example task (Dijkstra's algorithm)
python -m main examples/shortest_path.yaml

# Or launch the Gradio web interface
python app.py
```

---

## 🔮 Future Evolution

- Improved Heartly model training with more diverse code tasks
- Multi-turn conversational code generation
- Memory/state persistence for coding context
- Independent answer critic for confident-confabulation detection

---

## 🤝 Contributing

This is an open invitation to collaborate! Whether you're an AI researcher, a Python developer, or simply an enthusiast, your contributions are welcome.

- **Report Bugs**: Create an issue on GitHub!
- **Suggest Features**: Open an issue to discuss it!
- **Submit Pull Requests**: Fork, branch, write clean code, add tests, submit!

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE.md` file for details.

---

## 🙏 Credits & Attribution

This project is a **modified fork** of [**OpenAlpha_Evolve**](https://github.com/shyamsaktawat/OpenAlpha_Evolve) by **Shyam Saktawat** (MIT License, © 2024). We are deeply grateful for his open-source work on the evolutionary code-generation framework.

OpenAlpha_Evolve is inspired by the pioneering research of **Google DeepMind** on **AlphaEvolve** and related work in LLM-driven code generation and automated discovery.

The Heartly hallucination-reduction architecture is developed by **Eivin Tobias** (© 2026) as part of the [Heartly research project](https://github.com/eivintobias/heartly).

*Disclaimer: This is an experimental project. Generated code may not always be optimal, correct, or secure. Always review and test code thoroughly before using it in production.*