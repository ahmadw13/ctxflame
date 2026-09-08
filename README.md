# ctxflame

Context Window Profiler and Token Flamegraph for LLM and Agent Operations.

## Overview

`ctxflame` is a command-line profiler and static analysis tool for Large Language Model (LLM) context windows, agent tool definitions, and Retrieval-Augmented Generation (RAG) pipelines. It breaks down token consumption by payload section, detects prompt bloat and redundancy, evaluates "Lost in the Middle" attention degradation risks, and generates interactive flamegraphs.

## Installation

```bash
git clone https://github.com/ahmadw13/ctxflame.git
cd ctxflame
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

## Quick Start

```bash
# Profile an LLM payload in your terminal
ctxflame profile examples/openai_agent.json

# Generate an interactive HTML flamegraph
ctxflame flamegraph examples/openai_agent.json -o flamegraph.html

# Evaluate context budget in CI/CD pipelines
ctxflame budget examples/openai_agent.json --max-tokens 8000 --max-cost 0.05
```

## License

MIT License. Copyright (c) 2026 Ahmed A.
