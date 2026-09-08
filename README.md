# ctxflame

Context Window Profiler and Token Flamegraph for LLM and Agent Operations.

`ctxflame` is a command-line static analysis tool and profiler for Large Language Model (LLM) context windows, agent tool definitions, and Retrieval-Augmented Generation (RAG) pipelines. It breaks down token consumption by section, identifies prompt bloat and near-duplicate passages, calculates "Lost in the Middle" attention degradation risks, and generates interactive HTML flamegraphs.

---

## Why ctxflame?

In production AI systems, developers frequently stuff system directives, lengthy tool/function schemas, conversation histories, and retrieved document chunks into context windows without visibility into what is consuming tokens or degrading model attention:

1. **Hidden Tool Schema Bloat**: JSON Schema definitions for tools and function calls often silently consume 40% to 70% of the entire token budget before conversation even begins.
2. **"Lost in the Middle" Degradation**: Transformer models suffer severe recall degradation when critical instructions or retrieved facts are placed between 30% and 75% of the context stream.
3. **RAG Redundancy**: Vector retrievers often pull near-duplicate passages, burning budget on repetitive tokens.
4. **Multilingual Token Inflation**: Non-Latin scripts (Kurdish, Arabic, CJK) exhibit varying token-to-word ratios across tokenizers, leading to unexpected cost spikes.

`ctxflame` gives you complete static visibility and budget enforcement before requests reach external AI providers.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Input Payload Source                 │
│         (OpenAI JSON, Anthropic, Gemini, Raw)          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                    Payload Analyzers                   │
│   ├─ System Directives       ├─ Multi-turn Dialog      │
│   ├─ JSON Tool Schemas       ├─ Injected RAG Chunks    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Multi-Tokenizer Engine                 │
│   ├─ OpenAI (o200k/cl100k)   ├─ Gemini (SentencePiece) │
│   └─ Claude (Anthropic BPE)  └─ Script-Aware Metrics   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Diagnostic Suite                     │
│   ├─ Section Breakdown       ├─ Bloat / Jaccard Check  │
│   ├─ Attention Risk Scoring  ├─ Cost per 1M Calls      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Output Renderers                     │
│   ├─ Rich Terminal UI        ├─ Interactive HTML Flame │
│   ├─ Markdown / JSON Export  ├─ CI/CD Budget Gate      │
└────────────────────────────────────────────────────────┘
```

---

## Installation

### From Source

```bash
git clone https://github.com/ahmadw13/ctxflame.git
cd ctxflame

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # Linux / macOS

# Install dependencies and editable CLI binary
pip install -r requirements.txt
pip install -e .
```

### Docker Execution

```bash
docker build -t ctxflame .
docker run --rm -v ${PWD}/examples:/app/examples ctxflame profile examples/openai_agent.json
```

---

## Command Suite

### 1. `ctxflame profile`
Profiles a prompt payload and renders a comprehensive terminal dashboard:

```bash
# Terminal dashboard with tables and hierarchy tree
ctxflame profile examples/openai_agent.json

# Target a specific model for cost and tokenization calibration
ctxflame profile examples/gemini_rag.json --model gemini-2.5-flash

# Export structured JSON for piping into downstream tools
ctxflame profile examples/openai_agent.json --json

# Export GitHub-flavored Markdown for automated PR comments
ctxflame profile examples/raw_prompt.md --markdown
```

### 2. `ctxflame flamegraph`
Generates a standalone, self-contained interactive HTML flamegraph (works completely offline, zero CDN dependencies):

```bash
ctxflame flamegraph examples/openai_agent.json -o flamegraph.html --open
```

Hover over any block to inspect token count, percentage of total context, and estimated input cost. Click any parent section (such as Tool Schemas or Conversation Turns) to zoom into its internal breakdown.

### 3. `ctxflame diff`
Compares two payload versions side-by-side to track token, cost, and attention score deltas:

```bash
ctxflame diff prompt_v1.json prompt_v2.json --model gpt-4o
```

### 4. `ctxflame budget`
Automated CI/CD gate that verifies payload constraints and exits with code 1 upon violation:

```bash
# Fail if tokens exceed 8,000 or single-request cost exceeds $0.02
ctxflame budget examples/openai_agent.json --max-tokens 8000 --max-cost 0.02

# Enforce maximum Lost-in-the-Middle attention risk score
ctxflame budget examples/openai_agent.json --max-risk-score 60.0
```

### 5. `ctxflame models`
Displays the directory of supported model families, context limits, and current standard input pricing:

```bash
ctxflame models
```

---

## CLI Options Reference

| Command | Option | Description | Default |
| :--- | :--- | :--- | :--- |
| `profile` | `--model, -m` | Target model for tokenization and pricing | `gpt-4o` |
| `profile` | `--tokenizer, -t` | Explicit tokenizer override | Automatic |
| `profile` | `--json` | Output structured JSON to stdout | `False` |
| `profile` | `--markdown` | Output GitHub-flavored Markdown summary | `False` |
| `profile` | `--output, -o` | Write rendered output to a destination file | `None` |
| `flamegraph`| `--output, -o` | Path for generated standalone HTML report | `flamegraph.html` |
| `flamegraph`| `--open` | Automatically launch HTML in default browser | `False` |
| `diff` | `--model, -m` | Target model for delta calculation | `gpt-4o` |
| `budget` | `--max-tokens` | Token limit ceiling (exits 1 if exceeded) | `None` |
| `budget` | `--max-cost` | Cost ceiling in USD (exits 1 if exceeded) | `None` |
| `budget` | `--max-risk-score` | Max attention risk score 0-100 (exits 1 if exceeded) | `None` |

---

## Supported Payload Formats

`ctxflame` includes an intelligent auto-detection router (`ctxflame/parsers/auto.py`) that identifies formats without requiring manual configuration flags:

- **OpenAI Format**: JSON payloads containing `messages`, `tools`, and `functions`. Supports `tool_calls` and `tool` role responses.
- **Anthropic Format**: JSON payloads containing top-level `system`, `messages`, and `tools` with `input_schema`.
- **Google Gemini Format**: JSON payloads containing `systemInstruction`, `contents` turns, `functionCall`, and `functionDeclarations`.
- **Raw Text / Markdown**: Plain text or Markdown files. Sections are automatically delineated by Markdown headers (`# System`, `# Context`, `# Query`, etc.).

---

## GitHub Actions CI/CD Integration

Embed `ctxflame` into pull request workflows to prevent prompt regressions and budget inflation:

```yaml
name: Prompt Budget Gate

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'agents/**'

jobs:
  validate-prompts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install ctxflame
      - name: Verify Context Budget
        run: |
          ctxflame budget prompts/production_agent.json \
            --max-tokens 8000 \
            --max-cost 0.02 \
            --max-risk-score 50.0
```

---

## Testing

Execute the automated test suite covering tokenizers, parsers, bloat detection, pricing, and CLI commands:

```bash
python run_tests.py
```

All 22 unit tests execute in under 0.5 seconds.

---

## License

MIT License. Copyright (c) 2026 Ahmed A. See [LICENSE](LICENSE) for full text.
