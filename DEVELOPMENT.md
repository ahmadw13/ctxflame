# Developer Guide & Architecture Documentation — ctxflame

This document provides developer setup instructions, architecture guidelines, tokenizer extension guides, and pull request standards for contributing to `ctxflame`.

---

## 1. Prerequisites

- **Python**: Version 3.11 or higher (Python 3.13 tested and supported).
- **Git**: Configured for your GitHub account.
- **Docker**: (Optional) For containerized execution.

---

## 2. Local Development Setup

### 2.1 Virtual Environment Initialization

```bash
# Clone repository
git clone https://github.com/ahmadw13/ctxflame.git
cd ctxflame

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2.2 Running Automated Tests

Run the test suite using standard discovery or the dedicated runner:

```bash
# Using dedicated runner
python run_tests.py

# Using standard unittest discovery
python -m unittest discover -s tests
```

---

## 3. System Architecture

The `ctxflame` engine is organized into four modular layers:

```
┌────────────────────────────────────────────────────────┐
│                   CLI & Interface                      │
│   (Typer commands: profile, flamegraph, diff, budget)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Payload Parsers                      │
│ (OpenAI, Anthropic, Gemini, RawText Auto-Router)       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                Multi-Tokenizer Engine                  │
│ (OpenAI o200k/cl100k, Gemini SP256k, Claude BPE)       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Analysis Engine                       │
│ (Bloat/Jaccard, Positional Attention, Pricing Models)  │
└────────────────────────────────────────────────────────┘
```

### 3.1 Parsers (`ctxflame/parsers/`)
- All parsers inherit from `BaseParser` in `ctxflame/parsers/base.py`.
- Parsers implement:
  - `can_parse(data: Any) -> bool`: Detects whether payload matches format signatures.
  - `parse(data: Any, tokenizer: BaseTokenizer) -> List[PayloadNode]`: Extracts structured sections into a hierarchical node tree.
- To add a new format (e.g. Cohere, Mistral), implement `BaseParser` and register the instance in `ctxflame/parsers/auto.py`.

### 3.2 Tokenizers (`ctxflame/tokenizers/`)
- All tokenizers inherit from `BaseTokenizer` in `ctxflame/tokenizers/base.py`.
- Tokenizers implement:
  - `count_tokens(text: str) -> int`
  - `encode(text: str) -> List[int]`
  - `calculate_metrics(text: str) -> TokenMetric`: Automatically derives word count, character count, and token-to-word ratios.
- `GeminiTokenizer` uses SentencePiece 256k calibration optimized for multilingual Unicode text (Kurdish, Arabic, CJK).

### 3.3 Diagnostics (`ctxflame/analysis/`)
- **Bloat Detection (`bloat.py`)**: Uses 5-gram Jaccard similarity to identify near-duplicate document passages (>= 70% threshold), flags excessive indentation whitespace, and warns when tool schemas consume > 40% of the total context.
- **Positional Attention (`attention.py`)**: Calculates serial token offsets and flags prompt directives placed in the 30% to 75% middle danger zone where transformer recall degrades most.
- **Pricing (`pricing.py`)**: Maintains a registry of model specs, context window limits, and input costs per 1,000,000 tokens.

---

## 4. Development Guidelines & Standards

### 4.1 Zero Emoji Policy
Do not use emojis in code, CLI output, logging, test names, commit messages, or documentation. Use clean ASCII/Unicode box-drawing characters and text labels instead.

### 4.2 Multilingual and Non-Latin Script Parity
When implementing or modifying tokenizers or parsers, verify that non-Latin scripts (specifically Kurdish Sorani Unicode: `ڵ`, `ڕ`, `ێ`, `ۆ`, and Arabic/Persian) are handled cleanly without character mangling or incorrect word splitting.

### 4.3 Pull Request Standards
- Every pull request must adhere to the standard template in [.github/pull_request_template.md](.github/pull_request_template.md).
- All 22 automated tests must pass cleanly (`python run_tests.py`).
- Relevant domain labels must be assigned:
  - `core`: Profiler engine, tokenizers, and parsers.
  - `cli`: Typer application and CLI subcommands.
  - `visualizers`: Terminal dashboard and HTML flamegraph renderer.
  - `infra`: Packaging, virtualenv, and Docker.
  - `documentation`: Documentation and guides.

---

## 5. Containerized Execution

Build and run `ctxflame` inside a lightweight Docker container:

```bash
# Build Docker image
docker build -t ctxflame .

# Run CLI inside container
docker run --rm -v ${PWD}/examples:/app/examples ctxflame profile examples/openai_agent.json
```
