"""
Registry to resolve model names to specific tokenizer instances.
"""

from typing import Dict, Type
from ctxflame.tokenizers.base import BaseTokenizer
from ctxflame.tokenizers.openai import OpenAITokenizer
from ctxflame.tokenizers.gemini import GeminiTokenizer
from ctxflame.tokenizers.claude import ClaudeTokenizer


def get_tokenizer(model_or_encoding: str = "gpt-4o") -> BaseTokenizer:
    """
    Resolve a model name or tokenizer encoding to an active BaseTokenizer instance.
    """
    cleaned = (model_or_encoding or "").lower().strip()

    # Gemini family
    if "gemini" in cleaned:
        return GeminiTokenizer(model_name=cleaned)

    # Claude family
    if "claude" in cleaned or "anthropic" in cleaned:
        return ClaudeTokenizer(model_name=cleaned)

    # Legacy OpenAI models
    if any(m in cleaned for m in ["gpt-3.5", "gpt-4-turbo", "gpt-4-0314", "gpt-4-0613", "cl100k"]):
        return OpenAITokenizer(encoding_name="cl100k_base")

    # DeepSeek and Llama commonly use 128k BPE similar to cl100k
    if any(m in cleaned for m in ["deepseek", "llama-3", "mistral"]):
        return OpenAITokenizer(encoding_name="cl100k_base")

    # Default to OpenAI o200k_base (GPT-4o, GPT-4o-mini, o1)
    return OpenAITokenizer(encoding_name="o200k_base")
