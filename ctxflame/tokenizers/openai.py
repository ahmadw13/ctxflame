"""
OpenAI tiktoken tokenizer implementation for ctxflame.
Supports o200k_base (GPT-4o, GPT-4o-mini, o1) and cl100k_base (GPT-4, GPT-3.5).
"""

from typing import List, Optional
import tiktoken
from ctxflame.tokenizers.base import BaseTokenizer


class OpenAITokenizer(BaseTokenizer):
    """Tokenizes text using OpenAI's official tiktoken library."""

    def __init__(self, encoding_name: str = "o200k_base"):
        self._encoding_name = encoding_name
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to cl100k_base if o200k_base is unavailable
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self._encoding_name = "cl100k_base"

    @property
    def name(self) -> str:
        return self._encoding_name

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._encoding.encode(text, disallowed_special=()))
        except Exception:
            return len(self._encoding.encode_ordinary(text))

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        try:
            return self._encoding.encode(text, disallowed_special=())
        except Exception:
            return self._encoding.encode_ordinary(text)
