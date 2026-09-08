"""
Abstract base class for ctxflame tokenizers.
"""

from abc import ABC, abstractmethod
from typing import List
from ctxflame.models import TokenMetric


class BaseTokenizer(ABC):
    """Base interface for all tokenizer implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier of the tokenizer."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the total number of tokens for the given text."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token ID list."""
        pass

    def calculate_metrics(self, text: str) -> TokenMetric:
        """Calculate complete token, character, and word metrics."""
        if not text:
            return TokenMetric(
                token_count=0,
                char_count=0,
                word_count=0,
                token_to_word_ratio=1.0
            )

        char_count = len(text)
        words = text.split()
        word_count = len(words)
        token_count = self.count_tokens(text)

        ratio = round(token_count / word_count, 2) if word_count > 0 else 1.0

        return TokenMetric(
            token_count=token_count,
            char_count=char_count,
            word_count=word_count,
            token_to_word_ratio=ratio
        )
