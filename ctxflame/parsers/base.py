"""
Abstract base class for payload parsers in ctxflame.
"""

from abc import ABC, abstractmethod
from typing import Any, List
from ctxflame.models import PayloadNode
from ctxflame.tokenizers.base import BaseTokenizer


class BaseParser(ABC):
    """Base interface for parsing different LLM payload formats into PayloadNodes."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the payload format handled by this parser."""
        pass

    @abstractmethod
    def can_parse(self, data: Any) -> bool:
        """Return True if this parser can handle the given raw data."""
        pass

    @abstractmethod
    def parse(self, data: Any, tokenizer: BaseTokenizer) -> List[PayloadNode]:
        """Parse raw data into a structured tree of PayloadNodes."""
        pass
