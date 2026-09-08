"""
Parsers subpackage for ctxflame.
"""

from ctxflame.parsers.base import BaseParser
from ctxflame.parsers.openai import OpenAIParser
from ctxflame.parsers.anthropic import AnthropicParser
from ctxflame.parsers.gemini import GeminiParser
from ctxflame.parsers.text import RawTextParser
from ctxflame.parsers.auto import detect_and_parse

__all__ = [
    "BaseParser",
    "OpenAIParser",
    "AnthropicParser",
    "GeminiParser",
    "RawTextParser",
    "detect_and_parse",
]
