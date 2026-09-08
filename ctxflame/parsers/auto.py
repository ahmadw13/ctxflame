"""
Auto-detection router for payload parsers in ctxflame.
"""

import json
from typing import Any, List, Tuple
from ctxflame.models import PayloadNode
from ctxflame.parsers.base import BaseParser
from ctxflame.parsers.openai import OpenAIParser
from ctxflame.parsers.anthropic import AnthropicParser
from ctxflame.parsers.gemini import GeminiParser
from ctxflame.parsers.text import RawTextParser
from ctxflame.tokenizers.base import BaseTokenizer

PARSERS: List[BaseParser] = [
    AnthropicParser(),
    GeminiParser(),
    OpenAIParser(),
    RawTextParser(),
]


def detect_and_parse(data: Any, tokenizer: BaseTokenizer) -> Tuple[str, List[PayloadNode]]:
    """
    Automatically detect the payload format and parse it using the matching parser.
    Returns a tuple of (format_name, list_of_payload_nodes).
    """
    # If string, check if it's valid JSON first
    parsed_data = data
    if isinstance(data, str):
        trimmed = data.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            try:
                parsed_data = json.loads(trimmed)
            except Exception:
                parsed_data = data

    for parser in PARSERS:
        if parser.can_parse(parsed_data):
            nodes = parser.parse(parsed_data, tokenizer)
            return parser.format_name, nodes

    # Default to raw text parser fallback
    fallback_parser = RawTextParser()
    return fallback_parser.format_name, fallback_parser.parse(str(data), tokenizer)
