"""
Unit tests for ctxflame payload parsers.
"""

import json
import unittest
from ctxflame.models import NodeType
from ctxflame.tokenizers.openai import OpenAITokenizer
from ctxflame.parsers.auto import detect_and_parse
from ctxflame.parsers.openai import OpenAIParser
from ctxflame.parsers.anthropic import AnthropicParser
from ctxflame.parsers.gemini import GeminiParser
from ctxflame.parsers.text import RawTextParser


class TestParsers(unittest.TestCase):
    """Tests for OpenAI, Anthropic, Gemini, and RawText parsers."""

    def setUp(self):
        self.tokenizer = OpenAITokenizer("o200k_base")

    def test_openai_parser_messages_and_tools(self):
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"location\": \"Erbil\"}"}
                        }
                    ]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 28}"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                    }
                }
            ]
        }

        format_name, nodes = detect_and_parse(payload, self.tokenizer)
        self.assertIn("OpenAI", format_name)
        # Should have tools root + 4 messages
        self.assertEqual(len(nodes), 5)
        self.assertEqual(nodes[0].node_type, NodeType.TOOL_SCHEMA)
        self.assertEqual(nodes[1].node_type, NodeType.SYSTEM)
        self.assertEqual(nodes[2].node_type, NodeType.USER_MESSAGE)
        self.assertEqual(nodes[3].node_type, NodeType.ASSISTANT_MESSAGE)
        self.assertEqual(nodes[4].node_type, NodeType.TOOL_RESULT)

    def test_anthropic_parser(self):
        payload = {
            "system": "You are an analytical researcher.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Analyze document."}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu_1", "name": "search_db", "input": {"q": "reports"}}
                    ]
                }
            ],
            "tools": [
                {"name": "search_db", "description": "Search database", "input_schema": {"type": "object"}}
            ]
        }

        format_name, nodes = detect_and_parse(payload, self.tokenizer)
        self.assertIn("Anthropic", format_name)
        # System + tools root + 2 messages = 4 nodes
        self.assertEqual(len(nodes), 4)
        self.assertEqual(nodes[0].node_type, NodeType.SYSTEM)
        self.assertEqual(nodes[1].node_type, NodeType.TOOL_SCHEMA)
        self.assertEqual(nodes[2].node_type, NodeType.USER_MESSAGE)
        self.assertEqual(nodes[3].node_type, NodeType.ASSISTANT_MESSAGE)

    def test_gemini_parser(self):
        payload = {
            "systemInstruction": {"parts": [{"text": "You are a multilingual AI."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Translate to Kurdish."}]},
                {"role": "model", "parts": [{"text": "Translation done."}]}
            ]
        }

        format_name, nodes = detect_and_parse(payload, self.tokenizer)
        self.assertIn("Gemini", format_name)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0].node_type, NodeType.SYSTEM)
        self.assertEqual(nodes[1].node_type, NodeType.USER_MESSAGE)
        self.assertEqual(nodes[2].node_type, NodeType.ASSISTANT_MESSAGE)

    def test_raw_text_markdown_parser(self):
        raw_prompt = """# System Directives
Always output valid JSON without explanations.

# Retrieved Context Documents
Document passage 1: Customer record 402.

# User Query
Summarize the account status."""

        format_name, nodes = detect_and_parse(raw_prompt, self.tokenizer)
        self.assertIn("Raw Text", format_name)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0].node_type, NodeType.SYSTEM)
        self.assertEqual(nodes[1].node_type, NodeType.DOCUMENT_CHUNK)
        self.assertEqual(nodes[2].node_type, NodeType.USER_MESSAGE)


if __name__ == "__main__":
    unittest.main()
