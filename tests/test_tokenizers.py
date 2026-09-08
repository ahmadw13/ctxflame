"""
Unit tests for ctxflame multi-tokenizer engine.
"""

import unittest
from ctxflame.tokenizers.openai import OpenAITokenizer
from ctxflame.tokenizers.gemini import GeminiTokenizer
from ctxflame.tokenizers.claude import ClaudeTokenizer
from ctxflame.tokenizers.registry import get_tokenizer


class TestTokenizers(unittest.TestCase):
    """Tests for tokenizer implementations and registry resolution."""

    def test_openai_tokenizer_basic(self):
        tok = OpenAITokenizer(encoding_name="o200k_base")
        text = "Hello world! This is a test."
        count = tok.count_tokens(text)
        self.assertGreater(count, 0)
        self.assertEqual(len(tok.encode(text)), count)

    def test_gemini_tokenizer_multilingual_kurdish(self):
        tok = GeminiTokenizer(model_name="gemini-2.5-flash")
        # Kurdish text with standardized Kurdish Unicode
        kurdish_text = "ئەمە تاقیکردنەوەیەکی پرۆسێسکردنی بەڵگەنامەی کوردییە لە سیستەمەکەدا"
        metrics = tok.calculate_metrics(kurdish_text)

        self.assertGreater(metrics.token_count, 0)
        self.assertGreater(metrics.word_count, 0)
        self.assertGreater(metrics.char_count, 0)
        # Gemini token ratio should be efficient for Kurdish
        self.assertLessEqual(metrics.token_to_word_ratio, 3.0)

    def test_claude_tokenizer_metrics(self):
        tok = ClaudeTokenizer()
        text = "Claude analyzes structured agent prompts with precision."
        metrics = tok.calculate_metrics(text)
        self.assertEqual(metrics.word_count, 7)
        self.assertGreater(metrics.token_count, 5)

    def test_empty_string_metrics(self):
        tok = OpenAITokenizer()
        metrics = tok.calculate_metrics("")
        self.assertEqual(metrics.token_count, 0)
        self.assertEqual(metrics.word_count, 0)
        self.assertEqual(metrics.char_count, 0)

    def test_registry_resolution(self):
        self.assertIsInstance(get_tokenizer("gpt-4o"), OpenAITokenizer)
        self.assertIsInstance(get_tokenizer("gemini-1.5-pro"), GeminiTokenizer)
        self.assertIsInstance(get_tokenizer("claude-3-5-sonnet"), ClaudeTokenizer)
        self.assertIsInstance(get_tokenizer("deepseek-v3"), OpenAITokenizer)


if __name__ == "__main__":
    unittest.main()
