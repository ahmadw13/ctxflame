"""
Unit tests for ctxflame Typer CLI commands.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typer.testing import CliRunner
from ctxflame.cli import app

runner = CliRunner()


class TestCLI(unittest.TestCase):
    """Tests for ctxflame CLI commands."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.examples_dir = self.root_dir / "examples"
        self.openai_file = self.examples_dir / "openai_agent.json"
        self.gemini_file = self.examples_dir / "gemini_rag.json"
        self.raw_prompt_file = self.examples_dir / "raw_prompt.md"

    def test_cli_version(self):
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ctxflame version", result.output)

    def test_models_command(self):
        result = runner.invoke(app, ["models"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("gpt-4o", result.output)
        self.assertIn("gemini-2.5-flash", result.output)
        self.assertIn("claude-3-5-sonnet", result.output)

    def test_profile_command_terminal(self):
        result = runner.invoke(app, ["profile", str(self.openai_file)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Context Window Profile", result.output)
        self.assertIn("Section Breakdown", result.output)

    def test_profile_command_json(self):
        result = runner.invoke(app, ["profile", str(self.gemini_file), "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("total_tokens", data)
        self.assertIn("section_breakdown", data)
        self.assertGreater(data["total_tokens"], 0)

    def test_profile_command_markdown(self):
        result = runner.invoke(app, ["profile", str(self.raw_prompt_file), "--markdown"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ctxflame Context Profile Summary", result.output)

    def test_flamegraph_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "flame.html"
            result = runner.invoke(app, ["flamegraph", str(self.openai_file), "-o", str(out_file)])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(out_file.is_file())
            html_text = out_file.read_text(encoding="utf-8")
            self.assertIn("ctxflame Token Flamegraph", html_text)
            self.assertIn("flame-block", html_text)

    def test_diff_command(self):
        result = runner.invoke(app, ["diff", str(self.openai_file), str(self.gemini_file)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Prompt Payload Diff", result.output)
        self.assertIn("Token Delta", result.output)

    def test_budget_command_pass(self):
        # Allow up to 100,000 tokens
        result = runner.invoke(app, ["budget", str(self.openai_file), "--max-tokens", "100000", "--max-cost", "1.00"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Budget Check PASSED", result.output)

    def test_budget_command_fail(self):
        # Set artificially tiny limit of 5 tokens
        result = runner.invoke(app, ["budget", str(self.openai_file), "--max-tokens", "5"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Budget Check FAILED", result.output)


if __name__ == "__main__":
    unittest.main()
