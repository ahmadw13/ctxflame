"""
Unit tests for ctxflame analysis, bloat detection, pricing, and profiler.
"""

import unittest
from ctxflame.analysis.pricing import calculate_cost, get_model_spec
from ctxflame.analysis.bloat import analyze_bloat
from ctxflame.analysis.attention import analyze_attention
from ctxflame.analysis.profiler import profile_payload
from ctxflame.models import NodeType, PayloadNode, TokenMetric, CostMetric


class TestAnalysis(unittest.TestCase):
    """Tests for pricing, bloat diagnostics, attention curves, and profiler."""

    def test_pricing_calculation(self):
        # 1,000,000 tokens of gpt-4o should be $2.50
        cost_1m = calculate_cost(1_000_000, "gpt-4o")
        self.assertAlmostEqual(cost_1m.input_cost_usd, 2.50, places=2)

        # 10,000 tokens of gemini-2.5-flash ($0.10/1M) -> $0.001
        cost_gemini = calculate_cost(10_000, "gemini-2.5-flash")
        self.assertAlmostEqual(cost_gemini.input_cost_usd, 0.001, places=3)

    def test_bloat_duplicate_detection(self):
        chunk = "This is a detailed retrieval passage about Kurdish natural language processing and document parsing studio."
        node1 = PayloadNode(
            id="doc_1",
            name="Document Passage 1",
            node_type=NodeType.DOCUMENT_CHUNK,
            content=chunk,
            metrics=TokenMetric(token_count=100, char_count=len(chunk), word_count=len(chunk.split()))
        )
        node2 = PayloadNode(
            id="doc_2",
            name="Document Passage 2",
            node_type=NodeType.DOCUMENT_CHUNK,
            content=chunk,  # Exact duplicate
            metrics=TokenMetric(token_count=100, char_count=len(chunk), word_count=len(chunk.split()))
        )

        issues = analyze_bloat([node1, node2], total_tokens=200)
        # Should flag high similarity / redundancy
        dup_issues = [i for i in issues if i.category == "redundancy"]
        self.assertGreaterEqual(len(dup_issues), 1)
        self.assertIn("High similarity", dup_issues[0].description)

    def test_attention_lost_in_middle(self):
        # If a system instruction is in the middle (e.g. node 2 of 3 where node 1 is 5000 tokens)
        node1 = PayloadNode(
            id="n1", name="Huge Doc", node_type=NodeType.DOCUMENT_CHUNK,
            metrics=TokenMetric(token_count=5000)
        )
        node2 = PayloadNode(
            id="n2", name="Buried System Instruction", node_type=NodeType.SYSTEM,
            metrics=TokenMetric(token_count=500)
        )
        node3 = PayloadNode(
            id="n3", name="End Query", node_type=NodeType.USER_MESSAGE,
            metrics=TokenMetric(token_count=5000)
        )

        positions, score = analyze_attention([node1, node2, node3], total_tokens=10500)
        self.assertEqual(len(positions), 3)
        # Buried system directive in the middle should be flagged as high risk
        self.assertEqual(positions[1].attention_risk, "high")
        self.assertGreater(score, 50.0)

    def test_end_to_end_profile_payload(self):
        raw_payload = {
            "messages": [
                {"role": "system", "content": "You are a Kurdish language assistant."},
                {"role": "user", "content": "بەخێربێن بۆ سیستەمەکە"}
            ]
        }

        profile = profile_payload(raw_payload, model_name="gpt-4o")
        self.assertGreater(profile.total_tokens, 0)
        self.assertIn("system", profile.section_breakdown)
        self.assertIn("user_message", profile.section_breakdown)
        self.assertGreater(profile.total_cost_usd, 0.0)
        self.assertEqual(profile.model_name, "gpt-4o")
        self.assertGreater(len(profile.summary), 10)


if __name__ == "__main__":
    unittest.main()
