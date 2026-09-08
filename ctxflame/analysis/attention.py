"""
Attention degradation and "Lost in the Middle" analysis engine for ctxflame.
Measures placement risk based on positional attention curves in Transformer models.
"""

from typing import List, Tuple
from ctxflame.models import AttentionPosition, NodeType, PayloadNode


def analyze_attention(nodes: List[PayloadNode], total_tokens: int) -> Tuple[List[AttentionPosition], float]:
    """
    Calculate serial token positions and evaluate attention degradation risk.
    Returns (list of AttentionPositions, aggregate lost_in_the_middle_risk_score).
    """
    if total_tokens == 0 or not nodes:
        return [], 0.0

    positions: List[AttentionPosition] = []
    current_token_offset = 0
    middle_risk_penalties: List[float] = []

    for node in nodes:
        node_tokens = node.metrics.token_count
        start_tok = current_token_offset
        end_tok = current_token_offset + node_tokens
        current_token_offset = end_tok

        norm_start = round(start_tok / total_tokens, 3)
        norm_end = round(end_tok / total_tokens, 3)
        midpoint = (norm_start + norm_end) / 2.0

        # Evaluate risk
        risk = "low"
        reason = "Position sits within optimal attention zones (primacy or recency boundary)."

        # The danger zone in Transformer attention is roughly 0.30 to 0.75
        is_in_middle_zone = 0.30 <= midpoint <= 0.75

        if is_in_middle_zone:
            if node.node_type in (NodeType.SYSTEM, NodeType.USER_MESSAGE):
                # Critical directives buried in the middle are high risk
                risk = "high"
                reason = "Critical prompt directive placed in the middle 30-75% zone where recall degrades most."
                middle_risk_penalties.append(85.0)
            elif node.node_type == NodeType.TOOL_SCHEMA:
                risk = "medium"
                reason = "Tool schema placed mid-stream; model may miss fine-grained parameter constraints."
                middle_risk_penalties.append(45.0)
            elif node.node_type == NodeType.DOCUMENT_CHUNK:
                risk = "medium"
                reason = "Document passage in the middle zone has lower retrieval recall than start/end passages."
                middle_risk_penalties.append(25.0)
            else:
                risk = "low"
        elif midpoint < 0.20:
            # Primacy zone
            risk = "low"
            reason = "Primacy zone: Model attention and instruction compliance are strongest here."
        else:
            # Recency zone
            risk = "low"
            reason = "Recency zone: Positioned close to generation trigger; strong attention retention."

        positions.append(
            AttentionPosition(
                section_name=node.name,
                token_start=start_tok,
                token_end=end_tok,
                normalized_start=norm_start,
                normalized_end=norm_end,
                attention_risk=risk,
                reason=reason
            )
        )

    # Compute overall score: 0 (optimal) to 100 (maximum risk)
    if middle_risk_penalties:
        aggregate_score = min(100.0, round(sum(middle_risk_penalties) / len(middle_risk_penalties), 1))
    else:
        aggregate_score = 10.0  # baseline minimal risk

    return positions, aggregate_score
