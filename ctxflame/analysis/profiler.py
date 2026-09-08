"""
Core profiling orchestrator for ctxflame.
Parses payloads, computes multi-token metrics, costs, bloat diagnostics, and attention maps.
"""

from typing import Any, Dict, Optional
from ctxflame.models import ContextProfile, CostMetric, PayloadNode
from ctxflame.tokenizers.registry import get_tokenizer
from ctxflame.parsers.auto import detect_and_parse
from ctxflame.analysis.pricing import calculate_cost, get_model_spec
from ctxflame.analysis.bloat import analyze_bloat
from ctxflame.analysis.attention import analyze_attention


def profile_payload(
    data: Any,
    model_name: str = "gpt-4o",
    tokenizer_override: Optional[str] = None,
    context_limit_override: Optional[int] = None
) -> ContextProfile:
    """
    Profile an LLM payload across token distribution, cost, bloat, and attention risk.
    """
    # 1. Resolve Tokenizer
    tok_target = tokenizer_override if tokenizer_override else model_name
    tokenizer = get_tokenizer(tok_target)

    # 2. Parse Payload Nodes
    format_name, nodes = detect_and_parse(data, tokenizer)

    # 3. Aggregate Node Metrics & Costs
    model_spec = get_model_spec(model_name)
    context_limit = context_limit_override if context_limit_override else model_spec.context_limit

    total_tokens = 0
    total_chars = 0
    total_words = 0
    section_breakdown: Dict[str, int] = {}

    for node in nodes:
        node_cost = calculate_cost(node.metrics.token_count, model_name)
        node.cost = node_cost

        # If node has children, ensure child costs are also populated
        for child in node.children:
            child.cost = calculate_cost(child.metrics.token_count, model_name)

        total_tokens += node.metrics.token_count
        total_chars += node.metrics.char_count
        total_words += node.metrics.word_count

        sec_type = node.node_type.value
        section_breakdown[sec_type] = section_breakdown.get(sec_type, 0) + node.metrics.token_count

    # 4. Percentages
    section_percentages: Dict[str, float] = {}
    if total_tokens > 0:
        for sec, count in section_breakdown.items():
            section_percentages[sec] = round((count / total_tokens) * 100.0, 1)

    overall_ratio = round(total_tokens / total_words, 2) if total_words > 0 else 1.0
    total_cost = calculate_cost(total_tokens, model_name)
    utilization_pct = round((total_tokens / context_limit) * 100.0, 2) if context_limit > 0 else 0.0

    # 5. Bloat Diagnostics
    bloat_issues = analyze_bloat(nodes, total_tokens)

    # 6. Attention & Lost in the Middle Risk
    attention_positions, lost_score = analyze_attention(nodes, total_tokens)

    # 7. Summary
    summary = (
        f"Payload contains {total_tokens:,} tokens ({utilization_pct:.1f}% of {context_limit:,} context limit) "
        f"under model '{model_name}'. Format detected: '{format_name}'. "
        f"Estimated cost: ${total_cost.input_cost_usd:.5f} USD (${total_cost.cost_per_1m_calls_usd:.2f} / 1M calls)."
    )

    return ContextProfile(
        model_name=model_name,
        tokenizer_name=tokenizer.name,
        total_tokens=total_tokens,
        total_chars=total_chars,
        total_words=total_words,
        overall_token_to_word_ratio=overall_ratio,
        total_cost_usd=total_cost.input_cost_usd,
        cost_per_1m_usd=total_cost.cost_per_1m_calls_usd,
        context_limit=context_limit,
        utilization_percent=utilization_pct,
        section_breakdown=section_breakdown,
        section_percentages=section_percentages,
        nodes=nodes,
        bloat_issues=bloat_issues,
        attention_analysis=attention_positions,
        lost_in_middle_score=lost_score,
        summary=summary
    )
