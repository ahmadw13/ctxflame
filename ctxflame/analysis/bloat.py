"""
Bloat and redundancy diagnostic engine for ctxflame.
Identifies duplicate document chunks, excessive whitespace, and schema overhead.
"""

import re
from typing import List, Set
from ctxflame.models import BloatIssue, NodeType, PayloadNode


def _get_ngrams(text: str, n: int = 5) -> Set[str]:
    """Generate n-grams of words for fast text similarity comparison."""
    words = text.lower().split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity index between two n-gram sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def analyze_bloat(nodes: List[PayloadNode], total_tokens: int) -> List[BloatIssue]:
    """Inspect all nodes in payload and return detected bloat issues."""
    issues: List[BloatIssue] = []

    if total_tokens == 0:
        return issues

    # 1. Tool Schema Overhead Check
    tool_tokens = 0
    for node in nodes:
        if node.node_type == NodeType.TOOL_SCHEMA:
            tool_tokens += node.metrics.token_count

    if total_tokens > 0:
        tool_share = (tool_tokens / total_tokens) * 100.0
        if tool_share >= 40.0 and tool_tokens > 500:
            issues.append(
                BloatIssue(
                    severity="warning",
                    category="tool_schema_bloat",
                    description=f"Tool and function schemas consume {tool_share:.1f}% ({tool_tokens} tokens) of total payload.",
                    estimated_wasted_tokens=int(tool_tokens * 0.35),
                    suggestion="Prune unused tools or minify JSON schema parameter descriptions before prompting."
                )
            )

    # 2. Whitespace & Indentation Waste
    whitespace_wasted = 0
    for node in nodes:
        content = node.content or ""
        # Look for 4+ consecutive spaces or 3+ newlines
        excess_spaces = len(re.findall(r" {4,}", content))
        excess_newlines = len(re.findall(r"\n{3,}", content))
        if excess_spaces > 0 or excess_newlines > 0:
            whitespace_wasted += (excess_spaces * 2) + (excess_newlines * 2)

    if whitespace_wasted > 50:
        issues.append(
            BloatIssue(
                severity="info",
                category="whitespace",
                description=f"Found uncompacted formatting with ~{whitespace_wasted} tokens of whitespace and newline padding.",
                estimated_wasted_tokens=whitespace_wasted,
                suggestion="Strip extraneous indentation and collapse consecutive blank lines before sending."
            )
        )

    # 3. Duplicate / Near-Duplicate Content Check
    text_nodes = [n for n in nodes if n.content and len(n.content.split()) >= 15]
    ngrams_cache = [_get_ngrams(n.content or "") for n in text_nodes]

    seen_pairs = set()
    duplicate_tokens = 0

    for i in range(len(text_nodes)):
        for j in range(i + 1, len(text_nodes)):
            sim = _jaccard_similarity(ngrams_cache[i], ngrams_cache[j])
            if sim >= 0.70:
                pair_key = (min(i, j), max(i, j))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    waste = min(text_nodes[i].metrics.token_count, text_nodes[j].metrics.token_count)
                    duplicate_tokens += waste
                    issues.append(
                        BloatIssue(
                            severity="warning",
                            category="redundancy",
                            description=(
                                f"High similarity ({int(sim * 100)}%) detected between '{text_nodes[i].name}' "
                                f"and '{text_nodes[j].name}'."
                            ),
                            estimated_wasted_tokens=waste,
                            suggestion="Deduplicate retrieved RAG passages or conversation history turns."
                        )
                    )

    # 4. Monolithic Message Warning
    for node in nodes:
        if node.metrics.token_count > 16000 and node.node_type != NodeType.DOCUMENT_CHUNK:
            issues.append(
                BloatIssue(
                    severity="info",
                    category="repetition",
                    description=f"Node '{node.name}' is very large ({node.metrics.token_count} tokens) in a single block.",
                    estimated_wasted_tokens=0,
                    suggestion="Consider chunking or summarizing long message content into smaller referenced sections."
                )
            )

    return issues
