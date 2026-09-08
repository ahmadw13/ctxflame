"""
ctxflame: Context Window Profiler and Token Flamegraph for LLM and Agent Operations.
"""

from ctxflame.models import ContextProfile, PayloadNode, TokenMetric, CostMetric, BloatIssue, AttentionPosition
from ctxflame.analysis.profiler import profile_payload

__version__ = "0.1.0"
__all__ = [
    "profile_payload",
    "ContextProfile",
    "PayloadNode",
    "TokenMetric",
    "CostMetric",
    "BloatIssue",
    "AttentionPosition",
]
