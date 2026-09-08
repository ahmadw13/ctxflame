"""
Analysis subpackage for ctxflame.
"""

from ctxflame.analysis.pricing import calculate_cost, get_model_spec, MODEL_REGISTRY
from ctxflame.analysis.bloat import analyze_bloat
from ctxflame.analysis.attention import analyze_attention
from ctxflame.analysis.profiler import profile_payload

__all__ = [
    "calculate_cost",
    "get_model_spec",
    "MODEL_REGISTRY",
    "analyze_bloat",
    "analyze_attention",
    "profile_payload",
]
