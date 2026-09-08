"""
Model pricing database and context window specifications for ctxflame.
Provides current standard input pricing per 1M tokens in USD.
"""

from typing import Dict, NamedTuple
from ctxflame.models import CostMetric


class ModelSpec(NamedTuple):
    name: str
    provider: str
    context_limit: int
    input_cost_per_1m: float  # In USD


# Known model specifications and standard pricing (per 1M input tokens)
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # OpenAI
    "gpt-4o": ModelSpec("gpt-4o", "OpenAI", 128000, 2.50),
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", "OpenAI", 128000, 0.15),
    "gpt-4-turbo": ModelSpec("gpt-4-turbo", "OpenAI", 128000, 10.00),
    "gpt-4": ModelSpec("gpt-4", "OpenAI", 8192, 30.00),
    "gpt-3.5-turbo": ModelSpec("gpt-3.5-turbo", "OpenAI", 16385, 0.50),
    "o1": ModelSpec("o1", "OpenAI", 200000, 15.00),
    "o1-mini": ModelSpec("o1-mini", "OpenAI", 128000, 3.00),
    "o3-mini": ModelSpec("o3-mini", "OpenAI", 200000, 1.10),

    # Google Gemini
    "gemini-2.5-flash": ModelSpec("gemini-2.5-flash", "Google", 1048576, 0.10),
    "gemini-1.5-flash": ModelSpec("gemini-1.5-flash", "Google", 1048576, 0.075),
    "gemini-1.5-pro": ModelSpec("gemini-1.5-pro", "Google", 2097152, 1.25),

    # Anthropic Claude
    "claude-3-5-sonnet": ModelSpec("claude-3-5-sonnet", "Anthropic", 200000, 3.00),
    "claude-3-haiku": ModelSpec("claude-3-haiku", "Anthropic", 200000, 0.25),
    "claude-3-opus": ModelSpec("claude-3-opus", "Anthropic", 200000, 15.00),

    # Open / Other
    "deepseek-v3": ModelSpec("deepseek-v3", "DeepSeek", 64000, 0.14),
    "deepseek-r1": ModelSpec("deepseek-r1", "DeepSeek", 64000, 0.55),
    "llama-3.1-70b": ModelSpec("llama-3.1-70b", "Meta", 128000, 0.60),
    "llama-3.1-8b": ModelSpec("llama-3.1-8b", "Meta", 128000, 0.15),
}


def get_model_spec(model_name: str) -> ModelSpec:
    """Resolve model string to a known ModelSpec or sensible default."""
    cleaned = (model_name or "").lower().strip()

    if cleaned in MODEL_REGISTRY:
        return MODEL_REGISTRY[cleaned]

    for key, spec in MODEL_REGISTRY.items():
        if key in cleaned:
            return spec

    # Default fallback: 128k context, $2.50 / 1M
    return ModelSpec(
        name=model_name or "default-model",
        provider="Generic",
        context_limit=128000,
        input_cost_per_1m=2.50
    )


def calculate_cost(tokens: int, model_name: str) -> CostMetric:
    """Calculate USD input cost for a token count under a target model."""
    spec = get_model_spec(model_name)
    cost_per_token = spec.input_cost_per_1m / 1_000_000.0
    single_cost = round(tokens * cost_per_token, 6)
    cost_1m = round(single_cost * 1_000_000.0, 2)

    return CostMetric(
        input_cost_usd=single_cost,
        cost_per_1m_calls_usd=cost_1m
    )
