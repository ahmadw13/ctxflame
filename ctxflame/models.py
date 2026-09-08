"""
Data models for ctxflame payload profiling, metrics, and diagnostics.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    SYSTEM = "system"
    TOOL_SCHEMA = "tool_schema"
    CONVERSATION = "conversation"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DOCUMENT_CHUNK = "document_chunk"
    RAW_PROMPT = "raw_prompt"
    UNKNOWN = "unknown"


class TokenMetric(BaseModel):
    token_count: int = Field(default=0, description="Total tokens counted by selected tokenizer")
    char_count: int = Field(default=0, description="Character count in UTF-8")
    word_count: int = Field(default=0, description="Word count (whitespace delimited)")
    token_to_word_ratio: float = Field(default=1.0, description="Ratio of tokens per word (high in non-Latin scripts)")


class CostMetric(BaseModel):
    input_cost_usd: float = Field(default=0.0, description="Estimated input cost for this single payload in USD")
    cost_per_1m_calls_usd: float = Field(default=0.0, description="Estimated cost for 1,000,000 requests in USD")


class BloatIssue(BaseModel):
    severity: str = Field(description="Severity: info, warning, or critical")
    category: str = Field(description="Category: redundancy, tool_schema_bloat, whitespace, repetition")
    description: str = Field(description="Human-readable explanation of the bloat issue")
    estimated_wasted_tokens: int = Field(default=0, description="Estimated token overhead that can be recovered")
    suggestion: str = Field(description="Actionable suggestion to optimize the payload")


class AttentionPosition(BaseModel):
    section_name: str = Field(description="Name or title of the section")
    token_start: int = Field(description="Starting token index in serial context stream")
    token_end: int = Field(description="Ending token index in serial context stream")
    normalized_start: float = Field(description="Normalized starting position (0.0 to 1.0)")
    normalized_end: float = Field(description="Normalized ending position (0.0 to 1.0)")
    attention_risk: str = Field(description="Risk assessment: low, medium, high")
    reason: str = Field(description="Explanation of why this placement poses attention degradation risks")


class PayloadNode(BaseModel):
    id: str = Field(description="Unique node identifier")
    name: str = Field(description="Human-readable node label")
    node_type: NodeType = Field(default=NodeType.UNKNOWN, description="Structural type of node")
    role: Optional[str] = Field(default=None, description="Message role if applicable")
    content: Optional[str] = Field(default=None, description="Raw text or JSON content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attributes")
    metrics: TokenMetric = Field(default_factory=TokenMetric, description="Token and text metrics")
    cost: CostMetric = Field(default_factory=CostMetric, description="Cost metrics")
    children: List["PayloadNode"] = Field(default_factory=list, description="Sub-nodes or sections")


class ContextProfile(BaseModel):
    model_name: str = Field(description="Target model identifier (e.g. gpt-4o, gemini-2.5-flash)")
    tokenizer_name: str = Field(description="Tokenizer engine utilized (e.g. o200k_base, gemini-bpe)")
    total_tokens: int = Field(default=0, description="Total tokens in full payload")
    total_chars: int = Field(default=0, description="Total characters")
    total_words: int = Field(default=0, description="Total words")
    overall_token_to_word_ratio: float = Field(default=1.0, description="Global token to word ratio")
    total_cost_usd: float = Field(default=0.0, description="Total estimated input cost per execution")
    cost_per_1m_usd: float = Field(default=0.0, description="Total cost for 1,000,000 calls")
    context_limit: int = Field(default=128000, description="Maximum context window of target model")
    utilization_percent: float = Field(default=0.0, description="Percentage of context window utilized")
    section_breakdown: Dict[str, int] = Field(default_factory=dict, description="Tokens aggregated by NodeType")
    section_percentages: Dict[str, float] = Field(default_factory=dict, description="Percentage share per section")
    nodes: List[PayloadNode] = Field(default_factory=list, description="Top-level payload nodes")
    bloat_issues: List[BloatIssue] = Field(default_factory=list, description="Detected bloat or redundancy issues")
    attention_analysis: List[AttentionPosition] = Field(default_factory=list, description="Attention degradation map")
    lost_in_middle_score: float = Field(default=0.0, description="Score 0.0-100.0 indicating middle-degradation risk")
    summary: str = Field(default="", description="High-level diagnostic summary")
