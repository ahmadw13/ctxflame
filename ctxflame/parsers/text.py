"""
Raw text and markdown prompt parser for ctxflame.
Parses flat prompt files, breaking them down into sections by markdown headers or role markers.
"""

import re
from typing import Any, List
from ctxflame.models import CostMetric, NodeType, PayloadNode
from ctxflame.parsers.base import BaseParser
from ctxflame.tokenizers.base import BaseTokenizer


class RawTextParser(BaseParser):
    """Parses raw text or markdown prompts by detecting sections."""

    @property
    def format_name(self) -> str:
        return "Raw Text / Markdown Prompt"

    def can_parse(self, data: Any) -> bool:
        return isinstance(data, str)

    def parse(self, data: Any, tokenizer: BaseTokenizer) -> List[PayloadNode]:
        text = str(data).strip()
        if not text:
            return []

        # Check if text contains markdown section headers (e.g. # Header or ## Header)
        header_pattern = re.compile(r"^(#{1,3}\s+[^\n]+)", re.MULTILINE)
        splits = header_pattern.split(text)

        nodes: List[PayloadNode] = []

        if len(splits) > 1:
            # First element could be text before any header
            current_header = "Preamble"
            first_chunk = splits[0].strip()
            if first_chunk:
                metrics = tokenizer.calculate_metrics(first_chunk)
                nodes.append(
                    PayloadNode(
                        id="sec_0",
                        name="Preamble",
                        node_type=NodeType.RAW_PROMPT,
                        content=first_chunk,
                        metadata={},
                        metrics=metrics,
                        cost=CostMetric(),
                        children=[]
                    )
                )

            # Process header + content pairs
            i = 1
            idx = 1
            while i < len(splits):
                header_raw = splits[i].strip()
                header_clean = re.sub(r"^#+\s*", "", header_raw)
                content_chunk = splits[i + 1].strip() if i + 1 < len(splits) else ""

                header_lower = header_clean.lower()
                if any(k in header_lower for k in ["system", "instruction", "rules", "developer"]):
                    node_type = NodeType.SYSTEM
                elif any(k in header_lower for k in ["context", "document", "retriev", "chunk", "rag"]):
                    node_type = NodeType.DOCUMENT_CHUNK
                elif any(k in header_lower for k in ["user", "query", "question", "input"]):
                    node_type = NodeType.USER_MESSAGE
                elif any(k in header_lower for k in ["assistant", "response", "output"]):
                    node_type = NodeType.ASSISTANT_MESSAGE
                elif any(k in header_lower for k in ["tool", "function", "schema"]):
                    node_type = NodeType.TOOL_SCHEMA
                else:
                    node_type = NodeType.RAW_PROMPT

                full_section_text = f"{header_raw}\n\n{content_chunk}".strip()
                metrics = tokenizer.calculate_metrics(full_section_text)

                nodes.append(
                    PayloadNode(
                        id=f"sec_{idx}",
                        name=header_clean,
                        node_type=node_type,
                        content=full_section_text,
                        metadata={"header": header_clean},
                        metrics=metrics,
                        cost=CostMetric(),
                        children=[]
                    )
                )
                idx += 1
                i += 2
        else:
            # Single flat block of text
            metrics = tokenizer.calculate_metrics(text)
            nodes.append(
                PayloadNode(
                    id="raw_prompt",
                    name="Raw Prompt Content",
                    node_type=NodeType.RAW_PROMPT,
                    content=text,
                    metadata={},
                    metrics=metrics,
                    cost=CostMetric(),
                    children=[]
                )
            )

        return nodes
