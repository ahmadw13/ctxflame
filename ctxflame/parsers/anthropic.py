"""
Anthropic Messages API format parser for ctxflame.
Parses system prompts, messages, tool definitions, and tool results.
"""

import json
from typing import Any, List
from ctxflame.models import CostMetric, NodeType, PayloadNode
from ctxflame.parsers.base import BaseParser
from ctxflame.tokenizers.base import BaseTokenizer


class AnthropicParser(BaseParser):
    """Parses Anthropic Messages API payloads."""

    @property
    def format_name(self) -> str:
        return "Anthropic Messages Format"

    def can_parse(self, data: Any) -> bool:
        if isinstance(data, dict):
            if "messages" in data and ("system" in data or "anthropic_version" in data):
                return True
            tools = data.get("tools")
            if tools and isinstance(tools, list):
                if any(isinstance(t, dict) and "input_schema" in t for t in tools):
                    return True
        return False

    def parse(self, data: Any, tokenizer: BaseTokenizer) -> List[PayloadNode]:
        nodes: List[PayloadNode] = []

        # 1. System Prompt (top-level key in Anthropic API)
        system_content = data.get("system")
        if system_content:
            if isinstance(system_content, list):
                sys_text = "\n".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in system_content
                )
            else:
                sys_text = str(system_content)

            metrics = tokenizer.calculate_metrics(sys_text)
            nodes.append(
                PayloadNode(
                    id="system_prompt",
                    name="System Instructions",
                    node_type=NodeType.SYSTEM,
                    role="system",
                    content=sys_text,
                    metadata={},
                    metrics=metrics,
                    cost=CostMetric(),
                    children=[]
                )
            )

        # 2. Tool Definitions
        tools = data.get("tools") or []
        if tools:
            tool_children: List[PayloadNode] = []
            for idx, tool in enumerate(tools):
                t_name = tool.get("name", f"tool_{idx + 1}") if isinstance(tool, dict) else f"tool_{idx + 1}"
                t_content = json.dumps(tool, indent=2)
                t_metrics = tokenizer.calculate_metrics(t_content)

                tool_children.append(
                    PayloadNode(
                        id=f"tool_schema_{idx + 1}",
                        name=f"Tool: {t_name}",
                        node_type=NodeType.TOOL_SCHEMA,
                        content=t_content,
                        metadata={"name": t_name},
                        metrics=t_metrics,
                        cost=CostMetric(),
                        children=[]
                    )
                )

            total_tool_text = "\n".join(c.content or "" for c in tool_children)
            parent_metrics = tokenizer.calculate_metrics(total_tool_text)
            nodes.append(
                PayloadNode(
                    id="tool_schemas_root",
                    name=f"Tool Schemas ({len(tool_children)} defined)",
                    node_type=NodeType.TOOL_SCHEMA,
                    content=total_tool_text,
                    metadata={"count": len(tool_children)},
                    metrics=parent_metrics,
                    cost=CostMetric(),
                    children=tool_children
                )
            )

        # 3. Conversation Messages
        messages = data.get("messages") or []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("role", "unknown")).lower()
            raw_content = msg.get("content", "")

            children: List[PayloadNode] = []
            text_blocks: List[str] = []

            if isinstance(raw_content, list):
                for b_idx, block in enumerate(raw_content):
                    if not isinstance(block, dict):
                        text_blocks.append(str(block))
                        continue

                    b_type = block.get("type", "text")
                    if b_type == "text":
                        text_blocks.append(block.get("text", ""))
                    elif b_type == "tool_use":
                        tu_name = block.get("name", "unknown")
                        tu_input = json.dumps(block.get("input", {}))
                        tu_text = f"{tu_name}({tu_input})"
                        tu_metrics = tokenizer.calculate_metrics(tu_text)

                        children.append(
                            PayloadNode(
                                id=f"msg_{idx + 1}_tool_use_{b_idx + 1}",
                                name=f"Tool Use: {tu_name}",
                                node_type=NodeType.TOOL_CALL,
                                role="tool_use",
                                content=tu_text,
                                metadata={"id": block.get("id"), "input": block.get("input")},
                                metrics=tu_metrics,
                                cost=CostMetric(),
                                children=[]
                            )
                        )
                    elif b_type == "tool_result":
                        res_content = str(block.get("content", ""))
                        res_metrics = tokenizer.calculate_metrics(res_content)

                        children.append(
                            PayloadNode(
                                id=f"msg_{idx + 1}_tool_res_{b_idx + 1}",
                                name=f"Tool Result [{block.get('tool_use_id', b_idx + 1)}]",
                                node_type=NodeType.TOOL_RESULT,
                                role="tool_result",
                                content=res_content,
                                metadata={"tool_use_id": block.get("tool_use_id")},
                                metrics=res_metrics,
                                cost=CostMetric(),
                                children=[]
                            )
                        )
                full_content = "\n".join(text_blocks)
            else:
                full_content = str(raw_content)

            if role == "user":
                node_type = NodeType.USER_MESSAGE
                name = f"User Message #{idx + 1}"
            elif role == "assistant":
                node_type = NodeType.ASSISTANT_MESSAGE
                name = f"Assistant Message #{idx + 1}"
            else:
                node_type = NodeType.CONVERSATION
                name = f"Message #{idx + 1} ({role})"

            metrics = tokenizer.calculate_metrics(full_content)
            nodes.append(
                PayloadNode(
                    id=f"msg_{idx + 1}",
                    name=name,
                    node_type=node_type,
                    role=role,
                    content=full_content,
                    metadata={"role": role},
                    metrics=metrics,
                    cost=CostMetric(),
                    children=children
                )
            )

        return nodes
