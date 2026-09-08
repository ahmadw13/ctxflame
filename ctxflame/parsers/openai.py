"""
OpenAI format parser for ctxflame.
Parses Chat Completion payloads: messages, functions, and tools.
"""

import json
from typing import Any, Dict, List
from ctxflame.models import CostMetric, NodeType, PayloadNode
from ctxflame.parsers.base import BaseParser
from ctxflame.tokenizers.base import BaseTokenizer


class OpenAIParser(BaseParser):
    """Parses OpenAI chat completion requests."""

    @property
    def format_name(self) -> str:
        return "OpenAI Chat Completion Format"

    def can_parse(self, data: Any) -> bool:
        if isinstance(data, dict):
            # Exclude Anthropic payloads
            if "system" in data and isinstance(data["system"], (str, list)):
                return False
            tools = data.get("tools")
            if tools and isinstance(tools, list):
                if any(isinstance(t, dict) and "input_schema" in t for t in tools):
                    return False
            # Exclude Gemini payloads
            if "systemInstruction" in data or "contents" in data:
                return False
            return "messages" in data or "tools" in data or "functions" in data
        return False

    def parse(self, data: Any, tokenizer: BaseTokenizer) -> List[PayloadNode]:
        nodes: List[PayloadNode] = []

        # 1. Tools / Function Schemas
        tools = data.get("tools") or []
        functions = data.get("functions") or []

        if tools or functions:
            tool_children: List[PayloadNode] = []
            combined_tools = tools if tools else [{"type": "function", "function": f} for f in functions]

            for idx, tool in enumerate(combined_tools):
                fn = tool.get("function", {}) if isinstance(tool, dict) else {}
                fn_name = fn.get("name", f"tool_{idx + 1}")
                fn_content = json.dumps(tool, indent=2)
                metrics = tokenizer.calculate_metrics(fn_content)

                tool_node = PayloadNode(
                    id=f"tool_schema_{idx + 1}",
                    name=f"Tool: {fn_name}",
                    node_type=NodeType.TOOL_SCHEMA,
                    content=fn_content,
                    metadata={"tool_name": fn_name, "raw": tool},
                    metrics=metrics,
                    cost=CostMetric(),
                    children=[]
                )
                tool_children.append(tool_node)

            total_tool_text = "\n".join(c.content or "" for c in tool_children)
            tool_parent_metrics = tokenizer.calculate_metrics(total_tool_text)
            nodes.append(
                PayloadNode(
                    id="tool_schemas_root",
                    name=f"Tool Schemas ({len(tool_children)} defined)",
                    node_type=NodeType.TOOL_SCHEMA,
                    content=total_tool_text,
                    metadata={"count": len(tool_children)},
                    metrics=tool_parent_metrics,
                    cost=CostMetric(),
                    children=tool_children
                )
            )

        # 2. Messages
        messages = data.get("messages") or []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("role", "unknown")).lower()
            raw_content = msg.get("content", "")
            
            # Content can be a string or a list of parts (e.g. text/image)
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content_str = "\n".join(text_parts)
            elif isinstance(raw_content, str):
                content_str = raw_content
            else:
                content_str = json.dumps(raw_content) if raw_content else ""

            # Determine node type based on role
            if role == "system":
                node_type = NodeType.SYSTEM
                name = f"System Message #{idx + 1}"
            elif role == "user":
                node_type = NodeType.USER_MESSAGE
                name = f"User Message #{idx + 1}"
            elif role == "assistant":
                node_type = NodeType.ASSISTANT_MESSAGE
                name = f"Assistant Message #{idx + 1}"
            elif role in ("tool", "function"):
                node_type = NodeType.TOOL_RESULT
                tool_id = msg.get("tool_call_id") or msg.get("name") or str(idx + 1)
                name = f"Tool Result [{tool_id}]"
            else:
                node_type = NodeType.CONVERSATION
                name = f"Message #{idx + 1} ({role})"

            sub_children: List[PayloadNode] = []

            # Check if assistant called tools
            tool_calls = msg.get("tool_calls") or []
            if tool_calls and isinstance(tool_calls, list):
                for tc_idx, tc in enumerate(tool_calls):
                    tc_fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    tc_name = tc_fn.get("name", "unknown_function")
                    tc_args = tc_fn.get("arguments", "")
                    tc_content = f"{tc_name}({tc_args})"
                    tc_metrics = tokenizer.calculate_metrics(tc_content)

                    sub_children.append(
                        PayloadNode(
                            id=f"msg_{idx + 1}_tc_{tc_idx + 1}",
                            name=f"Call: {tc_name}",
                            node_type=NodeType.TOOL_CALL,
                            role="tool_call",
                            content=tc_content,
                            metadata={"function": tc_name, "arguments": tc_args},
                            metrics=tc_metrics,
                            cost=CostMetric(),
                            children=[]
                        )
                    )

            full_text = content_str
            if sub_children:
                full_text = f"{content_str}\n" + "\n".join(c.content or "" for c in sub_children)

            metrics = tokenizer.calculate_metrics(full_text)

            nodes.append(
                PayloadNode(
                    id=f"msg_{idx + 1}",
                    name=name,
                    node_type=node_type,
                    role=role,
                    content=full_text,
                    metadata={"index": idx, "role": role},
                    metrics=metrics,
                    cost=CostMetric(),
                    children=sub_children
                )
            )

        return nodes
