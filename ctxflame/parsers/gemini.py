"""
Google Gemini API format parser for ctxflame.
Parses systemInstruction, contents (user/model turns, functionCalls), and tool declarations.
"""

import json
from typing import Any, List
from ctxflame.models import CostMetric, NodeType, PayloadNode
from ctxflame.parsers.base import BaseParser
from ctxflame.tokenizers.base import BaseTokenizer


class GeminiParser(BaseParser):
    """Parses Google Gemini generateContent request payloads."""

    @property
    def format_name(self) -> str:
        return "Google Gemini Format"

    def can_parse(self, data: Any) -> bool:
        if isinstance(data, dict):
            return "contents" in data or "systemInstruction" in data
        return False

    def parse(self, data: Any, tokenizer: BaseTokenizer) -> List[PayloadNode]:
        nodes: List[PayloadNode] = []

        # 1. System Instruction
        sys_inst = data.get("systemInstruction") or data.get("system_instruction")
        if sys_inst:
            sys_text = ""
            if isinstance(sys_inst, dict):
                parts = sys_inst.get("parts", [])
                sys_text = "\n".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in parts
                )
            elif isinstance(sys_inst, str):
                sys_text = sys_inst

            if sys_text:
                metrics = tokenizer.calculate_metrics(sys_text)
                nodes.append(
                    PayloadNode(
                        id="system_instruction",
                        name="System Instruction",
                        node_type=NodeType.SYSTEM,
                        role="system",
                        content=sys_text,
                        metadata={},
                        metrics=metrics,
                        cost=CostMetric(),
                        children=[]
                    )
                )

        # 2. Tools (functionDeclarations)
        tools = data.get("tools") or []
        if tools:
            tool_children: List[PayloadNode] = []
            for t_idx, tool_group in enumerate(tools):
                declarations = tool_group.get("functionDeclarations", []) if isinstance(tool_group, dict) else []
                for d_idx, decl in enumerate(declarations):
                    d_name = decl.get("name", f"function_{d_idx + 1}")
                    d_content = json.dumps(decl, indent=2)
                    d_metrics = tokenizer.calculate_metrics(d_content)

                    tool_children.append(
                        PayloadNode(
                            id=f"tool_schema_{t_idx + 1}_{d_idx + 1}",
                            name=f"Function: {d_name}",
                            node_type=NodeType.TOOL_SCHEMA,
                            content=d_content,
                            metadata={"name": d_name},
                            metrics=d_metrics,
                            cost=CostMetric(),
                            children=[]
                        )
                    )

            if tool_children:
                total_text = "\n".join(c.content or "" for c in tool_children)
                p_metrics = tokenizer.calculate_metrics(total_text)
                nodes.append(
                    PayloadNode(
                        id="tools_root",
                        name=f"Function Declarations ({len(tool_children)} defined)",
                        node_type=NodeType.TOOL_SCHEMA,
                        content=total_text,
                        metadata={"count": len(tool_children)},
                        metrics=p_metrics,
                        cost=CostMetric(),
                        children=tool_children
                    )
                )

        # 3. Contents (Turns)
        contents = data.get("contents") or []
        for idx, turn in enumerate(contents):
            if not isinstance(turn, dict):
                continue

            role = str(turn.get("role", "user")).lower()
            parts = turn.get("parts", [])

            children: List[PayloadNode] = []
            text_lines: List[str] = []

            for p_idx, part in enumerate(parts):
                if not isinstance(part, dict):
                    text_lines.append(str(part))
                    continue

                if "text" in part:
                    text_lines.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    fc_name = fc.get("name", "unknown")
                    fc_args = json.dumps(fc.get("args", {}))
                    fc_str = f"{fc_name}({fc_args})"
                    fc_metrics = tokenizer.calculate_metrics(fc_str)

                    children.append(
                        PayloadNode(
                            id=f"turn_{idx + 1}_fc_{p_idx + 1}",
                            name=f"Call: {fc_name}",
                            node_type=NodeType.TOOL_CALL,
                            role="model",
                            content=fc_str,
                            metadata={"name": fc_name, "args": fc.get("args")},
                            metrics=fc_metrics,
                            cost=CostMetric(),
                            children=[]
                        )
                    )
                elif "functionResponse" in part:
                    fr = part["functionResponse"]
                    fr_name = fr.get("name", "response")
                    fr_res = json.dumps(fr.get("response", {}))
                    fr_metrics = tokenizer.calculate_metrics(fr_res)

                    children.append(
                        PayloadNode(
                            id=f"turn_{idx + 1}_fr_{p_idx + 1}",
                            name=f"Result: {fr_name}",
                            node_type=NodeType.TOOL_RESULT,
                            role="user",
                            content=fr_res,
                            metadata={"name": fr_name},
                            metrics=fr_metrics,
                            cost=CostMetric(),
                            children=[]
                        )
                    )
            full_text = "\n".join(text_lines)
            if children:
                children_text = "\n".join(c.content or "" for c in children)
                full_text = f"{full_text}\n{children_text}".strip() if full_text else children_text
            if role == "user":
                node_type = NodeType.USER_MESSAGE
                name = f"User Turn #{idx + 1}"
            elif role == "model":
                node_type = NodeType.ASSISTANT_MESSAGE
                name = f"Model Turn #{idx + 1}"
            else:
                node_type = NodeType.CONVERSATION
                name = f"Turn #{idx + 1} ({role})"

            metrics = tokenizer.calculate_metrics(full_text)
            nodes.append(
                PayloadNode(
                    id=f"turn_{idx + 1}",
                    name=name,
                    node_type=node_type,
                    role=role,
                    content=full_text,
                    metadata={"role": role},
                    metrics=metrics,
                    cost=CostMetric(),
                    children=children
                )
            )

        return nodes
