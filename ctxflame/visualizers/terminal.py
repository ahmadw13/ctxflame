"""
Rich terminal dashboard renderer for ctxflame.
Renders summary panels, breakdown tables, hierarchical trees, bloat diagnostics,
and positional attention risk heatmaps (zero emojis).
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from ctxflame.models import ContextProfile, PayloadNode, NodeType


def _render_bar(percentage: float, width: int = 24) -> str:
    """Render an ASCII/Unicode progress bar."""
    filled = int(round((percentage / 100.0) * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{'=' * filled}{'-' * empty}] {percentage:.1f}%"


def _add_node_to_tree(tree: Tree, node: PayloadNode):
    """Recursively add a PayloadNode to a Rich Tree."""
    type_color = {
        NodeType.SYSTEM: "bold cyan",
        NodeType.TOOL_SCHEMA: "bold magenta",
        NodeType.USER_MESSAGE: "bold green",
        NodeType.ASSISTANT_MESSAGE: "bold blue",
        NodeType.TOOL_CALL: "yellow",
        NodeType.TOOL_RESULT: "dim yellow",
        NodeType.DOCUMENT_CHUNK: "cyan",
        NodeType.RAW_PROMPT: "white",
        NodeType.UNKNOWN: "dim white",
    }.get(node.node_type, "white")

    label = Text()
    label.append(f"[{node.node_type.value.upper()}] ", style=type_color)
    label.append(f"{node.name} ", style="bold")
    label.append(f"({node.metrics.token_count:,} tokens, ${node.cost.input_cost_usd:.5f})", style="dim")

    sub_tree = tree.add(label)
    for child in node.children:
        _add_node_to_tree(sub_tree, child)


def render_terminal_dashboard(profile: ContextProfile, console: Optional[Console] = None):
    """Render the full ctxflame terminal profiling dashboard."""
    if console is None:
        console = Console()

    # 1. Header & Summary Card
    header_text = Text()
    header_text.append("Model: ", style="bold white")
    header_text.append(f"{profile.model_name} ", style="bold cyan")
    header_text.append(" | Tokenizer: ", style="bold white")
    header_text.append(f"{profile.tokenizer_name}\n", style="dim cyan")

    header_text.append("Total Tokens: ", style="bold white")
    header_text.append(f"{profile.total_tokens:,} ", style="bold yellow")
    header_text.append(f"/ {profile.context_limit:,} limit ", style="dim")
    header_text.append(f"({profile.utilization_percent:.2f}% utilization)\n", style="bold green" if profile.utilization_percent < 50 else "bold red")

    header_text.append("Context Bar:  ", style="bold white")
    header_text.append(f"{_render_bar(profile.utilization_percent, width=30)}\n")

    header_text.append("Text Metrics: ", style="bold white")
    header_text.append(f"{profile.total_chars:,} chars, {profile.total_words:,} words (ratio: {profile.overall_token_to_word_ratio:.2f} tok/word)\n", style="dim")

    header_text.append("Estimated Cost: ", style="bold white")
    header_text.append(f"${profile.total_cost_usd:.6f} USD ", style="bold green")
    header_text.append(f"(${profile.cost_per_1m_usd:.2f} / 1M calls)", style="dim green")

    console.print(Panel(header_text, title="ctxflame — Context Window Profile", border_style="cyan"))

    # 2. Section Breakdown Table
    table = Table(title="Payload Section Breakdown", border_style="dim", show_lines=True)
    table.add_column("Section Type", style="bold", min_width=18)
    table.add_column("Tokens", justify="right", style="yellow")
    table.add_column("Share (%)", justify="left")
    table.add_column("Cost (USD)", justify="right", style="green")

    for sec_type, tok_count in sorted(profile.section_breakdown.items(), key=lambda x: x[1], reverse=True):
        pct = profile.section_percentages.get(sec_type, 0.0)
        bar = _render_bar(pct, width=16)
        cost_val = (tok_count / 1_000_000.0) * (profile.cost_per_1m_usd / max(1, profile.total_tokens) * 1_000_000.0) if profile.total_tokens > 0 else 0.0
        table.add_row(sec_type.upper(), f"{tok_count:,}", bar, f"${profile.total_cost_usd * (pct / 100.0):.6f}")

    console.print(table)

    # 3. Positional Attention & "Lost in the Middle" Risk Map
    att_table = Table(title="Positional Attention & Middle-Zone Risk Map", border_style="dim")
    att_table.add_column("Section / Node", style="bold")
    att_table.add_column("Position Range", justify="center")
    att_table.add_column("Zone", justify="center")
    att_table.add_column("Risk", justify="center")
    att_table.add_column("Diagnostic Notes", justify="left", style="dim")

    for pos in profile.attention_analysis:
        pct_start = int(pos.normalized_start * 100)
        pct_end = int(pos.normalized_end * 100)
        pos_str = f"{pct_start}% - {pct_end}% ({pos.token_start:,} - {pos.token_end:,})"

        mid = (pos.normalized_start + pos.normalized_end) / 2.0
        if mid < 0.25:
            zone = "Primacy (Start)"
        elif mid > 0.75:
            zone = "Recency (End)"
        else:
            zone = "Middle (Danger)"

        if pos.attention_risk == "high":
            risk_badge = "[bold white on red] HIGH [/]"
        elif pos.attention_risk == "medium":
            risk_badge = "[bold black on yellow] MED [/]"
        else:
            risk_badge = "[bold black on green] LOW [/]"

        att_table.add_row(pos.section_name, pos_str, zone, risk_badge, pos.reason)

    console.print(att_table)

    score_color = "bold green" if profile.lost_in_middle_score < 30 else ("bold yellow" if profile.lost_in_middle_score < 60 else "bold red")
    console.print(f"Overall Middle-Zone Attention Risk Score: [{score_color}]{profile.lost_in_middle_score:.1f} / 100.0[/]\n")

    # 4. Bloat & Redundancy Diagnostics
    if profile.bloat_issues:
        bloat_panel_text = Text()
        for idx, issue in enumerate(profile.bloat_issues, 1):
            sev_color = "bold red" if issue.severity == "critical" else ("bold yellow" if issue.severity == "warning" else "bold blue")
            bloat_panel_text.append(f"[{idx}] ", style="bold white")
            bloat_panel_text.append(f"[{issue.severity.upper()}] ", style=sev_color)
            bloat_panel_text.append(f"({issue.category}): ", style="bold white")
            bloat_panel_text.append(f"{issue.description}\n", style="white")
            if issue.estimated_wasted_tokens > 0:
                bloat_panel_text.append(f"    Estimated Waste: ~{issue.estimated_wasted_tokens:,} tokens\n", style="dim yellow")
            bloat_panel_text.append(f"    Action: {issue.suggestion}\n\n", style="cyan")

        bloat_panel_text.rstrip()
        console.print(Panel(bloat_panel_text, title="Optimization & Bloat Diagnostics", border_style="yellow"))
    else:
        console.print(Panel("No major token bloat or redundancy detected. Payload structure is efficient.", title="Optimization Diagnostics", border_style="green"))

    # 5. Payload Hierarchy Tree
    root_tree = Tree(f"[bold cyan]Payload Hierarchy Tree[/] ({profile.total_tokens:,} total tokens)")
    for node in profile.nodes:
        _add_node_to_tree(root_tree, node)
    console.print(root_tree)
    console.print()
