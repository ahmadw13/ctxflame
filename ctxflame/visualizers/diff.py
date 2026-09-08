"""
Diff engine and visualizer for comparing two prompt payloads in ctxflame.
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from ctxflame.models import ContextProfile


def render_profile_diff(
    profile_before: ContextProfile,
    profile_after: ContextProfile,
    console: Optional[Console] = None
):
    """Render a visual side-by-side diff comparing two ContextProfile instances."""
    if console is None:
        console = Console()

    tok_delta = profile_after.total_tokens - profile_before.total_tokens
    cost_delta = profile_after.total_cost_usd - profile_before.total_cost_usd
    score_delta = profile_after.lost_in_middle_score - profile_before.lost_in_middle_score

    # Delta badge
    tok_sign = "+" if tok_delta >= 0 else ""
    cost_sign = "+" if cost_delta >= 0 else ""
    score_sign = "+" if score_delta >= 0 else ""

    summary = Text()
    summary.append("Target Model: ", style="bold")
    summary.append(f"{profile_after.model_name}\n", style="cyan")

    summary.append("Token Delta:  ", style="bold")
    tok_style = "bold red" if tok_delta > 0 else ("bold green" if tok_delta < 0 else "bold white")
    summary.append(f"{tok_sign}{tok_delta:,} tokens ", style=tok_style)
    summary.append(f"({profile_before.total_tokens:,} -> {profile_after.total_tokens:,})\n", style="dim")

    summary.append("Cost Delta:   ", style="bold")
    cost_style = "bold red" if cost_delta > 0 else ("bold green" if cost_delta < 0 else "bold white")
    summary.append(f"{cost_sign}${cost_delta:.6f} USD / call ", style=cost_style)
    summary.append(f"({cost_sign}${cost_delta * 1_000_000:.2f} / 1M calls)\n", style="dim")

    summary.append("Attention Risk: ", style="bold")
    att_style = "bold red" if score_delta > 0 else ("bold green" if score_delta < 0 else "bold white")
    summary.append(f"{score_sign}{score_delta:.1f} pts ", style=att_style)
    summary.append(f"({profile_before.lost_in_middle_score:.1f} -> {profile_after.lost_in_middle_score:.1f})", style="dim")

    console.print(Panel(summary, title="ctxflame — Prompt Payload Diff", border_style="cyan"))

    # Section-by-section comparison table
    all_sections = sorted(
        set(profile_before.section_breakdown.keys()).union(set(profile_after.section_breakdown.keys()))
    )

    table = Table(title="Section Token Changes", border_style="dim")
    table.add_column("Section Type", style="bold")
    table.add_column("Before", justify="right", style="dim")
    table.add_column("After", justify="right", style="bold")
    table.add_column("Delta", justify="right")

    for sec in all_sections:
        c_before = profile_before.section_breakdown.get(sec, 0)
        c_after = profile_after.section_breakdown.get(sec, 0)
        diff = c_after - c_before
        diff_str = f"{'+' if diff >= 0 else ''}{diff:,}"

        if diff > 0:
            diff_styled = f"[red]{diff_str}[/]"
        elif diff < 0:
            diff_styled = f"[green]{diff_str}[/]"
        else:
            diff_styled = "[dim]0[/]"

        table.add_row(sec.upper(), f"{c_before:,}", f"{c_after:,}", diff_styled)

    console.print(table)
    console.print()
