"""
Typer CLI application entrypoint for ctxflame.
Provides commands for profiling, generating interactive flamegraphs, diffing prompts,
evaluating CI/CD token budgets, and inspecting model pricing.
"""

import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from ctxflame import __version__
from ctxflame.analysis.profiler import profile_payload
from ctxflame.analysis.pricing import MODEL_REGISTRY
from ctxflame.visualizers.terminal import render_terminal_dashboard
from ctxflame.visualizers.html import generate_flamegraph_html
from ctxflame.visualizers.json_out import render_json, render_markdown_summary
from ctxflame.visualizers.diff import render_profile_diff

app = typer.Typer(
    name="ctxflame",
    help="Context Window Profiler and Token Flamegraph for LLM and Agent Operations.",
    add_completion=False,
    no_args_is_help=True
)

console = Console()


def _read_file_content(path: Path) -> str:
    """Read and return UTF-8 text from file path."""
    if not path.is_file():
        console.print(f"[bold red]Error:[/] File not found at '{path}'")
        raise typer.Exit(code=1)
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed reading file '{path}': {e}")
        raise typer.Exit(code=1)


@app.command("profile")
def profile_cmd(
    file: Path = typer.Argument(..., help="Path to payload file (.json, .txt, or .md)"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target model identifier"),
    tokenizer: Optional[str] = typer.Option(None, "--tokenizer", "-t", help="Tokenizer override"),
    json_out: bool = typer.Option(False, "--json", help="Output structured JSON instead of terminal UI"),
    markdown: bool = typer.Option(False, "--markdown", help="Output GitHub-flavored Markdown summary"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write result to output file"),
):
    """Profile an LLM payload and inspect token breakdown, costs, bloat, and attention risk."""
    content = _read_file_content(file)
    profile = profile_payload(content, model_name=model, tokenizer_override=tokenizer)

    if json_out:
        result = render_json(profile)
        if output:
            output.write_text(result, encoding="utf-8")
            console.print(f"[green]Saved JSON profile to '{output}'[/]")
        else:
            typer.echo(result)
        return

    if markdown:
        result = render_markdown_summary(profile)
        if output:
            output.write_text(result, encoding="utf-8")
            console.print(f"[green]Saved Markdown summary to '{output}'[/]")
        else:
            typer.echo(result)
        return

    render_terminal_dashboard(profile, console=console)

    if output:
        # Save JSON representation to output
        output.write_text(render_json(profile), encoding="utf-8")
        console.print(f"[green]Saved profile data to '{output}'[/]")


@app.command("flamegraph")
def flamegraph_cmd(
    file: Path = typer.Argument(..., help="Path to payload file (.json, .txt, or .md)"),
    output: Path = typer.Option(Path("flamegraph.html"), "--output", "-o", help="Destination HTML file path"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target model identifier"),
    open_browser: bool = typer.Option(False, "--open", help="Open generated HTML report in default browser"),
):
    """Generate a standalone, self-contained interactive HTML flamegraph."""
    content = _read_file_content(file)
    profile = profile_payload(content, model_name=model)

    html_content = generate_flamegraph_html(profile)
    try:
        output.write_text(html_content, encoding="utf-8")
        console.print(f"[bold green]Success:[/] Generated interactive flamegraph at [cyan]{output.resolve()}[/]")
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed writing HTML to '{output}': {e}")
        raise typer.Exit(code=1)

    if open_browser:
        webbrowser.open(output.resolve().as_uri())


@app.command("diff")
def diff_cmd(
    file_before: Path = typer.Argument(..., help="Baseline payload file (before changes)"),
    file_after: Path = typer.Argument(..., help="New payload file (after changes)"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target model identifier"),
):
    """Compare two prompt payload versions side-by-side to track token, cost, and attention deltas."""
    content_a = _read_file_content(file_before)
    content_b = _read_file_content(file_after)

    profile_a = profile_payload(content_a, model_name=model)
    profile_b = profile_payload(content_b, model_name=model)

    render_profile_diff(profile_a, profile_b, console=console)


@app.command("budget")
def budget_cmd(
    file: Path = typer.Argument(..., help="Path to payload file (.json, .txt, or .md)"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target model identifier"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum allowable token count"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Maximum allowable single-request cost in USD"),
    max_risk_score: Optional[float] = typer.Option(None, "--max-risk-score", help="Maximum allowable Lost-in-Middle risk score (0-100)"),
):
    """CI/CD gate: verify that payload satisfies token, cost, and attention constraints. Exits with 1 on failure."""
    content = _read_file_content(file)
    profile = profile_payload(content, model_name=model)

    violations = []

    if max_tokens is not None and profile.total_tokens > max_tokens:
        violations.append(
            f"Token limit exceeded: payload has {profile.total_tokens:,} tokens (max allowed: {max_tokens:,})"
        )

    if max_cost is not None and profile.total_cost_usd > max_cost:
        violations.append(
            f"Cost ceiling exceeded: payload costs ${profile.total_cost_usd:.6f} USD (max allowed: ${max_cost:.6f})"
        )

    if max_risk_score is not None and profile.lost_in_middle_score > max_risk_score:
        violations.append(
            f"Attention degradation risk exceeded: score is {profile.lost_in_middle_score:.1f} (max allowed: {max_risk_score:.1f})"
        )

    if violations:
        console.print("[bold red]Budget Check FAILED:[/] The following constraints were violated:")
        for v in violations:
            console.print(f"  - [red]{v}[/]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Budget Check PASSED:[/] {profile.total_tokens:,} tokens, ${profile.total_cost_usd:.6f} USD (satisfies all limits).")


@app.command("models")
def models_cmd():
    """List supported model specifications, context limits, and token pricing."""
    table = Table(title="Supported Models and Pricing Directory", border_style="dim")
    table.add_column("Model Identifier", style="bold cyan")
    table.add_column("Provider", style="white")
    table.add_column("Context Window Limit", justify="right", style="yellow")
    table.add_column("Input Cost (per 1M tokens)", justify="right", style="green")

    for key, spec in sorted(MODEL_REGISTRY.items(), key=lambda x: (x[1].provider, x[0])):
        table.add_row(
            spec.name,
            spec.provider,
            f"{spec.context_limit:,} tokens",
            f"${spec.input_cost_per_1m:.3f} USD"
        )

    console.print(table)


def version_callback(value: bool):
    if value:
        typer.echo(f"ctxflame version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show ctxflame version and exit", callback=version_callback, is_eager=True
    )
):
    """Context Window Profiler and Token Flamegraph CLI."""
    pass


if __name__ == "__main__":
    app()
