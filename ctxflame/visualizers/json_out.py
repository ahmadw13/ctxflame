"""
JSON and Markdown serializes for ctxflame.
Useful for CI/CD pipelines, automated PR comments, and stdout piping.
"""

import json
from ctxflame.models import ContextProfile


def render_json(profile: ContextProfile, indent: int = 2) -> str:
    """Render ContextProfile as structured JSON string."""
    return profile.model_dump_json(indent=indent)


def render_markdown_summary(profile: ContextProfile) -> str:
    """Render a GitHub-flavored Markdown summary table suitable for PR comments."""
    lines = [
        "### ctxflame Context Profile Summary",
        "",
        f"- **Model**: `{profile.model_name}` ({profile.tokenizer_name})",
        f"- **Total Tokens**: `{profile.total_tokens:,}` / `{profile.context_limit:,}` ({profile.utilization_percent:.2f}% context limit)",
        f"- **Estimated Cost**: `${profile.total_cost_usd:.6f}` USD (`${profile.cost_per_1m_usd:.2f}` / 1M calls)",
        f"- **Lost-in-Middle Attention Risk Score**: `{profile.lost_in_middle_score:.1f} / 100`",
        "",
        "| Section Type | Tokens | Share (%) | Estimated Cost |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for sec, count in sorted(profile.section_breakdown.items(), key=lambda x: x[1], reverse=True):
        pct = profile.section_percentages.get(sec, 0.0)
        cost = profile.total_cost_usd * (pct / 100.0)
        lines.append(f"| `{sec.upper()}` | {count:,} | {pct:.1f}% | ${cost:.6f} |")

    if profile.bloat_issues:
        lines.extend([
            "",
            "#### Detected Optimization Issues",
            "",
        ])
        for issue in profile.bloat_issues:
            lines.append(f"- **[{issue.severity.upper()}]** ({issue.category}): {issue.description}")
            if issue.estimated_wasted_tokens > 0:
                lines.append(f"  - *Estimated Waste*: ~{issue.estimated_wasted_tokens:,} tokens")
            lines.append(f"  - *Action*: {issue.suggestion}")

    return "\n".join(lines)
