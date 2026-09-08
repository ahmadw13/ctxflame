"""
Interactive HTML Flamegraph and Treemap generator for ctxflame.
Outputs a standalone, self-contained single-file HTML report (works completely offline).
"""

import json
from typing import Any, Dict
from ctxflame.models import ContextProfile, PayloadNode


def _node_to_dict(node: PayloadNode) -> Dict[str, Any]:
    """Convert a PayloadNode into a JSON-serializable hierarchy dictionary."""
    return {
        "id": node.id,
        "name": node.name,
        "type": node.node_type.value,
        "role": node.role or "",
        "content": node.content or "",
        "tokens": node.metrics.token_count,
        "chars": node.metrics.char_count,
        "words": node.metrics.word_count,
        "ratio": node.metrics.token_to_word_ratio,
        "cost": round(node.cost.input_cost_usd, 6),
        "children": [_node_to_dict(c) for c in node.children]
    }


def generate_flamegraph_html(profile: ContextProfile) -> str:
    """Generate interactive standalone flamegraph HTML."""
    root_data = {
        "name": f"Payload ({profile.total_tokens:,} tokens)",
        "type": "root",
        "role": "root",
        "content": "",
        "tokens": profile.total_tokens,
        "cost": round(profile.total_cost_usd, 6),
        "children": [_node_to_dict(n) for n in profile.nodes]
    }

    tree_json = json.dumps(root_data)
    summary_json = json.dumps({
        "model": profile.model_name,
        "tokenizer": profile.tokenizer_name,
        "total_tokens": profile.total_tokens,
        "context_limit": profile.context_limit,
        "utilization_pct": profile.utilization_percent,
        "total_cost": profile.total_cost_usd,
        "cost_1m": profile.cost_per_1m_usd,
        "lost_score": profile.lost_in_middle_score,
        "bloat_count": len(profile.bloat_issues)
    })
    bloat_json = json.dumps([b.model_dump() for b in profile.bloat_issues])
    breakdown_json = json.dumps(profile.section_breakdown)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ctxflame — {profile.model_name} Token Flamegraph</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #131b2e;
      --border: #202b42;
      --text: #e2e8f0;
      --text-muted: #8892b0;
      --system: #06b6d4;
      --tools: #d946ef;
      --user: #10b981;
      --assistant: #6366f1;
      --tool-call: #f59e0b;
      --doc: #0284c7;
      --other: #64748b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      background-color: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    h1 {{ font-size: 20px; font-weight: 700; letter-spacing: -0.5px; color: #fff; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      background: var(--card-bg);
      border: 1px solid var(--border);
    }}
    .grid-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .stat-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }}
    .stat-val {{ font-size: 22px; font-weight: 700; color: #fff; }}
    .stat-sub {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; }}
    .section-title {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; color: #fff; }}
    #flamegraph-container {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      overflow-x: auto;
    }}
    .flame-row {{
      display: flex;
      gap: 4px;
      margin-bottom: 8px;
      width: 100%;
      min-height: 42px;
    }}
    .flame-block {{
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 8px;
      border-radius: 5px;
      cursor: pointer;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      font-size: 11px;
      font-weight: 600;
      min-width: 55px;
      flex-shrink: 0;
      transition: opacity 0.15s, transform 0.15s, outline 0.15s;
    }}
    .flame-block:hover {{ opacity: 0.9; transform: scaleY(1.04); }}
    .flame-block.selected {{
      outline: 2px solid #38bdf8;
      box-shadow: 0 0 14px rgba(56, 189, 248, 0.4);
    }}
    .type-system {{ background: var(--system); color: #000; }}
    .type-tool_schema {{ background: var(--tools); color: #fff; }}
    .type-user_message {{ background: var(--user); color: #000; }}
    .type-assistant_message {{ background: var(--assistant); color: #fff; }}
    .type-tool_call {{ background: var(--tool-call); color: #000; }}
    .type-tool_result {{ background: #b45309; color: #fff; }}
    .type-document_chunk {{ background: var(--doc); color: #fff; }}
    .type-raw_prompt {{ background: #475569; color: #fff; }}
    .type-root {{ background: #1e293b; color: #cbd5e1; }}
    #tooltip {{
      position: fixed;
      display: none;
      background: #0f172a;
      border: 1px solid #38bdf8;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 12px;
      pointer-events: none;
      z-index: 1000;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    #inspector-panel {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      display: none;
    }}
    .inspector-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin: 14px 0;
      padding: 12px;
      background: #0b1120;
      border-radius: 6px;
      border: 1px solid var(--border);
    }}
    .inspector-item-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; }}
    .inspector-item-val {{ font-size: 15px; font-weight: 700; color: #fff; }}
    .inspector-content-box {{
      background: #070a12;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      font-family: ui-monospace, monospace;
      font-size: 12px;
      max-height: 240px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: #cbd5e1;
    }}
    .issues-list {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .issue-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-left-width: 4px;
      padding: 12px 16px;
      border-radius: 6px;
    }}
    .issue-critical {{ border-left-color: #ef4444; }}
    .issue-warning {{ border-left-color: #f59e0b; }}
    .issue-info {{ border-left-color: #3b82f6; }}
    .issue-header {{ display: flex; justify-content: space-between; font-weight: 600; font-size: 13px; margin-bottom: 4px; }}
    .issue-desc {{ font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }}
    .issue-sugg {{ font-size: 12px; color: #38bdf8; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>ctxflame Token Flamegraph</h1>
      <p style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Context Window Profiler for LLM & Agent Operations</p>
    </div>
    <div>
      <span class="badge" id="badge-model">{profile.model_name}</span>
      <span class="badge" style="margin-left: 8px;">{profile.tokenizer_name}</span>
    </div>
  </header>

  <div class="grid-stats">
    <div class="stat-card">
      <div class="stat-label">Total Tokens</div>
      <div class="stat-val">{profile.total_tokens:,}</div>
      <div class="stat-sub">{profile.utilization_percent:.2f}% of {profile.context_limit:,} context</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Estimated Input Cost</div>
      <div class="stat-val">${profile.total_cost_usd:.5f}</div>
      <div class="stat-sub">${profile.cost_per_1m_usd:.2f} per 1M requests</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Token/Word Ratio</div>
      <div class="stat-val">{profile.overall_token_to_word_ratio:.2f}</div>
      <div class="stat-sub">{profile.total_words:,} words / {profile.total_chars:,} chars</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Middle Attention Risk</div>
      <div class="stat-val">{profile.lost_in_middle_score:.1f} / 100</div>
      <div class="stat-sub">Lost in the Middle Risk Score</div>
    </div>
  </div>

  <div class="section-title">Context Window Token Flamegraph (Click any block to inspect full content)</div>
  <div id="flamegraph-container">
    <div id="flame-levels"></div>
  </div>

  <div id="inspector-panel">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <h3 id="inspector-title" style="font-size: 16px; color: #fff; font-weight: 700;"></h3>
      <span class="badge" id="inspector-badge"></span>
    </div>
    <div class="inspector-grid">
      <div>
        <div class="inspector-item-label">Tokens</div>
        <div class="inspector-item-val" id="insp-tokens"></div>
      </div>
      <div>
        <div class="inspector-item-label">Share</div>
        <div class="inspector-item-val" id="insp-share"></div>
      </div>
      <div>
        <div class="inspector-item-label">Estimated Cost</div>
        <div class="inspector-item-val" id="insp-cost"></div>
      </div>
      <div>
        <div class="inspector-item-label">Characters / Words</div>
        <div class="inspector-item-val" id="insp-chars"></div>
      </div>
    </div>
    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">Content Preview:</div>
    <div class="inspector-content-box" id="insp-content"></div>
  </div>

  <div class="section-title">Bloat & Optimization Diagnostics</div>
  <div class="issues-list" id="issues-container"></div>

  <div id="tooltip"></div>

  <script>
    const treeData = {tree_json};
    const bloatData = {bloat_json};
    const totalTokens = {profile.total_tokens} || 1;
    let selectedBlock = null;

    function getShortLabel(name, tokens, pct, containerWidth) {{
      const approxWidth = (pct / 100) * containerWidth;
      if (approxWidth >= 120) {{
        return `${{name}} (${{tokens.toLocaleString()}})`;
      }} else if (approxWidth >= 65) {{
        const concise = name.replace(/^System Instruction/, 'System')
                            .replace(/^User Turn #?/, 'User ')
                            .replace(/^User Message #?/, 'User ')
                            .replace(/^Assistant Message #?/, 'Assistant ')
                            .replace(/^Model Turn #?/, 'Model ')
                            .replace(/^Tool Schemas/, 'Tools')
                            .replace(/^Function Declarations/, 'Functions')
                            .replace(/^Tool: /, '')
                            .replace(/^Function: /, '')
                            .replace(/^Call: /, 'Call ');
        return `${{concise}} (${{tokens}})`;
      }} else {{
        return `${{tokens}} tok`;
      }}
    }}

    function renderFlamegraph(root) {{
      const container = document.getElementById("flame-levels");
      container.innerHTML = "";
      const containerWidth = container.offsetWidth || 1000;

      // Level 1: Top sections
      const row1 = document.createElement("div");
      row1.className = "flame-row";

      (root.children || []).forEach(child => {{
        const widthPct = Math.max(1.0, (child.tokens / totalTokens) * 100);
        const block = document.createElement("div");
        block.className = `flame-block type-${{child.type}}`;
        block.style.width = `${{widthPct}}%`;
        block.textContent = getShortLabel(child.name, child.tokens, widthPct, containerWidth);

        block.addEventListener("mousemove", (e) => showTooltip(e, child, widthPct));
        block.addEventListener("mouseleave", hideTooltip);
        block.addEventListener("click", () => inspectNode(block, child, widthPct));
        row1.appendChild(block);
      }});

      container.appendChild(row1);

      // Level 2: Children breakdown if present
      const hasLevel2 = (root.children || []).some(c => c.children && c.children.length > 0);
      if (hasLevel2) {{
        const row2 = document.createElement("div");
        row2.className = "flame-row";
        (root.children || []).forEach(parent => {{
          const pWidth = Math.max(1.0, (parent.tokens / totalTokens) * 100);
          if (!parent.children || parent.children.length === 0) {{
            const empty = document.createElement("div");
            empty.className = `flame-block type-${{parent.type}}`;
            empty.style.width = `${{pWidth}}%`;
            empty.style.opacity = "0.4";
            empty.textContent = getShortLabel(parent.name, parent.tokens, pWidth, containerWidth);
            empty.addEventListener("click", () => inspectNode(empty, parent, pWidth));
            row2.appendChild(empty);
          }} else {{
            parent.children.forEach(sub => {{
              const sWidth = Math.max(1.0, (sub.tokens / totalTokens) * 100);
              const subBlock = document.createElement("div");
              subBlock.className = `flame-block type-${{sub.type}}`;
              subBlock.style.width = `${{sWidth}}%`;
              subBlock.textContent = getShortLabel(sub.name, sub.tokens, sWidth, containerWidth);
              subBlock.addEventListener("mousemove", (e) => showTooltip(e, sub, sWidth));
              subBlock.addEventListener("mouseleave", hideTooltip);
              subBlock.addEventListener("click", () => inspectNode(subBlock, sub, sWidth));
              row2.appendChild(subBlock);
            }});
          }}
        }});
        container.appendChild(row2);
      }}
    }}

    function inspectNode(domElement, node, pct) {{
      if (selectedBlock) {{
        selectedBlock.classList.remove("selected");
      }}
      domElement.classList.add("selected");
      selectedBlock = domElement;

      const panel = document.getElementById("inspector-panel");
      panel.style.display = "block";
      document.getElementById("inspector-title").textContent = node.name;
      document.getElementById("inspector-badge").textContent = node.type.toUpperCase();
      document.getElementById("insp-tokens").textContent = `${{node.tokens.toLocaleString()}} tokens`;
      document.getElementById("insp-share").textContent = `${{pct.toFixed(1)}}% of payload`;
      document.getElementById("insp-cost").textContent = `$${{node.cost.toFixed(6)}} USD`;
      document.getElementById("insp-chars").textContent = `${{node.chars.toLocaleString()}} chars / ${{node.words.toLocaleString()}} words`;
      document.getElementById("insp-content").textContent = node.content || "(No direct text content; inspect nested sub-calls)";

      panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}

    const tooltip = document.getElementById("tooltip");
    function showTooltip(e, node, pct) {{
      tooltip.style.display = "block";
      tooltip.style.left = `${{e.clientX + 14}}px`;
      tooltip.style.top = `${{e.clientY + 14}}px`;
      tooltip.innerHTML = `
        <div style="font-weight: 700; color: #fff; margin-bottom: 4px;">${{node.name}}</div>
        <div>Type: <b>${{node.type}}</b></div>
        <div>Tokens: <b>${{node.tokens.toLocaleString()}}</b> (${{pct.toFixed(1)}}%)</div>
        <div>Estimated Cost: <b>$${{node.cost.toFixed(6)}}</b></div>
        <div style="font-size: 10px; color: #38bdf8; margin-top: 4px;">Click to inspect full content</div>
      `;
    }}
    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function renderIssues() {{
      const container = document.getElementById("issues-container");
      if (!bloatData || bloatData.length === 0) {{
        container.innerHTML = '<div class="issue-card issue-info"><div class="issue-header">No Bloat Detected</div><div class="issue-desc">All prompt payload sections are well-proportioned.</div></div>';
        return;
      }}
      container.innerHTML = bloatData.map(item => `
        <div class="issue-card issue-${{item.severity}}">
          <div class="issue-header">
            <span>[${{item.severity.toUpperCase()}}] ${{item.category}}</span>
            ${{item.estimated_wasted_tokens > 0 ? `<span style="color: #f59e0b;">~${{item.estimated_wasted_tokens.toLocaleString()}} tokens wasted</span>` : ''}}
          </div>
          <div class="issue-desc">${{item.description}}</div>
          <div class="issue-sugg">Recommendation: ${{item.suggestion}}</div>
        </div>
      `).join("");
    }}

    window.addEventListener("resize", () => renderFlamegraph(treeData));
    renderFlamegraph(treeData);
    renderIssues();
  </script>
</body>
</html>
"""
    return html_template
