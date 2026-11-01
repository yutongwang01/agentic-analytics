"""
analytics_agent.py
Iterative analytics agent with persistent execution namespace.

Design:
  - Receives only dataset file paths + user question (no pre-supplied schema)
  - Autonomously inspects datasets, reasons, generates & executes code
  - Maintains a persistent namespace so variables survive across iterations
  - Decides when analysis is complete (done=True) or continues
  - Returns structured output: final_answer + Plotly JSON plots
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 8          # hard stop to prevent infinite loops
MAX_RETRIES_ON_ERROR = 2    # retries within a single iteration on exec error


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationTrace:
    """Records what happened in one reasoning/execution iteration."""
    iteration: int
    reasoning: str
    code: str
    execution_output: str   # string repr of result or error
    error: Optional[str]
    done: bool


@dataclass
class AnalyticsResult:
    """Final structured output of the analytics agent."""
    final_answer: str
    plots: list[dict]       # list of {"title": str, "plotly_json": str}
    traces: list[IterationTrace]
    total_iterations: int
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Persistent namespace executor
# ---------------------------------------------------------------------------

class PersistentExecutor:
    """
    Executes Python code inside a single long-lived namespace dict.
    Variables, DataFrames, and imports survive across iterations.
    """

    def __init__(self, csv_paths: list[str]):
        self.namespace: dict[str, Any] = {
            "pd": pd,
            "go": go,
            "px": px,
            "json": json,
            # Pre-populate csv_paths so the agent can reference them by name
            "csv_paths": csv_paths,
        }
        # Also expose each path individually as csv_path_0, csv_path_1, ...
        for i, p in enumerate(csv_paths):
            self.namespace[f"csv_path_{i}"] = p
        # Convenience: first path is just csv_path
        if csv_paths:
            self.namespace["csv_path"] = csv_paths[0]

    def execute(self, code: str) -> tuple[str, Optional[str]]:
        """
        Run code in the persistent namespace.
        Returns (output_repr, error_message).
        output_repr is a string summary of `result` if set, else last expression.
        """
        try:
            exec(code, self.namespace)  # noqa: S102
        except Exception:
            error_msg = traceback.format_exc(limit=5)
            return "", error_msg
        
        # defrag all DataFrames
        for k, v in list(self.namespace.items()):
            if isinstance(v, pd.DataFrame):
                self.namespace[k] = v.copy()

        raw = self.namespace.get("result")
        return _repr_result(raw), None

    def get_result(self) -> Any:
        """Return the current value of `result` in the namespace."""
        return self.namespace.get("result")

    def snapshot_summary(self) -> str:
        """
        Return a short summary of meaningful variables currently in the namespace
        (excludes builtins, modules, and the pre-loaded constants).
        """
        skip = {"pd", "go", "px", "json", "csv_paths", "csv_path",
                "__builtins__", "__name__", "__doc__"}
        lines = []
        for k, v in self.namespace.items():
            if k.startswith("_") or k in skip or k.startswith("csv_path_"):
                continue
            if isinstance(v, pd.DataFrame):
                lines.append(f"  {k}: DataFrame{list(v.shape)} cols={list(v.columns)}")
            elif isinstance(v, pd.Series):
                lines.append(f"  {k}: Series len={len(v)}")
            elif isinstance(v, (int, float, str, bool)):
                lines.append(f"  {k} = {v!r}")
            else:
                lines.append(f"  {k}: {type(v).__name__}")
        return "\n".join(lines) if lines else "  (empty)"


def _repr_result(raw: Any, max_rows: int = 10) -> str:
    """Compact string representation of a result value for LLM consumption."""
    if raw is None:
        return "(no result variable set)"
    if isinstance(raw, pd.DataFrame):
        return (
            f"DataFrame shape={raw.shape} columns={list(raw.columns)}\n"
            f"{raw.head(max_rows).to_string()}"
        )
    if isinstance(raw, pd.Series):
        return f"Series len={len(raw)} name={raw.name}\n{raw.head(max_rows).to_string()}"
    if isinstance(raw, dict):
        return f"dict keys={list(raw.keys())}\n{json.dumps({k: str(v)[:120] for k, v in raw.items()}, indent=2)}"
    return str(raw)[:800]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dataset column reference builder
# ---------------------------------------------------------------------------

# Financial concepts → candidate column names in order of preference.
# The builder tries each candidate in order and uses the first one found.
_FINANCIAL_CONCEPTS: list[tuple[str, list[str]]] = [
    ("Ticker / company ID", ["Unnamed: 0"]),
    ("Sector",              ["Sector"]),
    ("P/E ratio",           ["PE ratio", "priceEarningsRatio"]),
    ("ROE",                 ["ROE", "returnOnEquity"]),
    ("P/B ratio",           ["PB ratio", "priceToBookRatio", "priceBookValueRatio"]),
    ("P/S ratio",           ["Price to Sales Ratio", "priceToSalesRatio", "priceSalesRatio"]),
    ("Gross Margin",        ["Gross Margin", "grossProfitMargin"]),
    ("Net Profit Margin",   ["Net Profit Margin", "netProfitMargin", "Profit Margin"]),
    ("Operating Margin",    ["EBIT Margin", "operatingProfitMargin"]),
    ("Debt Ratio",          ["debtRatio", "Debt to Assets"]),
    ("D/E Ratio",           ["debtEquityRatio", "Debt to Equity"]),
    ("Current Ratio",       ["Current ratio", "currentRatio"]),
    ("Market Cap",          ["Market Cap"]),
    ("Revenue",             ["Revenue"]),
    ("Net Income",          ["Net Income"]),
    ("EPS",                 ["EPS", "EPS Diluted"]),
    ("ROIC",                ["ROIC", "returnOnCapitalEmployed"]),
    ("Enterprise Value",    ["Enterprise Value"]),
    ("Free Cash Flow",      ["Free Cash Flow"]),
]


def _infer_unit(series: "pd.Series") -> str:
    """
    Infer the unit of a numeric column from its value distribution.

    Rule of thumb for this financial dataset:
    - If the median absolute value is < 2 and max < 25 → decimal ratio
      (0.15 = 15%).  ROE, margins, debtRatio all fall here.
    - If values cluster in single/double digits (1–200) → already a ratio
      or multiple (PE ratio, PB ratio).
    - Otherwise → absolute dollar / share count value.
    """
    import numpy as np
    s = series.dropna()
    if len(s) == 0:
        return "unknown"
    abs_med = float(np.abs(s).median())
    abs_max = float(np.abs(s).max())
    if abs_max <= 25 and abs_med <= 2:
        return "decimal (0.15 = 15%) — use threshold 0.15, NOT 15"
    elif abs_max <= 500 and abs_med <= 50:
        return "ratio/multiple (e.g. 15.0 = 15x or 15%)"
    else:
        return "absolute value (dollars, counts, etc.)"


def build_dataset_reference(csv_paths: list[str]) -> str:
    """
    Read the first available CSV and produce a compact column-reference block.

    This is injected into both the analytics agent and the validator planner
    so neither has to guess column names or value units.

    Returns a multi-line string ready to be appended to a system/user prompt.
    """
    if not csv_paths:
        return ""
    try:
        df = pd.read_csv(csv_paths[0], nrows=200)
    except Exception:
        return ""

    lines: list[str] = [
        "",
        "DATASET COLUMN REFERENCE",
        "Use the EXACT column names listed below. Do NOT paraphrase or abbreviate.",
        "Check the unit description before applying numeric thresholds.",
        "",
        f"{'Concept':<25} {'Exact column name':<32} Unit / scale",
        "-" * 80,
    ]
    for concept, candidates in _FINANCIAL_CONCEPTS:
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            continue
        if df[found].dtype.kind in ("f", "i"):
            unit = _infer_unit(df[found])
        else:
            unit = "categorical / string"
        lines.append(f"  {concept:<23} '{found}'{'':>{32 - len(found)}} {unit}")

    lines += [
        "",
        "Notes:",
        "- 'ROE' and most margin columns are DECIMAL (0.15 = 15%).",
        "  When filtering e.g. ROE > 15%, write: ROE > 0.15",
        "- 'PE ratio' is already a ratio/multiple (15.0 = 15x).",
        "- Ticker symbols are in column 'Unnamed: 0'.",
        "- For year-over-year analysis, each CSV file is one year;",
        "  check the filename (e.g. 2014_Financial_Data.csv) for the year.",
        "",
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an autonomous data analytics agent.
You have access to a persistent Python execution environment.
Variables you define in one iteration remain available in the next.

Your goal: answer the user's question by iteratively inspecting data,
generating analysis code, observing results, and deciding whether more
analysis is needed.

In EVERY response you must output a single JSON object — nothing else.
Schema:
{
  "reasoning": "<your chain-of-thought: what you observe, what you plan to do>",
  "code": "<valid Python code to execute, or empty string if done>",
  "done": <true | false>,
  "final_answer": "<natural language answer — only when done=true, else empty string>",
  "plot_requests": [
    {"title": "<chart title>", "description": "<what to plot>"}
  ]
}

Rules:
- `plot_requests` is optional; include it only on the final iteration (done=true)
  when a chart would help communicate the answer.
- When done=false, `final_answer` must be an empty string.
- When done=true, `code` should be empty unless a final aggregation is needed.
- Never use markdown fences inside the JSON values.
- `code` must be valid Python. Imports are allowed. Use pandas, numpy, plotly.
- Available in the namespace: pd, go, px, json, csv_paths (list), csv_path (first path),
  csv_path_0, csv_path_1, ... (individual paths), plus any variables you defined earlier.
- Store intermediate results in clearly named variables (e.g. df_2014, merged, top_stocks).
- Always assign the most important result to `result` at the end of your code block.
- Do NOT print anything.

Data quality rules (apply to EVERY analysis):
- ALWAYS check for extreme outliers before computing means on financial metrics.
  Prefer median, OR clip outliers explicitly (IQR method or fixed domain bounds).
  Never use raw mean on financial ratios (ROE, P/E, debt ratios) without outlier handling.
- ALWAYS verify which CSV file corresponds to which year before drawing year-specific
  conclusions. Check the filename or inspect a date/year column if available.
- For multi-year analysis, label each DataFrame with its source year before merging
  (e.g. df_2014["year"] = 2014).
- Document your outlier handling choice in the `reasoning` field so the validator
  can independently verify it.
"""


def _build_iteration_prompt(
    question: str,
    iteration: int,
    traces: list[IterationTrace],
    executor: PersistentExecutor,
) -> str:
    """Build the user-turn message for the current iteration."""
    history_parts = []
    for t in traces:
        history_parts.append(
            f"=== Iteration {t.iteration} ===\n"
            f"Reasoning: {t.reasoning}\n"
            f"Code executed:\n{t.code}\n"
            f"Output: {t.execution_output}\n"
            + (f"Error: {t.error}\n" if t.error else "")
        )

    history_block = "\n".join(history_parts) if history_parts else "(none yet)"
    namespace_block = executor.snapshot_summary()

    return f"""QUESTION: {question}

ITERATION: {iteration + 1} / {MAX_ITERATIONS}

HISTORY OF PREVIOUS ITERATIONS:
{history_block}

CURRENT NAMESPACE VARIABLES:
{namespace_block}

Now produce your next JSON response."""


# ---------------------------------------------------------------------------
# Visualization generator
# ---------------------------------------------------------------------------

def _generate_plot(
    title: str,
    description: str,
    executor: PersistentExecutor,
    client: OpenAI,
    model_name: str,
) -> Optional[str]:
    """
    Ask the LLM to write Plotly code for the requested chart,
    execute it in the persistent namespace, return fig.to_json() or None.
    """
    namespace_block = executor.snapshot_summary()

    prompt = f"""You are a Plotly expert.
Write Python code that creates a chart described below.

Chart title   : {title}
Chart request : {description}

EXECUTION ENVIRONMENT (variables already available):
{namespace_block}
Also available: pd, go, px

RULES:
- Return ONLY valid Python code. No markdown fences, no prose.
- Last line must assign the figure to `fig`.
- Use existing namespace variables — do NOT re-read CSV files.
- Limit displayed rows/categories to at most 25 (use .head(25)).

LAYOUT RULES (follow exactly — a post-processor will override anything that conflicts):
- Set title via fig.update_layout(title=dict(text="<short title>")). Keep
  the title SHORT (under 60 chars). Do NOT embed subtitles in the title string.
- NEVER use template="plotly_dark" or any dark theme. Use template="plotly_white"
  or omit the template entirely — the post-processor enforces white.
- Do NOT set yaxis=dict(range=[...]) or xaxis=dict(range=[...]). Hard-coded
  axis ranges break autofit after data clipping. Let Plotly auto-range.
- Do NOT set legend positioning — the post-processor moves the legend to the
  bottom automatically to prevent title overlap.
- For ANY secondary y-axis, keep axis titles SHORT (<= 30 chars).
- Rotate long x-axis tick labels: tickangle=-30.
- For time-series charts where x-axis represents discrete years or 
  categories, always cast x values to strings before plotting 
  (e.g. df['year'] = df['year'].astype(str)) and use 
  fig.update_xaxes(type='category') to prevent Plotly from 
  interpolating between years.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    code = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Execute in a copy of the namespace (don't pollute persistent state with `fig`)
    plot_env = dict(executor.namespace)
    try:
        exec(code, plot_env)  # noqa: S102
    except Exception:
        return None

    fig = plot_env.get("fig")
    if fig is None:
        return None

    # Deterministic safety pass: enforce margins / automargin / non-overlapping
    # title regardless of what the LLM produced. This guarantees no clipped
    # axes or overlapping title+subtitle in the rendered output.
    try:
        fig = _apply_safe_layout(fig, title)
    except Exception:
        # If the safety pass fails for any reason, fall back to the raw fig
        # rather than dropping the plot entirely.
        pass

    try:
        return fig.to_json()
    except Exception:
        return None


def _apply_safe_layout(fig: "go.Figure", title: str) -> "go.Figure":
    """
    Deterministically enforce layout safety on an LLM-generated figure.

    Fixes ALL recurring rendering problems observed in practice:
      1. Title / legend overlap -> legend moved to BOTTOM, title at top with
         generous margin.
      2. Clipped secondary axis titles -> automargin + reserved right margin.
      3. Dark background (LLM ignores template prompt) -> force plotly_white.
      4. Y-axis range locked by data-clip logic -> force autorange=True on
         all primary axes so Plotly re-fits the visible data.

    This runs AFTER LLM code, overriding whatever the model chose.
    Data (traces) is never touched — only layout.
    """
    layout = fig.layout

    # ── detect secondary y-axis ──────────────────────────────────────────
    has_secondary_y = (
        getattr(layout, "yaxis2", None) is not None
        and getattr(layout.yaxis2, "overlaying", None) is not None
    )
    right_margin = 100 if has_secondary_y else 60

    # ── core layout overrides ────────────────────────────────────────────
    fig.update_layout(
        template="plotly_white",          # force light background always
        paper_bgcolor="white",
        plot_bgcolor="white",

        # Title: short text, anchored top-left, well below figure top
        title=dict(
            text=title,
            x=0.02, xanchor="left",
            y=0.97, yanchor="top",
            font=dict(size=17, color="#222222"),
        ),

        # Margin: t=110 gives room for title + any top annotation;
        # b=100 gives room for legend at bottom + rotated x labels.
        margin=dict(t=110, r=right_margin, b=100, l=75),

        # Legend: move to BOTTOM so it never competes with the title.
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,          # below the chart area
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),

        font=dict(color="#333333"),
    )

    # ── axis fixes ───────────────────────────────────────────────────────
    # automargin: prevent long axis titles / tick labels from being clipped.
    # autorange=True: override any range the LLM hard-coded after data clipping
    # (fixes the "box chart body disappears above the top" bug).
    fig.update_xaxes(automargin=True, tickangle=-30)
    fig.update_yaxes(automargin=True, autorange=True)

    if has_secondary_y:
        fig.update_layout(
            yaxis2=dict(automargin=True, autorange=True)
        )

    return fig


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str) -> dict:
    """
    Extract JSON from the LLM response.
    Handles accidental markdown fences and leading/trailing prose.
    """
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: treat the whole text as final_answer
    return {
        "reasoning": "Could not parse structured response.",
        "code": "",
        "done": True,
        "final_answer": raw.strip(),
        "plot_requests": [],
    }


# ---------------------------------------------------------------------------
# Main analytics agent
# ---------------------------------------------------------------------------

def run_analytics_agent(
    question: str,
    csv_paths: list[str],
    model_name: str,
    api_key: str,
) -> AnalyticsResult:
    """
    Entry point for the iterative analytics agent.

    Args:
        question  : user's natural language question
        csv_paths : list of CSV file paths (agent will self-inspect)
        model_name: OpenAI model name
        api_key   : OpenAI API key

    Returns:
        AnalyticsResult with final_answer, plots, and iteration traces
    """
    client = OpenAI(api_key=api_key)
    executor = PersistentExecutor(csv_paths=csv_paths)
    traces: list[IterationTrace] = []

    # Build a dataset-specific column reference and append it to the system
    # prompt. This gives the agent the exact column names and unit scales
    # for this dataset upfront, eliminating the need to guess names like
    # 'P/E' vs 'PE ratio', or to assume ROE is a percentage rather than
    # a decimal ratio.
    dataset_ref = build_dataset_reference(csv_paths)
    system_content = SYSTEM_PROMPT + dataset_ref

    messages = [{"role": "system", "content": system_content}]

    for iteration in range(MAX_ITERATIONS):
        # Build user message for this iteration
        user_msg = _build_iteration_prompt(
            question=question,
            iteration=iteration,
            traces=traces,
            executor=executor,
        )
        messages.append({"role": "user", "content": user_msg})

        # Call LLM
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        raw_content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_content})

        # Parse response
        parsed = _parse_llm_response(raw_content)
        reasoning = parsed.get("reasoning", "")
        code = parsed.get("code", "").strip()
        done = bool(parsed.get("done", False))
        final_answer = parsed.get("final_answer", "").strip()
        plot_requests = parsed.get("plot_requests", [])

        # Execute code (with retry on error)
        output_repr = ""
        exec_error = None

        if code:
            for attempt in range(MAX_RETRIES_ON_ERROR + 1):
                output_repr, exec_error = executor.execute(code)
                if exec_error is None:
                    break
                # On error: ask LLM to fix the code
                if attempt < MAX_RETRIES_ON_ERROR:
                    fix_msg = (
                        f"The code raised an error:\n{exec_error}\n"
                        f"Fix the code and return only the corrected JSON."
                    )
                    messages.append({"role": "user", "content": fix_msg})
                    fix_response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0,
                    )
                    fix_raw = fix_response.choices[0].message.content
                    messages.append({"role": "assistant", "content": fix_raw})
                    fix_parsed = _parse_llm_response(fix_raw)
                    code = fix_parsed.get("code", code).strip()
                    reasoning += f" [retry {attempt + 1}: fixing error]"

        # Record trace
        trace = IterationTrace(
            iteration=iteration + 1,
            reasoning=reasoning,
            code=code,
            execution_output=output_repr,
            error=exec_error,
            done=done,
        )
        traces.append(trace)

        # Check stopping condition
        if done:
            # Generate plots if requested
            plots = []
            for req in plot_requests:
                title = req.get("title", "Chart")
                description = req.get("description", "")
                fig_json = _generate_plot(
                    title=title,
                    description=description,
                    executor=executor,
                    client=client,
                    model_name=model_name,
                )
                if fig_json:
                    plots.append({"title": title, "plotly_json": fig_json})

            # If final_answer is empty, generate it from the last result
            if not final_answer:
                final_answer = _generate_final_answer(
                    question=question,
                    executor=executor,
                    client=client,
                    model_name=model_name,
                )

            return AnalyticsResult(
                final_answer=final_answer,
                plots=plots,
                traces=traces,
                total_iterations=iteration + 1,
                success=True,
            )

    # Reached MAX_ITERATIONS without done=True — generate best-effort answer
    final_answer = _generate_final_answer(
        question=question,
        executor=executor,
        client=client,
        model_name=model_name,
    )
    return AnalyticsResult(
        final_answer=final_answer,
        plots=[],
        traces=traces,
        total_iterations=MAX_ITERATIONS,
        success=False,
        error=f"Reached maximum iterations ({MAX_ITERATIONS}) without completion.",
    )


def _generate_final_answer(
    question: str,
    executor: PersistentExecutor,
    client: OpenAI,
    model_name: str,
) -> str:
    """Fallback: ask LLM to synthesize a final answer from current namespace state."""
    result_repr = _repr_result(executor.get_result())
    namespace_block = executor.snapshot_summary()

    prompt = f"""You are a data analyst. Based on the computed results below,
write a clear and concise natural language answer to the user's question.

Question: {question}

Most recent result:
{result_repr}

Available computed variables:
{namespace_block}

Instructions:
- Answer the question directly using specific numbers and facts from the data.
- Keep it to 2-4 sentences.
- Do not mention code, DataFrames, or implementation details.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_workflow(
    question: str,
    csv_paths: list[str],
    model_name: str,
    api_key: str,
) -> dict:
    
    result = run_analytics_agent(
        question=question,
        csv_paths=csv_paths,
        model_name=model_name,
        api_key=api_key,
    )
    return {
        "final_answer": result.final_answer,
        "plots": result.plots,
        "traces": [
            {
                "iteration": t.iteration,
                "reasoning": t.reasoning,
                "code": t.code,
                "output": t.execution_output,
                "error": t.error,
                "done": t.done,
            }
            for t in result.traces
        ],
        "total_iterations": result.total_iterations,
        "success": result.success,
        "error": result.error,
    }
