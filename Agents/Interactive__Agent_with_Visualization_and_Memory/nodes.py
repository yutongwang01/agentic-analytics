from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env_file(env_path: str = ".env") -> dict:
    """Load simple KEY=VALUE pairs from a .env file without extra dependencies."""
    values: dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


# ---------------------------------------------------------------------------
# Execution Result
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class ExecutionResult:
    data: Any                        # DataFrame / scalar / dict / Series
    data_type: str                   # "dataframe" | "scalar" | "dict" | "series"
    summary: str
    schema: Optional[dict]
    provenance: str = ""
    turn_id: int = 0
    question: str = ""

    def to_context_str(self) -> str:
        """The context string injected into the follow-up prompt."""
        if self.data_type == "dataframe":
            preview = self.data.head(3).to_string()
            return (
                f"Previous result: DataFrame with shape {self.data.shape}, "
                f"columns={list(self.data.columns)}\nPreview:\n{preview}"
            )
        elif self.data_type == "series":
            preview = self.data.head(5).to_string()
            return f"Previous result: Series with {len(self.data)} entries\nPreview:\n{preview}"
        else:
            return f"Previous result ({self.data_type}): {self.data}"


def _make_execution_result(raw: Any, code: str = "", question: str = "", turn_id: int = 0) -> ExecutionResult:
    """Wrap the raw output of `execute_code` into an `ExecutionResult`."""
    if isinstance(raw, pd.DataFrame):
        data_type = "dataframe"
        summary = f"DataFrame with {raw.shape[0]} rows and {raw.shape[1]} columns, columns: {list(raw.columns)}"
        schema = {
            "columns": list(raw.columns),
            "dtypes": {col: str(dt) for col, dt in raw.dtypes.items()},
            "shape": list(raw.shape),
        }
    elif isinstance(raw, pd.Series):
        data_type = "series"
        summary = f"Series with {len(raw)} entries, name={raw.name}"
        schema = {"name": raw.name, "dtype": str(raw.dtype), "length": len(raw)}
    elif isinstance(raw, dict):
        data_type = "dict"
        summary = f"Dict with keys: {list(raw.keys())}"
        schema = {"keys": list(raw.keys())}
    else:
        data_type = "scalar"
        summary = f"Scalar value: {raw}"
        schema = None

    return ExecutionResult(
        data=raw,
        data_type=data_type,
        summary=summary,
        schema=schema,
        provenance=code,
        turn_id=turn_id,
        question=question,
    )


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def inspect_csv_schema(csv_path: str) -> dict:
    """Read a small sample of the CSV so code generation can adapt to the schema."""
    df = pd.read_csv(csv_path)
    preview_df = df.head(3).where(pd.notna(df.head(3)), None)
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview_rows": preview_df.to_dict(orient="records"),
        "shape": df.shape,
    }


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def generate_code(
    question: str,
    csv_path: str,
    model_name: str,
    api_key: str,
    memory_context: str = "",
    is_followup: bool = False,
) -> str:
    """Ask the LLM to produce Python/pandas code that stores its answer in `result`."""
    client = OpenAI(api_key=api_key)
    schema = inspect_csv_schema(csv_path)

    columns_text = ", ".join(schema["columns"])
    dtypes_text = "\n".join(
        f"  - {col}: {dtype}" for col, dtype in schema["dtypes"].items()
    )
    preview_text = "\n".join(f"  - {row}" for row in schema["preview_rows"])
    n_rows, n_cols = schema["shape"]

    memory_block = ""
    if memory_context:
        followup_hint = (
            "- `prior_data` is already available in the execution environment; "
            "use it directly instead of reading the CSV again.\n"
            if is_followup else ""
        )
        memory_block = f"""
=== CONVERSATION CONTEXT ===
{memory_context}

=== FOLLOW-UP RULES ===
{followup_hint}- If the question refers to a previous result, operate on `prior_data` directly.
- If the question is independent, read the CSV as usual.
"""

    prompt = f"""You are a Python data-analysis expert.
Your task: write self-contained Python code that answers a question about a CSV file.

=== STRICT OUTPUT RULES ===
- Return ONLY valid Python code.
- No markdown fences, no prose, no comments explaining the answer.
- The very last line must assign the answer to a variable called `result`.

=== CODE RULES ===
- `csv_path` is already defined in the execution environment; do NOT redefine it.
- Read the CSV exactly once: df = pd.read_csv(csv_path)
- import pandas as pd at the top (and any other stdlib you need).
- Do NOT import third-party libraries other than pandas and numpy.
- Do NOT print anything.
- Use only columns that exist in the schema below; never invent column names.
- If the question refers to a concept absent from the schema, set:
    result = {{"message": "The dataset does not contain information about <concept>."}}

=== DATA HANDLING RULES (apply to every question) ===
- Prefer grouped / aggregated results over returning raw rows.
- When cells contain comma-separated values (e.g. country lists), split and explode
  before grouping so each value is counted individually.
- When iterating over groupby, use `for name, group_df in ...`; never use `group_df.name`.
- For geographic analysis, bin latitude/longitude into ranges rather than treating
  them as continuous values.
- Before ranking, filter out groups with very few observations (e.g. < 30 rows or
  < 1 % of total) to avoid noise dominating the top results.
- `result` must be a scalar, dict, pandas Series, or pandas DataFrame —
  never a plain English sentence.

=== DATASET INFO ===
File     : {csv_path}
Shape    : {n_rows} rows x {n_cols} columns
Columns  : {columns_text}
Dtypes   :
{dtypes_text}
Preview  :
{preview_text}
{memory_block}
=== QUESTION ===
{question}
""".strip()

    response = client.responses.create(model=model_name, input=prompt, temperature=0)
    code = response.output_text.strip()

    # Strip accidental markdown fences
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return code.strip()


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def execute_code(
    code: str,
    csv_path: str,
    prior_result: Optional["ExecutionResult"] = None,
    question: str = "",
    turn_id: int = 0,
) -> tuple[ExecutionResult, str | None]:
    """Execute the generated code, returning (ExecutionResult, error_message)."""
    env: dict = {"csv_path": csv_path, "pd": pd}

    if prior_result is not None:
        env["prior_data"] = prior_result.data

    try:
        exec(code, env)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        err_result = _make_execution_result(
            {"error": str(exc)}, code=code, question=question, turn_id=turn_id
        )
        return err_result, error_msg

    raw = env.get("result")
    exec_result = _make_execution_result(raw, code=code, question=question, turn_id=turn_id)
    return exec_result, None


# ---------------------------------------------------------------------------
# Evaluation  (LLM-based)
# ---------------------------------------------------------------------------

def evaluate_result(
    execution_error: str | None,
    result: ExecutionResult,
    question: str,
    model_name: str,
    api_key: str,
) -> str:
    """Use an LLM to judge whether the execution result sufficiently answers the question."""
    # Hard fail: execution itself crashed
    if execution_error is not None:
        return "FAIL"

    client = OpenAI(api_key=api_key)

    # Serialize result for the prompt
    if result.data_type == "dataframe":
        result_repr = (
            f"DataFrame shape: {result.data.shape}, "
            f"columns: {list(result.data.columns)}\n"
            f"Preview:\n{result.data.head(5).to_string()}"
        )
    elif result.data_type == "series":
        result_repr = (
            f"Series length: {len(result.data)}\n"
            f"Preview:\n{result.data.head(5).to_string()}"
        )
    else:
        result_repr = str(result.data)

    prompt = f"""You are evaluating whether a data analysis result sufficiently answers a user's question.

Question: {question}

Execution Result:
{result_repr}

Does this result contain enough information to answer the question?
Reply with exactly one word: PASS or FAIL.
- PASS: the result directly addresses the question with concrete data.
- FAIL: the result is empty, irrelevant, erroneous, or too incomplete to answer the question.
"""

    response = client.responses.create(model=model_name, input=prompt)
    verdict = response.output_text.strip().upper()
    return "PASS" if "PASS" in verdict else "FAIL"


# ---------------------------------------------------------------------------
# Answer formatting  (LLM-based)
# ---------------------------------------------------------------------------

def format_final_answer(
    result: ExecutionResult,
    evaluation: str,
    question: str,
    model_name: str,
    api_key: str,
    error_message: str | None = None,
) -> str:
    """Use an LLM to generate a natural language answer from the execution result."""
    if evaluation != "PASS":
        return (
            f"Execution failed: {error_message}"
            if error_message
            else "Execution failed: no usable result was produced."
        )

    client = OpenAI(api_key=api_key)

    # Serialize result for the prompt
    if result.data_type == "dataframe":
        result_repr = (
            f"DataFrame shape: {result.data.shape}, "
            f"columns: {list(result.data.columns)}\n"
            f"Top rows:\n{result.data.head(10).to_string()}"
        )
    elif result.data_type == "series":
        result_repr = (
            f"Series length: {len(result.data)}\n"
            f"Values:\n{result.data.head(10).to_string()}"
        )
    else:
        result_repr = str(result.data)

    prompt = f"""You are a data analyst. Given a user question and the computed result,
write a clear, concise natural language answer.

Question: {question}

Computed Result:
{result_repr}

Instructions:
- Answer the question directly using the data above.
- Be specific: include actual numbers, names, or values from the result.
- Keep it to 1-3 sentences.
- Do not mention code, DataFrames, or technical implementation details.
"""

    response = client.responses.create(model=model_name, input=prompt)
    return response.output_text.strip()


# ---------------------------------------------------------------------------
# Session Memory
# ---------------------------------------------------------------------------

FOLLOWUP_KEYWORDS = [
    "previous", "prior", "last result", "that result",
    "top 5", "top 10", "filter", "sort this", "now show",
    "above result", "from before",
]


class SessionMemory:
    def __init__(self, max_turns: int = 5):
        self.turns: list[dict] = []
        self.max_turns = max_turns

    @property
    def last_result(self) -> Optional[ExecutionResult]:
        if self.turns:
            return self.turns[-1]["result"]
        return None

    def add_turn(self, question: str, result: ExecutionResult, answer: str):
        self.turns.append({"question": question, "result": result, "answer": answer})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def is_followup(self, question: str) -> bool:
        q = question.lower()
        return any(k in q for k in FOLLOWUP_KEYWORDS)

    def get_context_for_prompt(self) -> str:
        """Returns a summary of the last 2 rounds, injected into the codegen prompt."""
        if not self.turns:
            return ""
        recent = self.turns[-2:]
        parts = []
        for i, t in enumerate(recent):
            parts.append(
                f"[Turn {i+1}] Question: {t['question']}\n"
                f"Result: {t['result'].to_context_str()}"
            )
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Visualization Agent
# ---------------------------------------------------------------------------

import plotly.graph_objects as go
import plotly.express as px


@dataclass
class VisualizationResult:
    should_visualize: bool
    chart_type: Optional[str]      # "bar" | "line" | "scatter" | "pie" | "histogram" | None
    plotly_code: Optional[str]
    figure_json: Optional[str]     # fig.to_json() — consumed directly by the frontend
    error: Optional[str]


def _build_data_repr(result: ExecutionResult, max_rows: int = 20) -> str:
    """Serialize ExecutionResult into a compact string for LLM prompts."""
    if result.data_type == "dataframe":
        df = result.data
        preview = df.head(max_rows).to_string()
        return (
            f"Type: DataFrame\n"
            f"Shape: {df.shape}\n"
            f"Columns: {list(df.columns)}\n"
            f"Dtypes: { {col: str(dt) for col, dt in df.dtypes.items()} }\n"
            f"Preview (first {min(max_rows, len(df))} rows):\n{preview}"
        )
    elif result.data_type == "series":
        s = result.data
        preview = s.head(max_rows).to_string()
        return (
            f"Type: Series\n"
            f"Name: {s.name}\n"
            f"Length: {len(s)}\n"
            f"Dtype: {s.dtype}\n"
            f"Preview:\n{preview}"
        )
    elif result.data_type == "dict":
        return f"Type: dict\nKeys: {list(result.data.keys())}\nValue: {result.data}"
    else:
        return f"Type: scalar\nValue: {result.data}"


def decide_visualization(
    question: str,
    result: ExecutionResult,
    model_name: str,
    api_key: str,
) -> tuple[bool, Optional[str]]:
    """
    Ask the LLM whether a visualization is useful and which chart type to use.
    Returns (should_visualize, chart_type).
    chart_type is one of: "bar", "line", "scatter", "pie", "histogram", or None.
    """
    # Scalars almost never benefit from a chart
    if result.data_type == "scalar":
        return False, None

    client = OpenAI(api_key=api_key)
    data_repr = _build_data_repr(result)

    prompt = f"""You are a data visualization expert.
Given a user question and a computed data result, decide whether a chart would help communicate the answer.

=== USER QUESTION ===
{question}

=== COMPUTED RESULT ===
{data_repr}

=== DECISION RULES ===
- Visualization IS useful for: comparisons across categories, trends over time, distributions, rankings, proportions.
- Visualization is NOT useful for: a single scalar value, error messages, free-text answers, dict with a single key.
- If the result is a DataFrame or Series with multiple rows/entries, visualization is usually useful.

=== CHART TYPE GUIDE ===
- "bar"       : comparisons across categories (counts, averages, totals)
- "line"      : trends over time or ordered sequences
- "scatter"   : correlation between two numeric variables
- "pie"       : part-of-whole proportions (use only when ≤ 8 categories)
- "histogram" : distribution of a single numeric variable

=== OUTPUT FORMAT ===
Reply with ONLY a JSON object on a single line. No prose, no markdown.
Example: {{"should_visualize": true, "chart_type": "bar", "reason": "comparing counts across categories"}}
If should_visualize is false, set chart_type to null.
"""

    response = client.responses.create(model=model_name, input=prompt)
    raw = response.output_text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(raw)
        should_viz = bool(parsed.get("should_visualize", False))
        chart_type = parsed.get("chart_type") if should_viz else None
        return should_viz, chart_type
    except json.JSONDecodeError:
        # Fallback: if we can't parse, don't visualize
        return False, None


def generate_plot_code(
    question: str,
    result: ExecutionResult,
    chart_type: str,
    model_name: str,
    api_key: str,
) -> str:
    """
    Ask the LLM to generate Plotly code that visualizes the execution result.
    The code must use `viz_data` (already in the execution environment) and
    assign the final figure to `fig`.
    """
    client = OpenAI(api_key=api_key)
    data_repr = _build_data_repr(result)

    # Build column hint so the LLM never invents column names
    if result.data_type == "dataframe":
        col_hint = f"Available columns: {list(result.data.columns)}"
    elif result.data_type == "series":
        col_hint = f"Series name: {result.data.name}, index name: {result.data.index.name}"
    else:
        col_hint = ""

    prompt = f"""You are a Python Plotly expert.
Write code that creates a {chart_type} chart to visualize the data described below.

=== STRICT OUTPUT RULES ===
- Return ONLY valid Python code.
- No markdown fences, no prose, no comments.
- The very last line must assign the Plotly figure to a variable called `fig`.

=== EXECUTION ENVIRONMENT ===
- `viz_data` is already defined and contains the data (do NOT redefine it).
  {col_hint}
- `go` = plotly.graph_objects  (already imported)
- `px` = plotly.express        (already imported)
- `pd` = pandas                (already imported)
- Do NOT import anything else.
- Do NOT read any files.

=== DATA DESCRIPTION ===
{data_repr}

=== CHART REQUIREMENTS ===
- Chart type: {chart_type}
- The chart must directly answer or illustrate the user question: "{question}"
- Use a clear, descriptive title.
- Label axes where applicable.
- If viz_data is a Series, use its index as the x-axis and values as y-axis.
- If viz_data is a DataFrame, use the most meaningful columns for x and y.
- Limit displayed categories to at most 20 to avoid clutter (use .head(20) if needed).
- Use `fig.update_layout(template="plotly_white")` for a clean look.
"""

    response = client.responses.create(model=model_name, input=prompt)
    code = response.output_text.strip()

    # Strip accidental markdown fences
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return code.strip()


def execute_plot_code(
    code: str,
    result: ExecutionResult,
) -> tuple[Optional[str], Optional[str]]:
    """
    Execute the generated Plotly code in a controlled environment.
    Returns (figure_json, error_message).
    figure_json is fig.to_json() — ready for the frontend.
    """
    # Cap data size to prevent huge JSON payloads
    if result.data_type == "dataframe":
        viz_data = result.data.head(50)
    elif result.data_type == "series":
        viz_data = result.data.head(50)
    else:
        viz_data = result.data

    env: dict = {
        "viz_data": viz_data,
        "go": go,
        "px": px,
        "pd": pd,
    }

    try:
        exec(code, env)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

    fig = env.get("fig")
    if fig is None:
        return None, "Plot code did not assign a figure to `fig`."

    try:
        figure_json = fig.to_json()
    except Exception as exc:  # noqa: BLE001
        return None, f"fig.to_json() failed: {exc}"

    return figure_json, None


def run_visualization_agent(
    question: str,
    result: ExecutionResult,
    model_name: str,
    api_key: str,
) -> VisualizationResult:
    """
    Full visualization pipeline:
      1. Decide whether to visualize (LLM).
      2. Generate Plotly code (LLM).
      3. Execute the code and return figure_json.
    Always returns a VisualizationResult — never raises.
    """
    # Step 1 — Decision
    should_viz, chart_type = decide_visualization(
        question=question,
        result=result,
        model_name=model_name,
        api_key=api_key,
    )

    if not should_viz:
        return VisualizationResult(
            should_visualize=False,
            chart_type=None,
            plotly_code=None,
            figure_json=None,
            error=None,
        )

    # Step 2 — Code generation
    try:
        plot_code = generate_plot_code(
            question=question,
            result=result,
            chart_type=chart_type,
            model_name=model_name,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001
        return VisualizationResult(
            should_visualize=True,
            chart_type=chart_type,
            plotly_code=None,
            figure_json=None,
            error=f"Code generation failed: {exc}",
        )

    # Step 3 — Execution
    figure_json, exec_error = execute_plot_code(code=plot_code, result=result)

    return VisualizationResult(
        should_visualize=True,
        chart_type=chart_type,
        plotly_code=plot_code,
        figure_json=figure_json,
        error=exec_error,
    )
