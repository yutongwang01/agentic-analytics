"""
validator_agent.py

Design principles:
  - INDEPENDENT: generates its own plan without seeing analytics agent's code
  - REASONING-BASED: checks correctness & completeness, not just PASS/FAIL
  - MULTI-TABLE: supports joins, unions via extended operators.py
  - Returns a structured verdict with actionable retry suggestions

Verdict values:
  "PASS"        - analysis is correct and complete
  "INCOMPLETE"  - question not fully answered (e.g. missing year check)
  "SUSPICIOUS"  - numerical inconsistency detected
  "RETRY"       - critical error, must redo analysis
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from openai import OpenAI

from executor import ExecutionResult, infer_schema, load_csv, run_plan
from operators import TABLE_REGISTRY
from analytics_agent import build_dataset_reference


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ValidatorVerdict:
    verdict: str                    # "PASS" | "INCOMPLETE" | "SUSPICIOUS" | "RETRY"
    reason: str                     # human-readable explanation
    suggestion: str                 # what the analytics agent should do on retry
    validator_result: Optional[Any] # the validator's own computed result (for comparison)
    plan_used: Optional[dict]       # the plan the validator generated
    execution_trace: Optional[str]  # pretty-printed trace table


# ---------------------------------------------------------------------------
# Prompt: plan generation
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are an independent data validation agent.
Your job: given a user question and dataset schema(s), generate a structured
execution plan to independently verify the analysis.

You must output a single JSON object — nothing else. Schema:
{
  "reasoning": "<why you chose these steps>",
  "plan": {
    "steps": [ <operator step>, ... ]
  }
}

Available operators and their step formats:

1. derive_columns
   {"op": "derive_columns", "derive": [{"new_column": "name", "expr": <expr>}]}
   Expressions: {"type": "column", "value": "col"} | {"type": "literal", "value": 3.14}
                {"type": "arithmetic", "operation": "add|subtract|multiply|divide", "left": <expr>, "right": <expr>}
                {"type": "compare", "operation": "eq|ne|gt|gte|lt|lte", "left": <expr>, "right": <expr>}
                {"type": "logical", "operation": "and|or", "left": <expr>, "right": <expr>}
                {"type": "logical", "operation": "not", "operand": <expr>}

2. filter_rows
   {"op": "filter_rows", "conditions": [{"column": "col", "operator": "==|!=|>|>=|<|<=|in|not_in|contains|is_null|not_null", "value": <val>}], "logic": "and"}

3. group_and_aggregate
   {"op": "group_and_aggregate", "group_by": ["col"], "metrics": [{"function": "mean|sum|count|min|max|median|std|nunique", "column": "col", "as": "alias"}]}

4. sort_rows
   {"op": "sort_rows", "sort_by": [{"column": "col", "direction": "asc|desc"}]}

5. limit_rows
   {"op": "limit_rows", "k": 5}

6. select_columns
   {"op": "select_columns", "columns": ["col1", "col2"]}

7. distinct_rows
   {"op": "distinct_rows"} or {"op": "distinct_rows", "columns": ["col"]}

8. dropna
   {"op": "dropna"} or {"op": "dropna", "columns": ["col"]}

9. rank
   {"op": "rank", "column": "col", "as": "rank_col", "direction": "desc", "partition_by": ["group_col"]}

10. join_tables   [MULTI-TABLE]
    {"op": "join_tables", "right_table": "<table_name>", "on": ["key_col"], "how": "inner|left|right|outer"}
    OR: {"op": "join_tables", "right_table": "<table_name>", "left_on": ["col_a"], "right_on": ["col_b"], "how": "inner"}

11. union_tables  [MULTI-TABLE]
    {"op": "union_tables", "other_tables": ["<table_name>", ...], "align": "common|all"}
    CRITICAL: use align="all" (not "common") whenever you add a literal column
    before union. align="common" drops columns that don't exist in ALL tables,
    so a 'year' column added only to the primary table will be silently lost.
    CORRECT multi-year pattern for comparing 2014 vs 2017:
      Option A — union approach (use align="all"):
        1. add_literal_column year=2014  (primary = df_2014)
        2. [manually note: df_2017 rows will have NaN for 'year' after union]
        This still does not work cleanly. Use Option B instead.
      Option B — separate pipelines (RECOMMENDED for year comparison):
        Run steps 1-N on df_2014 (filter→aggregate), note the result.
        Then re-run the same steps on df_2017 as a second plan.
        Compare the two results in the verdict step.
        To do this, set "primary_table" hint in your reasoning, and generate
        TWO separate plans — one per year — and note both results.

12. add_literal_column  [MULTI-TABLE]
    {"op": "add_literal_column", "column": "year", "value": 2014}

IMPORTANT RULES:
- Plan must produce a result comparable to the analytics agent's answer.
- For multi-year questions, use union_tables or join_tables with the registered table names.
- Keep plans focused: 2-6 steps is typical. Do not over-engineer.
- Do NOT use natural language anywhere in the plan steps.
- COLUMN NAMES ARE CASE- AND SPACING-SENSITIVE. You MUST copy column names
  CHARACTER-FOR-CHARACTER from the DATASET SCHEMAS section below. Do NOT
  invent, reformat, capitalize, or "tidy up" column names. For example, if
  the schema lists 'Total liabilities' you must write exactly that — never
  'Total Liabilities' or 'total_liabilities'.
- If a column you want is NOT listed in the provided schema, do NOT guess a
  name. Either pick the closest column that IS listed, or use a pre-computed
  ratio column if one is available (e.g. 'debtRatio', 'debtEquityRatio').
"""


def _select_relevant_columns(
    question: str,
    analytics_summary: str,
    all_columns: list[str],
    max_cols: int = 80,
) -> list[str]:
    """
    Choose which columns to show the planner.

    The previous implementation took the first 40 columns blindly. With a
    225-column financial dataset, this hid critical columns (e.g.
    'Total liabilities' at index 53), forcing the planner to hallucinate
    column names. This selector instead guarantees that columns whose names
    overlap with the question / analytics terms are ALWAYS included, then
    fills the remaining budget with the leading columns for general context.

    Deterministic: no LLM call, no randomness.
    """
    # Tokenize question + analytics summary into lowercase word stems.
    text = (question + " " + analytics_summary).lower()
    # Pull alphabetic tokens of length >= 3 (drops 'of', 'to', etc.)
    tokens = set(re.findall(r"[a-z]{3,}", text))

    relevant: list[str] = []
    for col in all_columns:
        col_l = col.lower()
        # A column is relevant if any question token appears in the column
        # name, OR any whitespace-split part of the column name appears as a
        # question token. This catches both 'debt'->'debtRatio' and
        # 'liabilities'->'Total liabilities'.
        col_parts = set(re.findall(r"[a-z]{3,}", col_l))
        if tokens & col_parts or any(tok in col_l for tok in tokens):
            relevant.append(col)

    # Always-include leading columns for general context (sector, name, etc.)
    leading = all_columns[:25]

    # Merge preserving original order, dedup.
    seen = set()
    selected: list[str] = []
    for col in leading + relevant:
        if col not in seen:
            seen.add(col)
            selected.append(col)

    # If still over budget, keep leading + all relevant (relevant never dropped).
    if len(selected) > max_cols:
        # Prioritise: leading context first, then relevant, cap total.
        priority = [c for c in selected if c in set(relevant)] + \
                   [c for c in selected if c not in set(relevant)]
        # Re-sort by original column order for readability.
        order = {c: i for i, c in enumerate(all_columns)}
        selected = sorted(set(priority[:max_cols]), key=lambda c: order.get(c, 1e9))

    return selected


def _build_planner_prompt(
    question: str,
    schemas: dict[str, dict],
    analytics_output_summary: str,
    registered_tables: list[str],
    dataset_ref: str = "",
) -> str:
    schema_block = ""
    for name, schema in schemas.items():
        all_cols = schema["columns"]
        # Smart selection instead of blind [:40] truncation.
        shown_cols = _select_relevant_columns(
            question=question,
            analytics_summary=analytics_output_summary,
            all_columns=all_cols,
        )
        cols_text = ", ".join(shown_cols)
        schema_block += f"\n--- Table: {name} ---\n"
        schema_block += f"Shape: {schema['shape']}\n"
        schema_block += f"Total columns: {len(all_cols)} (showing {len(shown_cols)} most relevant)\n"
        schema_block += f"Columns: {cols_text}\n"
        # Show dtypes for the SHOWN numeric columns (so types match what's visible).
        numeric_cols = {c: t for c, t in schema["dtypes"].items()
                        if ("float" in t or "int" in t) and c in set(shown_cols)}
        if numeric_cols:
            schema_block += "Numeric columns (sample): " + ", ".join(
                list(numeric_cols.keys())[:20]
            ) + "\n"

    table_list = ", ".join(registered_tables) if registered_tables else "(none)"

    return f"""QUESTION TO VALIDATE: {question}

REGISTERED TABLE NAMES (use these in join_tables / union_tables):
{table_list}

DATASET SCHEMAS:
{schema_block}

ANALYTICS AGENT OUTPUT SUMMARY (what you need to verify):
{analytics_output_summary}

{dataset_ref}
Generate an independent validation plan. Output only the JSON object."""


# ---------------------------------------------------------------------------
# Prompt: verdict generation
# ---------------------------------------------------------------------------

VERDICT_SYSTEM_PROMPT = """You are a data analysis auditor.
Compare the analytics agent's output with the validator's independently computed result.
Determine whether the analytics agent's answer is correct, complete, and trustworthy.

Output a single JSON object — nothing else:
{
  "verdict": "PASS" | "INCOMPLETE" | "SUSPICIOUS" | "RETRY",
  "reason": "<specific explanation referencing actual numbers or missing steps>",
  "suggestion": "<concrete action the analytics agent should take on retry, or empty string if PASS>"
}

Verdict definitions:
- PASS       : answer is numerically consistent and the question is fully addressed
- INCOMPLETE : answer addresses part of the question but misses something important
               (e.g. year not verified, only one CSV used when two were needed,
               outlier handling not explained, sector coverage incomplete)
- SUSPICIOUS : answer contains values that are implausible or inconsistent with
               the validator's independent computation (>20% relative difference
               on key metrics, or directional inconsistency in rankings)
- RETRY      : execution failed, result is empty, or answer is completely wrong

Be specific in 'reason'. Reference actual numbers where possible.
Be actionable in 'suggestion'. Say exactly what to recompute or re-check.
"""


def _build_verdict_prompt(
    question: str,
    analytics_summary: str,
    validator_result_repr: str,
    execution_trace: str,
    exec_error: Optional[str],
) -> str:
    return f"""QUESTION: {question}

ANALYTICS AGENT OUTPUT:
{analytics_summary}

VALIDATOR INDEPENDENTLY COMPUTED RESULT:
{validator_result_repr}

VALIDATOR EXECUTION TRACE:
{execution_trace}

{"VALIDATOR EXECUTION ERROR: " + exec_error if exec_error else ""}

Compare and produce your verdict JSON."""


# ---------------------------------------------------------------------------
# Helper: summarise analytics agent output for prompts
# ---------------------------------------------------------------------------

def _summarise_analytics_output(analytics_output: dict) -> str:
    """Convert analytics agent's output dict into a compact string for prompts."""
    answer = analytics_output.get("final_answer", "")
    traces = analytics_output.get("traces", [])

    # Last iteration's output gives most context
    last_output = ""
    if traces:
        last = traces[-1]
        last_output = f"Last iteration output:\n{last.get('output', '')[:600]}"

    return f"""Final answer: {answer}

{last_output}

Total iterations: {analytics_output.get('total_iterations', '?')}
Success: {analytics_output.get('success', '?')}"""


def _repr_df(df: Optional[pd.DataFrame], max_rows: int = 10) -> str:
    """Compact string representation of a DataFrame for verdict prompt."""
    if df is None:
        return "(no result — execution failed)"
    if df.empty:
        return "(empty DataFrame)"
    return (
        f"Shape: {df.shape}\nColumns: {list(df.columns)}\n"
        f"{df.head(max_rows).to_string()}"
    )


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _call_llm(
    client: OpenAI,
    model_name: str,
    system: str,
    user: str,
) -> dict:
    """Call LLM with system+user and parse JSON response."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw": raw}


# ---------------------------------------------------------------------------
# Main validator entry point
# ---------------------------------------------------------------------------

def run_validator(
    question: str,
    csv_paths: list[str],
    analytics_output: dict,
    model_name: str,
    api_key: str,
) -> ValidatorVerdict:
    """
    Independently validate the analytics agent's output.

    Args:
        question        : original user question
        csv_paths       : list of CSV paths (same as given to analytics agent)
        analytics_output: dict returned by analytics_agent.run_workflow()
        model_name      : OpenAI model name
        api_key         : OpenAI API key

    Returns:
        ValidatorVerdict with verdict, reason, suggestion
    """
    client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Step 1: Load all CSVs, register in TABLE_REGISTRY, collect schemas
    # ------------------------------------------------------------------
    TABLE_REGISTRY.clear()
    schemas: dict[str, dict] = {}
    primary_csv = csv_paths[0]

    for i, path in enumerate(csv_paths):
        df = load_csv(path)
        # Derive a clean table name from filename
        table_name = _path_to_table_name(path, i)
        TABLE_REGISTRY[table_name] = df
        schemas[table_name] = infer_schema(df)

    registered_tables = list(TABLE_REGISTRY.keys())
    analytics_summary = _summarise_analytics_output(analytics_output)
    dataset_ref = build_dataset_reference(csv_paths)

    # ------------------------------------------------------------------
    # Step 2: Generate independent validation plan (LLM)
    # ------------------------------------------------------------------
    planner_user = _build_planner_prompt(
        question=question,
        schemas=schemas,
        analytics_output_summary=analytics_summary,
        registered_tables=registered_tables,
        dataset_ref=dataset_ref,
    )
    planner_response = _call_llm(
        client=client,
        model_name=model_name,
        system=PLANNER_SYSTEM_PROMPT,
        user=planner_user,
    )

    plan = planner_response.get("plan")
    planner_reasoning = planner_response.get("reasoning", "")

    # ------------------------------------------------------------------
    # Step 2b: Validate & fuzzy-fix column names BEFORE execution.
    # A weak model often hallucinates column names (e.g. 'Total Liabilities'
    # vs the real 'Total liabilities'). We auto-correct case/whitespace
    # mismatches and flag anything we cannot resolve.
    # ------------------------------------------------------------------
    column_fix_note = ""
    if plan and isinstance(plan, dict) and "steps" in plan:
        plan, unresolved_cols = _validate_and_fix_plan_columns(plan, schemas)
        if unresolved_cols:
            column_fix_note = (
                f"WARNING: the validation plan referenced column(s) that do not "
                f"exist in the dataset and could not be matched: {unresolved_cols}. "
                f"This usually means the plan is unreliable."
            )

    # ------------------------------------------------------------------
    # Step 3: Execute the validation plan
    #
    # Multi-CSV strategy: running a union-based plan is fragile because the
    # LLM (especially weaker models) often produces broken union+group_by
    # patterns where the 'year' column added by add_literal_column is dropped
    # by align="common". Instead, run the SAME plan independently against
    # each CSV and collect per-table results. This gives the verdict LLM
    # concrete per-year numbers to compare against the analytics agent.
    # ------------------------------------------------------------------
    exec_error: Optional[str] = None
    exec_result: Optional[ExecutionResult] = None
    trace_table = ""
    validator_result_repr = "(plan generation failed)"

    if plan and isinstance(plan, dict) and "steps" in plan:
        if len(csv_paths) > 1:
            # Multi-table: run plan on each CSV separately.
            per_table_results: list[str] = []
            per_table_traces: list[str] = []
            any_success = False

            for idx, path in enumerate(csv_paths):
                tname = _path_to_table_name(path, idx)
                # Fix column names against THIS table's schema only.
                table_plan, _ = _validate_and_fix_plan_columns(
                    plan, {tname: schemas[tname]}
                )
                result_i = run_plan(plan=table_plan, csv_path=path)
                per_table_traces.append(
                    f"=== {tname} ===\n{result_i.trace_as_table()}"
                )
                if result_i.error:
                    per_table_results.append(
                        f"{tname}: FAILED at step {result_i.failed_step} "
                        f"— {result_i.error}"
                    )
                else:
                    any_success = True
                    per_table_results.append(
                        f"{tname}:\n{_repr_df(result_i.final_df)}"
                    )
                    exec_result = result_i  # keep last success for compat

            trace_table = "\n\n".join(per_table_traces)
            validator_result_repr = "\n\n".join(per_table_results)
            if not any_success:
                exec_error = "All per-table executions failed."
        else:
            # Single table: original behaviour.
            exec_result = run_plan(plan=plan, csv_path=primary_csv)
            trace_table = exec_result.trace_as_table()
            if exec_result.error:
                exec_error = exec_result.error
                validator_result_repr = (
                    f"Execution failed at step {exec_result.failed_step}: "
                    f"{exec_error}"
                )
            else:
                validator_result_repr = _repr_df(exec_result.final_df)
    else:
        exec_error = "Planner did not return a valid plan."
        validator_result_repr = (
            f"Plan generation failed. Raw response: "
            f"{planner_response.get('raw', str(planner_response))[:400]}"
        )

    # Surface unresolved-column warning to the verdict stage.
    if column_fix_note:
        exec_error = (column_fix_note + ("\n" + exec_error if exec_error else ""))

    # ------------------------------------------------------------------
    # Step 4: Generate verdict (LLM)
    # ------------------------------------------------------------------
    verdict_user = _build_verdict_prompt(
        question=question,
        analytics_summary=analytics_summary,
        validator_result_repr=validator_result_repr,
        execution_trace=trace_table,
        exec_error=exec_error,
    )
    verdict_response = _call_llm(
        client=client,
        model_name=model_name,
        system=VERDICT_SYSTEM_PROMPT,
        user=verdict_user,
    )

    verdict = verdict_response.get("verdict", "RETRY")
    reason = verdict_response.get("reason", "Could not determine verdict.")
    suggestion = verdict_response.get("suggestion", "")

    # Normalise verdict
    if verdict not in ("PASS", "INCOMPLETE", "SUSPICIOUS", "RETRY"):
        verdict = "RETRY"
        reason = f"Unexpected verdict value from LLM: {verdict_response}"

    return ValidatorVerdict(
        verdict=verdict,
        reason=reason,
        suggestion=suggestion,
        validator_result=exec_result.final_df if exec_result else None,
        plan_used=plan,
        execution_trace=trace_table,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_known_columns(schemas: dict[str, dict]) -> set[str]:
    """Union of every column name across all registered tables."""
    cols: set[str] = set()
    for schema in schemas.values():
        cols.update(schema["columns"])
    return cols


def _fuzzy_match_column(name: str, known: set[str]) -> Optional[str]:
    """
    Try to map a (possibly hallucinated) column name to a real one.

    Handles the common failure modes of a weak model on this dataset:
      - case differences:   'Total Liabilities' -> 'Total liabilities'
      - whitespace/underscore: 'total_liabilities' -> 'Total liabilities'
      - near-miss spelling via difflib as a last resort.

    Returns the real column name, or None if no confident match.
    """
    if name in known:
        return name

    def norm(s: str) -> str:
        return re.sub(r"[\s_]+", " ", s.strip().lower())

    target = norm(name)
    # 1. Exact match after normalisation (case + whitespace insensitive).
    norm_map = {norm(c): c for c in known}
    if target in norm_map:
        return norm_map[target]

    # 2. difflib close match on normalised names (cutoff high to avoid
    #    mapping 'debt' to an unrelated column).
    import difflib
    candidates = list(norm_map.keys())
    close = difflib.get_close_matches(target, candidates, n=1, cutoff=0.92)
    if close:
        return norm_map[close[0]]

    return None


def _extract_plan_columns(plan: dict) -> list[str]:
    """
    Walk a plan dict and collect every column name that REFERENCES the input
    schema. Columns produced WITHIN the plan (via derive 'new_column',
    aggregate 'as', or rank 'as') are tracked separately and excluded from
    schema validation, since they legitimately won't exist in the raw data.
    """
    referenced: list[str] = []
    produced: set[str] = set()   # column names created by the plan itself

    def visit_expr(expr: Any) -> None:
        if isinstance(expr, dict):
            if expr.get("type") == "column" and "value" in expr:
                referenced.append(expr["value"])
            for v in expr.values():
                visit_expr(v)
        elif isinstance(expr, list):
            for item in expr:
                visit_expr(item)

    for step in plan.get("steps", []):
        op = step.get("op", "")
        if op == "derive_columns":
            for d in step.get("derive", []):
                visit_expr(d.get("expr"))
                if d.get("new_column"):
                    produced.add(d["new_column"])
        elif op == "filter_rows":
            for c in step.get("conditions", []):
                if "column" in c:
                    referenced.append(c["column"])
        elif op == "group_and_aggregate":
            referenced.extend(step.get("group_by", []))
            for m in step.get("metrics", []):
                if m.get("column") and m["column"] != "*":
                    referenced.append(m["column"])
                if m.get("as"):
                    produced.add(m["as"])
        elif op == "sort_rows":
            for s in step.get("sort_by", []):
                if "column" in s:
                    referenced.append(s["column"])
        elif op == "select_columns":
            referenced.extend(step.get("columns", []))
        elif op == "distinct_rows":
            referenced.extend(step.get("columns", []) or [])
        elif op == "dropna":
            referenced.extend(step.get("columns", []) or [])
        elif op == "rank":
            if step.get("column"):
                referenced.append(step["column"])
            referenced.extend(step.get("partition_by", []) or [])
            if step.get("as"):
                produced.add(step["as"])
        elif op in ("join_tables",):
            referenced.extend(step.get("on", []) or [])
            referenced.extend(step.get("left_on", []) or [])
            referenced.extend(step.get("right_on", []) or [])
        elif op == "add_literal_column":
            if step.get("column"):
                produced.add(step["column"])

    # Exclude plan-produced columns from validation against the raw schema.
    return [c for c in referenced if c not in produced]


def _validate_and_fix_plan_columns(
    plan: dict,
    schemas: dict[str, dict],
) -> tuple[dict, list[str]]:
    """
    Check every column the plan references against the real schema.
    Fuzzy-fix what we can; report what we cannot.

    Returns (possibly_corrected_plan, unresolved_column_names).
    A non-empty unresolved list means the plan should NOT be trusted —
    the caller can treat it like a plan-generation failure and retry.
    """
    known = _all_known_columns(schemas)
    referenced = _extract_plan_columns(plan)

    rename: dict[str, str] = {}
    unresolved: list[str] = []
    for col in referenced:
        if col in known:
            continue
        fixed = _fuzzy_match_column(col, known)
        if fixed is not None:
            rename[col] = fixed
        else:
            unresolved.append(col)

    # Apply renames by serialising → string-replace → reparse. Safe because
    # column names are JSON string values; we replace exact quoted tokens.
    if rename:
        plan_str = json.dumps(plan)
        for wrong, right in rename.items():
            plan_str = plan_str.replace(
                json.dumps(wrong), json.dumps(right)
            )
        plan = json.loads(plan_str)

    return plan, unresolved


def _path_to_table_name(path: str, index: int) -> str:
    """
    Derive a clean table name from a file path.
    e.g. "datasets/2014_Financial_Data.csv" -> "df_2014"
         "datasets/2017_Financial_Data.csv" -> "df_2017"
    Falls back to "df_0", "df_1", ... if no year found.
    """
    import os
    import re as _re
    basename = os.path.basename(path)
    year_match = _re.search(r"(20\d{2})", basename)
    if year_match:
        return f"df_{year_match.group(1)}"
    return f"df_{index}"


# ---------------------------------------------------------------------------
# Convenience: plain dict output
# ---------------------------------------------------------------------------

def run_validator_workflow(
    question: str,
    csv_paths: list[str],
    analytics_output: dict,
    model_name: str,
    api_key: str,
) -> dict:
    """
    Thin wrapper returning a plain dict:
    {
        "verdict":          "PASS" | "INCOMPLETE" | "SUSPICIOUS" | "RETRY",
        "reason":           str,
        "suggestion":       str,
        "plan_used":        dict | None,
        "execution_trace":  str | None,
    }
    """
    result = run_validator(
        question=question,
        csv_paths=csv_paths,
        analytics_output=analytics_output,
        model_name=model_name,
        api_key=api_key,
    )
    return {
        "verdict": result.verdict,
        "reason": result.reason,
        "suggestion": result.suggestion,
        "plan_used": result.plan_used,
        "execution_trace": result.execution_trace,
    }
