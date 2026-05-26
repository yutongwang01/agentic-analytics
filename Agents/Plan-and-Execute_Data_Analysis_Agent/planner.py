"""LLM planner: turns a natural-language question into a structured JSON plan.

The planner is the ONLY component that "reasons" about the question.
The executor that consumes the plan never touches natural language.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from executor import infer_schema, load_csv


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_env_file(env_path: str = ".env") -> dict:
    """Load simple KEY=VALUE pairs from a .env file."""
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
# Operator catalog (shown to the LLM)
# ---------------------------------------------------------------------------

OPERATOR_SPEC = """\
You may use ONLY the following operators. Use exactly these names. Do not invent fields.

1. derive_columns       — add new columns from an expression AST.
   {
     "op": "derive_columns",
     "derive": [
       {"new_column": "<name>", "expr": <expression>}, ...
     ]
   }

2. filter_rows          — keep rows matching conditions (AND by default).
   {
     "op": "filter_rows",
     "conditions": [
       {"column": "<col>", "operator": "<cmp>", "value": <v>}, ...
     ],
     "logic": "and" | "or"        // optional, default "and"
   }
   Comparator values: "==", "!=", ">", ">=", "<", "<=", "in", "not_in",
                      "contains", "is_null", "not_null".
   For "in"/"not_in", value must be a list.
   For "is_null"/"not_null", value is ignored (use null).

3. group_and_aggregate  — group by columns and compute aggregates.
   {
     "op": "group_and_aggregate",
     "group_by": ["<col>", ...],    // [] = global aggregation (single-row result)
     "metrics": [
       {"function": "<fn>", "column": "<col>", "as": "<out_name>"}, ...
     ]
   }
   Functions: mean, sum, count, min, max, median, std, var, nunique, first, last.
   Special: {"function": "count", "as": "n_rows"}  (no "column") = COUNT(*).
   The "as" field is REQUIRED for every metric.

4. sort_rows            — sort by one or more columns.
   {
     "op": "sort_rows",
     "sort_by": [
       {"column": "<col>", "direction": "asc"|"desc"}, ...
     ]
   }

5. limit_rows           — keep the first k rows.
   {"op": "limit_rows", "k": <int>}

6. select_columns       — project a subset of columns.
   {"op": "select_columns", "columns": ["<col>", ...]}

7. distinct_rows        — drop duplicate rows.
   {"op": "distinct_rows"}                                  // all columns
   {"op": "distinct_rows", "columns": ["<col>", ...]}       // subset

8. dropna               — drop rows with NaN.
   {"op": "dropna"}                                         // any column
   {"op": "dropna", "columns": ["<col>", ...]}              // subset

9. rank                 — add a rank column (optionally within partitions).
   {
     "op": "rank",
     "column": "<col>",
     "as": "<out_name>",
     "direction": "asc"|"desc",                  // default "desc"
     "method": "dense"|"min"|"max"|"average",    // default "dense"
     "partition_by": ["<col>", ...]              // optional
   }

EXPRESSION AST (used inside derive_columns "expr"):
- Column reference  : {"type": "column",   "value": "<col_name>"}
- Literal value     : {"type": "literal",  "value": <number|string|bool>}
- Arithmetic        : {"type": "arithmetic", "operation": "add"|"subtract"|"multiply"|"divide",
                       "left": <expr>, "right": <expr>}
- Comparison        : {"type": "compare",    "operation": "eq"|"ne"|"gt"|"gte"|"lt"|"lte",
                       "left": <expr>, "right": <expr>}                 -> boolean
- Logical           : {"type": "logical",    "operation": "and"|"or",
                       "left": <expr>, "right": <expr>}                 -> boolean
                      {"type": "logical",    "operation": "not",
                       "operand": <expr>}                               -> boolean
- Date extraction   : {"type": "date_part", "column": "<datetime_col>",
                       "part": "year"|"month"|"quarter"|"day_of_week"|"year_month"|"date"}
   ("year_month" returns a string like "2011-03"; useful for monthly trends.
    "day_of_week" is 0..6 with 0=Monday, 5=Saturday, 6=Sunday.)
- Case/when         : {"type": "case_when",
                       "cases": [{"when": <bool_expr>, "then": <expr>}, ...],
                       "else":  <expr>}
   Evaluates cases in order; the first matching "when" wins. The else branch
   is required. Use this for categorical labels (e.g. weekend vs weekday) and
   for creating boolean flags from numeric thresholds.
"""

EXAMPLES = """\
=== EXAMPLE 1 ===
Question: "What is the top test program by average yield (yield = pass / tested)?"
Plan:
{
  "steps": [
    {"op": "derive_columns",
     "derive": [{"new_column": "yield",
                 "expr": {"type": "arithmetic", "operation": "divide",
                          "left":  {"type": "column", "value": "pass_count"},
                          "right": {"type": "column", "value": "tested_count"}}}]},
    {"op": "group_and_aggregate",
     "group_by": ["test_program"],
     "metrics": [{"function": "mean", "column": "yield", "as": "avg_yield"}]},
    {"op": "sort_rows",  "sort_by": [{"column": "avg_yield", "direction": "desc"}]},
    {"op": "limit_rows", "k": 1}
  ]
}

=== EXAMPLE 2 ===
Question: "How many distinct customers placed at least one non-cancelled order in 2011?"
Plan:
{
  "steps": [
    {"op": "filter_rows",
     "conditions": [
       {"column": "is_cancel", "operator": "==", "value": false},
       {"column": "invoice_date", "operator": ">=", "value": "2011-01-01"},
       {"column": "invoice_date", "operator": "<",  "value": "2012-01-01"}
     ]},
    {"op": "dropna", "columns": ["customer_id"]},
    {"op": "group_and_aggregate",
     "group_by": ["customer_id"],
     "metrics": [{"function": "count", "as": "n_orders"}]},
    {"op": "group_and_aggregate",
     "group_by": [],
     "metrics": [{"function": "count", "as": "distinct_customers"}]}
  ]
}
(Note: a single COUNT DISTINCT can also be done with nunique on the unfiltered table:
 {"function": "nunique", "column": "customer_id", "as": "distinct_customers"}.)

=== EXAMPLE 3 ===
Question: "Show monthly revenue trend for the United Kingdom (revenue = quantity * unit_price), non-cancelled only."
Plan:
{
  "steps": [
    {"op": "filter_rows",
     "conditions": [
       {"column": "country",  "operator": "==", "value": "United Kingdom"},
       {"column": "is_cancel","operator": "==", "value": false}
     ]},
    {"op": "derive_columns",
     "derive": [
       {"new_column": "revenue",
        "expr": {"type": "arithmetic", "operation": "multiply",
                 "left":  {"type": "column", "value": "quantity"},
                 "right": {"type": "column", "value": "unit_price"}}},
       {"new_column": "year_month",
        "expr": {"type": "date_part", "column": "invoice_date", "part": "year_month"}}
     ]},
    {"op": "group_and_aggregate",
     "group_by": ["year_month"],
     "metrics": [{"function": "sum", "column": "revenue", "as": "monthly_revenue"}]},
    {"op": "sort_rows", "sort_by": [{"column": "year_month", "direction": "asc"}]}
  ]
}

=== EXAMPLE 4 (case_when for categorical labels) ===
Question: "Label each non-cancelled order's day as 'weekend' (Sat/Sun) or 'weekday', then count rows per label."
Plan:
{
  "steps": [
    {"op": "filter_rows",
     "conditions": [{"column": "is_cancel", "operator": "==", "value": false}]},
    {"op": "derive_columns",
     "derive": [
       {"new_column": "dow",
        "expr": {"type": "date_part", "column": "invoice_date", "part": "day_of_week"}},
       {"new_column": "day_type",
        "expr": {
          "type": "case_when",
          "cases": [
            {"when": {"type": "logical", "operation": "or",
                      "left":  {"type": "compare", "operation": "eq",
                                "left": {"type": "column", "value": "dow"},
                                "right": {"type": "literal", "value": 5}},
                      "right": {"type": "compare", "operation": "eq",
                                "left": {"type": "column", "value": "dow"},
                                "right": {"type": "literal", "value": 6}}},
             "then": {"type": "literal", "value": "weekend"}}
          ],
          "else": {"type": "literal", "value": "weekday"}
        }}
     ]},
    {"op": "group_and_aggregate",
     "group_by": ["day_type"],
     "metrics": [{"function": "count", "as": "n_rows"}]}
  ]
}
(Note: pandas day_of_week is 0=Monday .. 6=Sunday, so weekend = {5, 6}.)

=== EXAMPLE 5 (two-view question: top-K groups + per-group champion) ===
Question: "For each country (top 5 by total revenue), find the stock code with the
highest total revenue. Return country, stock_code, stock_revenue, country_total_revenue."

WRONG approach (do NOT do this):
  Step 1-3: group by country → get country_total_revenue → limit 10
  Step 4: filter is_cancel == false   ← BROKEN: is_cancel no longer exists after step 3
  ...

CORRECT approach — single pipeline, carry all needed keys through:
{
  "steps": [
    {"op": "filter_rows",
     "conditions": [{"column": "is_cancel", "operator": "==", "value": false}]},
    {"op": "derive_columns",
     "derive": [{"new_column": "revenue",
                 "expr": {"type": "arithmetic", "operation": "multiply",
                          "left":  {"type": "column", "value": "quantity"},
                          "right": {"type": "column", "value": "unit_price"}}}]},
    {"op": "group_and_aggregate",
     "group_by": ["country", "stock_code"],
     "metrics": [{"function": "sum", "column": "revenue", "as": "stock_revenue"}]},
    {"op": "rank",
     "column": "stock_revenue",
     "as": "stock_rank",
     "direction": "desc",
     "partition_by": ["country"]},
    {"op": "group_and_aggregate",
     "group_by": ["country"],
     "metrics": [
       {"function": "sum",  "column": "stock_revenue",  "as": "country_total_revenue"},
       {"function": "first", "column": "stock_code",     "as": "top_stock_code"},
       {"function": "first", "column": "stock_revenue",  "as": "top_stock_revenue"}
     ]},
    {"op": "sort_rows",
     "sort_by": [{"column": "country_total_revenue", "direction": "desc"}]},
    {"op": "limit_rows", "k": 5}
  ]
}

WAIT — the above is still wrong because after rank, stock_rank==1 rows are mixed with
rank==2,3... rows, so "first" would not reliably pick the top stock.
The CORRECT single-pipeline is:

{
  "steps": [
    {"op": "filter_rows",
     "conditions": [{"column": "is_cancel", "operator": "==", "value": false}]},
    {"op": "derive_columns",
     "derive": [{"new_column": "revenue",
                 "expr": {"type": "arithmetic", "operation": "multiply",
                          "left":  {"type": "column", "value": "quantity"},
                          "right": {"type": "column", "value": "unit_price"}}}]},
    {"op": "group_and_aggregate",
     "group_by": ["country", "stock_code"],
     "metrics": [
       {"function": "sum",  "column": "revenue", "as": "stock_revenue"}
     ]},
    {"op": "rank",
     "column": "stock_revenue",
     "as": "stock_rank",
     "direction": "desc",
     "partition_by": ["country"]},
    {"op": "rank",
     "column": "stock_revenue",
     "as": "country_total_rank",
     "direction": "desc",
     "partition_by": []},
    {"op": "filter_rows",
     "conditions": [{"column": "stock_rank", "operator": "==", "value": 1}]},
    {"op": "rank",
     "column": "stock_revenue",
     "as": "country_rev_rank",
     "direction": "desc",
     "partition_by": []},
    {"op": "sort_rows",
     "sort_by": [{"column": "stock_revenue", "direction": "desc"}]},
    {"op": "limit_rows", "k": 5}
  ]
}

KEY LESSONS from this example:
- group_and_aggregate by (country, stock_code) gives you BOTH the per-stock revenue
  AND implicitly the country grouping — you never need to "go back" to raw data.
- rank with partition_by=["country"] then filter stock_rank==1 selects each
  country's champion stock in a single pass.
- To get the top-K countries by total revenue, add a second rank (partition_by=[],
  i.e. global rank) on stock_revenue AFTER filtering to rank==1, then limit.
- You never re-issue filter_rows on a column (like is_cancel) that was consumed
  by an earlier group_and_aggregate.
"""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_RULES = """\
You are a data analysis PLANNER. Your only output is a JSON plan that the
deterministic executor will run on a pandas DataFrame.

HARD RULES:
1. Output exactly ONE JSON object with a top-level "steps" array. No prose, no markdown.
2. Never produce Python code, expressions as strings, or natural-language hints.
3. NEVER use vague words like "high", "low", "top", "recent", "popular" inside the plan.
   Resolve them into numeric conditions, sort+limit, or date comparisons.
4. Only reference columns that appear in the schema below. Do not invent columns.
   Derived columns must be created (in a prior step) before being referenced.
5. After group_and_aggregate, the table has ONLY: the group_by columns + the
   "as" names of each metric. Any subsequent sort/filter must use these names.
6. For top-K questions: sort first, then limit.
7. For "how many distinct X", prefer nunique over count+distinct.
8. Boolean columns (e.g. is_cancel) must be compared with JSON booleans:
   true / false (not "True" / 1).
9. For monthly/quarterly/yearly analysis, use derive_columns with
   {"type":"date_part", ...} to create a grouping column first.
10. Output JSON only. The first character of your response must be '{'.

PIPELINE EXECUTION MODEL — read carefully, this is the single most common
source of broken plans:

11. Execution is STRICTLY LINEAR. Step N can only see the DataFrame produced
    by step N-1. There is NO way to "go back" to the original CSV, "rejoin"
    earlier intermediate tables, or "preserve" a column that an earlier step
    dropped. Once you call select_columns or group_and_aggregate, only the
    columns named in that step survive.

12. If a later step needs columns from the original data (e.g. is_cancel,
    invoice_date, raw stock_code), you must CARRY those columns through
    every intermediate step. Do not select_columns down to a single id-like
    column and then expect to filter the original data again.

13. If a question conceptually requires two views of the data (e.g. "find top-K
    countries by total revenue, then within those countries find each
    country's top product"), implement it as a SINGLE end-to-end pipeline:
      a. Compute the country-level aggregation as one branch of the plan,
      b. Use rank within partition + a post-rank filter to identify the
         per-country top product on the SAME aggregated table,
      c. Combine country selection (e.g. filter on country_total_revenue or
         a separate country rank) into the same flow.
    Do NOT issue a second "filter on is_cancel" after the table has already
    been aggregated; is_cancel will no longer exist.

14. case_when is the correct way to create categorical labels (weekend vs
    weekday, high vs low, etc). Do NOT try to simulate labels by filtering
    rows and then re-deriving the label as a literal -- filtering throws
    away the rows you wanted to keep.

15. Avoid useless steps. Do not derive a column that no later step references.
    Do not select_columns immediately before another step that needs more columns.

16. TWO-VIEW QUESTIONS (e.g. "top-K groups AND per-group champion"):
    The ONLY correct pattern is a single pipeline that carries all keys together:
      a. filter_rows (apply all base conditions ONCE, at the very start),
      b. derive_columns (compute revenue or other derived values),
      c. group_and_aggregate by (group_key, sub_key) — this gives both levels at once,
      d. rank with partition_by=[group_key] → identifies the per-group champion,
      e. filter_rows where rank == 1,
      f. sort + limit to pick the top-K groups.

    NEVER split this into two phases where you aggregate → limit → then try to
    re-access the original rows. After group_and_aggregate the original columns
    are GONE. There is no "phase 2" that can read is_cancel, invoice_date, or
    any other raw column.

    Concrete anti-pattern to avoid:
      BAD:  group(country) → limit(10) → filter(is_cancel==false)  ← is_cancel gone!
      BAD:  group(country) → limit(10) → derive(country=column(country))  ← no-op!
      GOOD: filter(is_cancel) → derive(revenue) → group(country,stock_code)
              → rank(partition=country) → filter(rank==1) → sort → limit
"""


def _build_user_prompt(question: str, schema: dict, csv_path: str) -> str:
    cols = ", ".join(schema["columns"])
    dtypes = "\n".join(f"  - {c}: {t}" for c, t in schema["dtypes"].items())
    preview = "\n".join(f"  - {r}" for r in schema["preview_rows"])
    n_rows, n_cols = schema["shape"]

    return f"""\
=== DATASET ===
File   : {csv_path}
Shape  : {n_rows} rows x {n_cols} columns
Columns: {cols}
Dtypes :
{dtypes}
Preview:
{preview}

=== OPERATOR CATALOG ===
{OPERATOR_SPEC}

=== EXAMPLES ===
{EXAMPLES}

=== QUESTION ===
{question}

Now produce the plan. JSON only.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plan(
    question: str,
    csv_path: str,
    model_name: str,
    api_key: str,
) -> tuple[dict, str]:
    """Ask the LLM for a plan. Returns (plan_dict, raw_text).

    Raises ValueError if the model output cannot be parsed as JSON.
    """
    client = OpenAI(api_key=api_key)

    df = load_csv(csv_path)
    schema = infer_schema(df)

    user_prompt = _build_user_prompt(question, schema, csv_path)
    full_prompt = SYSTEM_RULES + "\n\n" + user_prompt

    response = client.responses.create(
        model=model_name,
        input=full_prompt,
        temperature=0,
    )
    raw = response.output_text.strip()

    # Strip accidental code fences
    cleaned = raw
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # drop first fence line
        lines = lines[1:]
        # drop trailing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Be lenient: find the outermost JSON object if there's leading prose
    if not cleaned.startswith("{"):
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1 and last > first:
            cleaned = cleaned[first:last + 1]

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Planner returned non-JSON output ({exc}). Raw output:\n{raw}"
        ) from exc

    if not isinstance(plan, dict) or "steps" not in plan:
        raise ValueError(f"Planner output missing 'steps' key. Got: {plan!r}")

    return plan, raw
