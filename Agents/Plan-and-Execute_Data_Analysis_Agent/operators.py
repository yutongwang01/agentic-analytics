"""Deterministic table operators for the plan-and-execute agent.

Each operator:
- takes a pandas DataFrame and a structured `step` dict,
- returns a new DataFrame,
- raises `OperatorError` on any malformed input (never silently),
- does NOT interpret natural language.

The expression AST (`_eval_expr`) is the only place where "values" are computed.
It is intentionally limited to:
- column references         : {"type": "column", "value": "col_name"}
- literals                  : {"type": "literal", "value": 3.14}
- arithmetic                : {"type": "arithmetic", "operation": "add"|"subtract"|"multiply"|"divide", "left": ..., "right": ...}
- date_part extraction      : {"type": "date_part", "column": "invoice_date", "part": "year"|"month"|"quarter"|"day_of_week"|"year_month"|"date"}

No string parsing, no eval(), no NL.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OperatorError(Exception):
    """Raised when an operator receives malformed input or hits a runtime error.

    The error message must be precise enough that the planner (or a human
    reading the failure trace) can fix the plan without inspecting Python tracebacks.
    """


# ---------------------------------------------------------------------------
# Expression AST evaluator
# ---------------------------------------------------------------------------

_ARITH_OPS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    "add":      lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide":   lambda a, b: a / b.replace(0, np.nan),  # avoid div-by-zero -> NaN
}

_DATE_PARTS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "year":        lambda s: s.dt.year,
    "month":       lambda s: s.dt.month,
    "quarter":     lambda s: s.dt.quarter,
    "day_of_week": lambda s: s.dt.dayofweek,            # 0 = Monday
    "year_month":  lambda s: s.dt.strftime("%Y-%m"),    # string, sortable
    "date":        lambda s: s.dt.date.astype("string"),
}

# Comparison operators: produce boolean Series.
_CMP_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "eq":  lambda a, b: a == b,
    "ne":  lambda a, b: a != b,
    "gt":  lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt":  lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}

# Logical operators: combine boolean Series.
_LOGIC_BIN_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "and": lambda a, b: a & b,
    "or":  lambda a, b: a | b,
}


def _to_bool_series(value: Any, df: pd.DataFrame) -> pd.Series:
    """Coerce a scalar bool or pandas Series into a boolean Series aligned to df."""
    if isinstance(value, pd.Series):
        return value.astype(bool)
    # scalar -> broadcast
    return pd.Series([bool(value)] * len(df), index=df.index)


def _eval_expr(expr: Any, df: pd.DataFrame) -> pd.Series | Any:
    """Recursively evaluate an expression node against `df`.

    Supported node types:
      - literal      : {"type": "literal",  "value": <scalar>}
      - column       : {"type": "column",   "value": "<col>"}
      - arithmetic   : {"type": "arithmetic",  "operation": "add|subtract|multiply|divide",
                        "left": <expr>, "right": <expr>}
      - compare      : {"type": "compare",     "operation": "eq|ne|gt|gte|lt|lte",
                        "left": <expr>, "right": <expr>}                  -> bool Series
      - logical      : {"type": "logical",     "operation": "and|or",
                        "left": <expr>, "right": <expr>}                  -> bool Series
                       {"type": "logical",     "operation": "not",
                        "operand": <expr>}                                -> bool Series
      - date_part    : {"type": "date_part",   "column": "<col>",
                        "part": "year|month|quarter|day_of_week|year_month|date"}
      - case_when    : {"type": "case_when",
                        "cases": [{"when": <bool_expr>, "then": <expr>}, ...],
                        "else":  <expr>}                                  -> Series
    """
    if not isinstance(expr, dict):
        raise OperatorError(f"expression must be a dict, got {type(expr).__name__}: {expr!r}")

    etype = expr.get("type")
    if etype is None:
        raise OperatorError(f"expression missing 'type': {expr!r}")

    if etype == "literal":
        if "value" not in expr:
            raise OperatorError(f"literal expression missing 'value': {expr!r}")
        return expr["value"]

    if etype == "column":
        col = expr.get("value")
        if not isinstance(col, str):
            raise OperatorError(f"column expression needs string 'value': {expr!r}")
        if col not in df.columns:
            raise OperatorError(f"column '{col}' not found. available: {list(df.columns)}")
        return df[col]

    if etype == "arithmetic":
        op = expr.get("operation")
        if op not in _ARITH_OPS:
            raise OperatorError(f"unknown arithmetic op '{op}'. allowed: {list(_ARITH_OPS)}")
        left = _eval_expr(expr.get("left"), df)
        right = _eval_expr(expr.get("right"), df)
        try:
            return _ARITH_OPS[op](left, right)
        except Exception as exc:
            raise OperatorError(f"arithmetic '{op}' failed: {exc}") from exc

    if etype == "compare":
        op = expr.get("operation")
        if op not in _CMP_OPS:
            raise OperatorError(f"unknown compare op '{op}'. allowed: {list(_CMP_OPS)}")
        left = _eval_expr(expr.get("left"), df)
        right = _eval_expr(expr.get("right"), df)
        try:
            result = _CMP_OPS[op](left, right)
        except Exception as exc:
            raise OperatorError(f"compare '{op}' failed: {exc}") from exc
        # Always return a bool Series so downstream logical ops work cleanly.
        return _to_bool_series(result, df)

    if etype == "logical":
        op = expr.get("operation")
        if op == "not":
            operand = _eval_expr(expr.get("operand"), df)
            return ~_to_bool_series(operand, df)
        if op not in _LOGIC_BIN_OPS:
            raise OperatorError(f"unknown logical op '{op}'. allowed: and, or, not")
        left = _to_bool_series(_eval_expr(expr.get("left"), df), df)
        right = _to_bool_series(_eval_expr(expr.get("right"), df), df)
        try:
            return _LOGIC_BIN_OPS[op](left, right)
        except Exception as exc:
            raise OperatorError(f"logical '{op}' failed: {exc}") from exc

    if etype == "date_part":
        col = expr.get("column")
        part = expr.get("part")
        if col not in df.columns:
            raise OperatorError(f"date_part: column '{col}' not found")
        if part not in _DATE_PARTS:
            raise OperatorError(f"date_part: unknown part '{part}'. allowed: {list(_DATE_PARTS)}")
        series = df[col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            raise OperatorError(
                f"date_part: column '{col}' is not datetime (dtype={series.dtype}). "
                "Executor should auto-convert date-like columns; check loader."
            )
        return _DATE_PARTS[part](series)

    if etype == "case_when":
        cases = expr.get("cases")
        else_branch = expr.get("else")
        if not isinstance(cases, list) or not cases:
            raise OperatorError("case_when: 'cases' must be a non-empty list of {when, then}")
        if else_branch is None:
            raise OperatorError("case_when: 'else' branch is required")

        # Build a result Series starting from the else branch, then overwrite
        # for each case (later cases overwrite earlier when the masks overlap;
        # to make EARLIER cases win we apply them in reverse).
        else_val = _eval_expr(else_branch, df)
        if isinstance(else_val, pd.Series):
            result = else_val.copy()
        else:
            result = pd.Series([else_val] * len(df), index=df.index)

        for i, case in enumerate(reversed(cases)):
            if not isinstance(case, dict) or "when" not in case or "then" not in case:
                raise OperatorError(
                    f"case_when.cases[{len(cases) - 1 - i}]: each case needs 'when' and 'then'"
                )
            mask = _to_bool_series(_eval_expr(case["when"], df), df)
            then_val = _eval_expr(case["then"], df)
            if isinstance(then_val, pd.Series):
                result = result.where(~mask, then_val)
            else:
                result = result.where(~mask, pd.Series([then_val] * len(df), index=df.index))
        return result

    raise OperatorError(
        f"unknown expression type '{etype}'. allowed: literal, column, arithmetic, "
        "compare, logical, date_part, case_when"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(step: dict, key: str, op_name: str) -> Any:
    if key not in step:
        raise OperatorError(f"{op_name}: missing required field '{key}'")
    return step[key]


def _check_columns_exist(df: pd.DataFrame, cols: list[str], op_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise OperatorError(
            f"{op_name}: column(s) {missing} not found. available: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Operator: derive_columns
# ---------------------------------------------------------------------------

def op_derive_columns(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Add new columns computed from an expression AST.

    Step format:
      {
        "op": "derive_columns",
        "derive": [
          {"new_column": "revenue", "expr": <expr>},
          ...
        ]
      }

    Legacy / spec-style format (matches HW3 PDF example):
      {"new_column": "yield", "type": "arithmetic", "operation": "divide",
       "left": {...}, "right": {...}}
    is also accepted: if no "expr" key is present, the rest of the dict (minus
    "new_column") is treated as the expression itself.
    """
    derive_list = _require(step, "derive", "derive_columns")
    if not isinstance(derive_list, list) or len(derive_list) == 0:
        raise OperatorError("derive_columns: 'derive' must be a non-empty list")

    out = df.copy()
    for i, item in enumerate(derive_list):
        if not isinstance(item, dict):
            raise OperatorError(f"derive_columns[{i}]: entry must be a dict")
        new_col = item.get("new_column")
        if not isinstance(new_col, str) or not new_col:
            raise OperatorError(f"derive_columns[{i}]: 'new_column' must be a non-empty string")

        if "expr" in item:
            expr = item["expr"]
        else:
            # Treat the entry itself as the expression (HW3 PDF style)
            expr = {k: v for k, v in item.items() if k != "new_column"}

        out[new_col] = _eval_expr(expr, out)
    return out


# ---------------------------------------------------------------------------
# Operator: filter_rows
# ---------------------------------------------------------------------------

_COMPARATORS: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    "==":      lambda s, v: s == v,
    "!=":      lambda s, v: s != v,
    ">":       lambda s, v: s > v,
    ">=":      lambda s, v: s >= v,
    "<":       lambda s, v: s < v,
    "<=":      lambda s, v: s <= v,
    "in":      lambda s, v: s.isin(v if isinstance(v, (list, tuple, set)) else [v]),
    "not_in":  lambda s, v: ~s.isin(v if isinstance(v, (list, tuple, set)) else [v]),
    "contains": lambda s, v: s.astype("string").str.contains(str(v), case=False, na=False),
    "is_null":  lambda s, v: s.isna(),
    "not_null": lambda s, v: s.notna(),
}


def _eval_condition(cond: dict, df: pd.DataFrame) -> pd.Series:
    """Evaluate one filter condition to a boolean mask."""
    if not isinstance(cond, dict):
        raise OperatorError(f"filter condition must be a dict, got {cond!r}")

    # Composite condition: {"logic": "and"|"or", "conditions": [...]}
    if "logic" in cond:
        logic = cond["logic"]
        sub = cond.get("conditions", [])
        if logic not in ("and", "or") or not isinstance(sub, list) or not sub:
            raise OperatorError(f"composite filter needs logic in ['and','or'] and non-empty conditions: {cond!r}")
        masks = [_eval_condition(c, df) for c in sub]
        result = masks[0]
        for m in masks[1:]:
            result = result & m if logic == "and" else result | m
        return result

    column = cond.get("column")
    op = cond.get("operator")
    value = cond.get("value", None)

    if column is None or op is None:
        raise OperatorError(f"filter condition needs 'column' and 'operator': {cond!r}")
    if column not in df.columns:
        raise OperatorError(f"filter_rows: column '{column}' not found. available: {list(df.columns)}")
    if op not in _COMPARATORS:
        raise OperatorError(f"filter_rows: unknown operator '{op}'. allowed: {list(_COMPARATORS)}")

    series = df[column]

    # Auto-coerce types for comparison when reasonable
    if op in ("==", "!=", ">", ">=", "<", "<="):
        # Datetime column with string value: parse value
        if pd.api.types.is_datetime64_any_dtype(series) and isinstance(value, str):
            try:
                value = pd.to_datetime(value)
            except Exception as exc:
                raise OperatorError(f"filter_rows: cannot parse '{value}' as datetime: {exc}") from exc
        # Numeric column with string-looking number: cast value
        elif pd.api.types.is_numeric_dtype(series) and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise OperatorError(
                    f"filter_rows: column '{column}' is numeric but value '{value}' is not a number"
                )

    try:
        return _COMPARATORS[op](series, value)
    except Exception as exc:
        raise OperatorError(f"filter_rows: comparison '{column} {op} {value!r}' failed: {exc}") from exc


def op_filter_rows(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Filter rows by a list of conditions (AND by default).

    Step format:
      {
        "op": "filter_rows",
        "conditions": [
          {"column": "yield", "operator": "<", "value": 0.9},
          {"column": "country", "operator": "in", "value": ["UK", "France"]}
        ],
        "logic": "and"   # optional, default "and"
      }
    """
    conditions = _require(step, "conditions", "filter_rows")
    if not isinstance(conditions, list) or not conditions:
        raise OperatorError("filter_rows: 'conditions' must be a non-empty list")

    logic = step.get("logic", "and")
    if logic not in ("and", "or"):
        raise OperatorError(f"filter_rows: top-level logic must be 'and' or 'or', got {logic!r}")

    masks = [_eval_condition(c, df) for c in conditions]
    final_mask = masks[0]
    for m in masks[1:]:
        final_mask = final_mask & m if logic == "and" else final_mask | m
    return df[final_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Operator: group_and_aggregate
# ---------------------------------------------------------------------------

_AGG_FUNCS: dict[str, str] = {
    # name -> pandas agg name (or special handling)
    "mean":     "mean",
    "average":  "mean",
    "avg":      "mean",
    "sum":      "sum",
    "count":    "count",
    "min":      "min",
    "max":      "max",
    "median":   "median",
    "std":      "std",
    "var":      "var",
    "nunique":  "nunique",
    "first":    "first",
    "last":     "last",
}


def op_group_and_aggregate(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Group by columns and compute aggregations.

    Step format:
      {
        "op": "group_and_aggregate",
        "group_by": ["country"],
        "metrics": [
          {"function": "sum",   "column": "revenue",    "as": "total_revenue"},
          {"function": "count", "column": "invoice_id", "as": "n_orders"},
          {"function": "count", "as": "n_rows"}                 # count(*) style
        ]
      }
    """
    group_by = _require(step, "group_by", "group_and_aggregate")
    metrics = _require(step, "metrics", "group_and_aggregate")

    if not isinstance(group_by, list):
        raise OperatorError("group_and_aggregate: 'group_by' must be a list (use [] for global aggregation)")
    if not isinstance(metrics, list) or not metrics:
        raise OperatorError("group_and_aggregate: 'metrics' must be a non-empty list")

    _check_columns_exist(df, group_by, "group_and_aggregate.group_by")

    # Validate metrics up front (do this before short-circuiting on empty input)
    for i, m in enumerate(metrics):
        func = m.get("function")
        as_name = m.get("as")
        if func not in _AGG_FUNCS:
            raise OperatorError(
                f"group_and_aggregate.metrics[{i}]: unknown function '{func}'. allowed: {list(_AGG_FUNCS)}"
            )
        if not isinstance(as_name, str) or not as_name:
            raise OperatorError(
                f"group_and_aggregate.metrics[{i}]: 'as' (output column name) is required"
            )
        col = m.get("column")
        if not (func == "count" and col is None):
            if col is None:
                raise OperatorError(
                    f"group_and_aggregate.metrics[{i}]: function '{func}' requires 'column'"
                )
            if col not in df.columns:
                raise OperatorError(
                    f"group_and_aggregate.metrics[{i}]: column '{col}' not found"
                )

    # Global aggregation: no group_by columns, produce a single-row frame
    if len(group_by) == 0:
        row: dict[str, Any] = {}
        for m in metrics:
            func = m["function"]
            as_name = m["as"]
            col = m.get("column")
            if func == "count" and col is None:
                row[as_name] = len(df)
            else:
                # Use pandas Series .agg for consistency with grouped path
                if len(df) == 0:
                    row[as_name] = 0 if func in ("count", "nunique") else pd.NA
                else:
                    row[as_name] = df[col].agg(_AGG_FUNCS[func])
        return pd.DataFrame([row])

    # Empty input + non-empty group_by: return correctly-shaped empty frame
    if len(df) == 0:
        cols = list(group_by) + [m["as"] for m in metrics]
        return pd.DataFrame(columns=cols)

    grouped = df.groupby(group_by, dropna=False)

    out_cols: dict[str, pd.Series] = {}
    for m in metrics:
        func = m["function"]
        as_name = m["as"]
        col = m.get("column")
        if func == "count" and col is None:
            series = grouped.size()
        else:
            series = grouped[col].agg(_AGG_FUNCS[func])
        out_cols[as_name] = series

    out = pd.DataFrame(out_cols).reset_index()
    return out


# ---------------------------------------------------------------------------
# Operator: sort_rows
# ---------------------------------------------------------------------------

def op_sort_rows(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Sort by one or more columns.

    Step format:
      {
        "op": "sort_rows",
        "sort_by": [
          {"column": "total_revenue", "direction": "desc"},
          {"column": "country",       "direction": "asc"}
        ]
      }
    """
    sort_by = _require(step, "sort_by", "sort_rows")
    if not isinstance(sort_by, list) or not sort_by:
        raise OperatorError("sort_rows: 'sort_by' must be a non-empty list")

    cols, ascending = [], []
    for i, item in enumerate(sort_by):
        if not isinstance(item, dict):
            raise OperatorError(f"sort_rows.sort_by[{i}]: must be a dict")
        col = item.get("column")
        direction = item.get("direction", "asc")
        if col is None:
            raise OperatorError(f"sort_rows.sort_by[{i}]: missing 'column'")
        if direction not in ("asc", "desc"):
            raise OperatorError(f"sort_rows.sort_by[{i}]: direction must be 'asc' or 'desc'")
        cols.append(col)
        ascending.append(direction == "asc")

    _check_columns_exist(df, cols, "sort_rows")
    return df.sort_values(by=cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Operator: limit_rows
# ---------------------------------------------------------------------------

def op_limit_rows(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Keep the first k rows.

    Step format:  {"op": "limit_rows", "k": 10}
    """
    k = _require(step, "k", "limit_rows")
    if not isinstance(k, int) or k < 0:
        raise OperatorError(f"limit_rows: 'k' must be a non-negative integer, got {k!r}")
    return df.head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Operator: select_columns
# ---------------------------------------------------------------------------

def op_select_columns(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Project a subset of columns.

    Step format: {"op": "select_columns", "columns": ["country", "total"]}
    """
    cols = _require(step, "columns", "select_columns")
    if not isinstance(cols, list) or not cols:
        raise OperatorError("select_columns: 'columns' must be a non-empty list")
    _check_columns_exist(df, cols, "select_columns")
    return df[cols].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Operator: distinct_rows
# ---------------------------------------------------------------------------

def op_distinct_rows(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Drop duplicate rows.

    Step format:
      {"op": "distinct_rows"}                              # all columns
      {"op": "distinct_rows", "columns": ["country"]}      # subset
    """
    cols = step.get("columns")
    if cols is None:
        return df.drop_duplicates().reset_index(drop=True)
    if not isinstance(cols, list) or not cols:
        raise OperatorError("distinct_rows: 'columns' must be a non-empty list when provided")
    _check_columns_exist(df, cols, "distinct_rows")
    return df.drop_duplicates(subset=cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bonus operators
# ---------------------------------------------------------------------------

def op_dropna(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Drop rows with NaN in the specified columns (or any if unspecified).

    Step format:
      {"op": "dropna"}
      {"op": "dropna", "columns": ["customer_id"]}
    """
    cols = step.get("columns")
    if cols is None:
        return df.dropna().reset_index(drop=True)
    if not isinstance(cols, list) or not cols:
        raise OperatorError("dropna: 'columns' must be a non-empty list when provided")
    _check_columns_exist(df, cols, "dropna")
    return df.dropna(subset=cols).reset_index(drop=True)


def op_rank(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Add a rank column based on one column, optionally within partitions.

    Step format:
      {
        "op": "rank",
        "column": "total_revenue",
        "as": "revenue_rank",
        "direction": "desc",                # default "desc"
        "method": "dense",                  # default "dense"; pandas rank methods
        "partition_by": ["country"]         # optional
      }
    """
    col = _require(step, "column", "rank")
    as_name = _require(step, "as", "rank")
    direction = step.get("direction", "desc")
    method = step.get("method", "dense")
    partition_by = step.get("partition_by")

    _check_columns_exist(df, [col], "rank")
    if direction not in ("asc", "desc"):
        raise OperatorError("rank: direction must be 'asc' or 'desc'")
    ascending = (direction == "asc")

    out = df.copy()
    if partition_by:
        if not isinstance(partition_by, list) or not partition_by:
            raise OperatorError("rank: 'partition_by' must be a non-empty list when provided")
        _check_columns_exist(df, partition_by, "rank.partition_by")
        out[as_name] = (
            out.groupby(partition_by, dropna=False)[col]
               .rank(method=method, ascending=ascending)
               .astype("int64", errors="ignore")
        )
    else:
        out[as_name] = out[col].rank(method=method, ascending=ascending).astype("int64", errors="ignore")
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPERATORS: dict[str, Callable[[pd.DataFrame, dict], pd.DataFrame]] = {
    # required
    "derive_columns":      op_derive_columns,
    "filter_rows":         op_filter_rows,
    "group_and_aggregate": op_group_and_aggregate,
    "sort_rows":           op_sort_rows,
    "limit_rows":          op_limit_rows,
    "select_columns":      op_select_columns,
    "distinct_rows":       op_distinct_rows,
    # bonus
    "dropna":              op_dropna,
    "rank":                op_rank,
}
