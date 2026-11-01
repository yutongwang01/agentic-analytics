"""Plan executor: applies a sequence of operators to a DataFrame.

The executor is intentionally simple and deterministic:
- It loads the CSV (auto-converting date-like columns to datetime).
- It dispatches each step to the corresponding operator in `OPERATORS`.
- It records an execution trace (step / op / in_rows / out_rows).
- It never inspects the natural-language question and never calls an LLM.

Any malformed plan or runtime error from an operator surfaces as an
`OperatorError`; the executor catches it and returns a structured failure
(so the caller can show it to the user / planner without crashing the agent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from operators import OPERATORS, OperatorError


# ---------------------------------------------------------------------------
# Trace + result containers
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    step: int
    op: str
    in_rows: int
    out_rows: int
    in_cols: int
    out_cols: int
    note: str = ""               # filled on error

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "op": self.op,
            "in_rows": self.in_rows,
            "out_rows": self.out_rows,
            "in_cols": self.in_cols,
            "out_cols": self.out_cols,
            "note": self.note,
        }


@dataclass
class ExecutionResult:
    final_df: pd.DataFrame | None
    trace: list[TraceEntry] = field(default_factory=list)
    error: str | None = None
    failed_step: int | None = None        # 1-indexed step number that failed

    def trace_as_table(self) -> str:
        """Pretty-print the trace as a fixed-width table (for logs / reports)."""
        header = f"{'Step':<5} {'Operation':<22} {'In Rows':>9} {'Out Rows':>10} {'In Cols':>8} {'Out Cols':>9}  Note"
        lines = [header, "-" * len(header)]
        for t in self.trace:
            lines.append(
                f"{t.step:<5} {t.op:<22} {t.in_rows:>9} {t.out_rows:>10} "
                f"{t.in_cols:>8} {t.out_cols:>9}  {t.note}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV loading with datetime auto-detection
# ---------------------------------------------------------------------------

def _looks_like_date_column(name: str) -> bool:
    n = name.lower()
    return ("date" in n) or ("time" in n) or n.endswith("_at") or n.endswith("_dt")


def load_csv(csv_path: str) -> pd.DataFrame:
    """Read the CSV and auto-convert columns whose name suggests a date.

    Non-parseable values become NaT. This keeps plans simple: planners never
    need an explicit cast step.
    """
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if _looks_like_date_column(col) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                # Leave as-is if conversion blows up; operator will complain later.
                pass
    return df


def infer_schema(df: pd.DataFrame, preview_rows: int = 3) -> dict:
    """Compact schema dict used by the planner prompt."""
    preview = df.head(preview_rows).where(pd.notna(df.head(preview_rows)), None)
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "preview_rows": preview.to_dict(orient="records"),
        "shape": list(df.shape),
    }


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------

def run_plan(plan: dict, csv_path: str) -> ExecutionResult:
    """Execute a plan and return final DataFrame + trace.

    On operator failure, returns an `ExecutionResult` with `error` set; the
    trace will include every step up to and including the failed one.
    """
    if not isinstance(plan, dict) or "steps" not in plan:
        return ExecutionResult(
            final_df=None,
            error=f"Plan is malformed: expected dict with 'steps' key, got {type(plan).__name__}",
            failed_step=0,
        )

    steps = plan["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        return ExecutionResult(
            final_df=None,
            error="Plan has no steps. A valid plan must contain at least one operation.",
            failed_step=0,
        )

    df = load_csv(csv_path)
    trace: list[TraceEntry] = []

    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return ExecutionResult(
                final_df=df,
                trace=trace,
                error=f"Step {idx} is not a dict.",
                failed_step=idx,
            )
        op_name = step.get("op")
        if op_name not in OPERATORS:
            return ExecutionResult(
                final_df=df,
                trace=trace,
                error=f"Step {idx}: unknown operator '{op_name}'. allowed: {list(OPERATORS)}",
                failed_step=idx,
            )

        in_rows, in_cols = df.shape
        try:
            df = OPERATORS[op_name](df, step)
        except OperatorError as exc:
            trace.append(TraceEntry(
                step=idx, op=op_name,
                in_rows=in_rows, out_rows=in_rows,
                in_cols=in_cols, out_cols=in_cols,
                note=f"ERROR: {exc}",
            ))
            return ExecutionResult(
                final_df=None,
                trace=trace,
                error=f"Step {idx} ({op_name}) failed: {exc}",
                failed_step=idx,
            )
        except Exception as exc:  # safety net for non-OperatorError bugs
            trace.append(TraceEntry(
                step=idx, op=op_name,
                in_rows=in_rows, out_rows=in_rows,
                in_cols=in_cols, out_cols=in_cols,
                note=f"INTERNAL ERROR: {type(exc).__name__}: {exc}",
            ))
            return ExecutionResult(
                final_df=None,
                trace=trace,
                error=f"Step {idx} ({op_name}) raised unexpected error: {exc}",
                failed_step=idx,
            )

        out_rows, out_cols = df.shape
        trace.append(TraceEntry(
            step=idx, op=op_name,
            in_rows=in_rows, out_rows=out_rows,
            in_cols=in_cols, out_cols=out_cols,
        ))

    return ExecutionResult(final_df=df, trace=trace)
