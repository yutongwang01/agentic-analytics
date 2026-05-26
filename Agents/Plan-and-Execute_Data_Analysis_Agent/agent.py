"""Plan-and-Execute Data Analysis Agent (LangGraph orchestration).

Pipeline:
    question --> plan (LLM, JSON only)
              --> execute (deterministic operators, builds trace)
              --> check   (deterministic: did the plan run and produce something usable?)
              --> [retry once on FAIL]
              --> answer  (LLM, formats the final DataFrame into natural language)
"""

from __future__ import annotations

import json
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from planner import generate_plan, load_env_file
from executor import run_plan, ExecutionResult

# `format_final_answer` is reused from the HW2 nodes module style. We re-implement
# a lean version here so this project has no dependency on HW2's nodes.py.
from openai import OpenAI
import pandas as pd


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # inputs
    question: str
    csv_path: str
    model_name: str
    api_key: str
    # planner
    plan: dict
    plan_raw: str
    plan_error: str | None
    # executor
    execution: Any           # ExecutionResult
    # check
    check_status: str        # "PASS" | "FAIL"
    check_reason: str
    retry_count: int
    # output
    final_answer: str


# ---------------------------------------------------------------------------
# Answer formatter (LLM, but only on the final table — never inside execution)
# ---------------------------------------------------------------------------

def _result_to_repr(df: pd.DataFrame | None, max_rows: int = 15) -> str:
    if df is None:
        return "(no result table)"
    if len(df) == 0:
        return f"Empty DataFrame with columns: {list(df.columns)}"
    shown = df.head(max_rows)
    return (
        f"Shape: {df.shape}, columns: {list(df.columns)}\n"
        f"Rows (first {min(max_rows, len(df))}):\n{shown.to_string(index=False)}"
    )


def _format_answer(
    question: str,
    df: pd.DataFrame | None,
    error: str | None,
    model_name: str,
    api_key: str,
) -> str:
    if error is not None or df is None:
        return f"Execution failed: {error or 'no result produced'}"

    client = OpenAI(api_key=api_key)
    prompt = f"""You are a data analyst. Given a user question and the computed result table,
write a clear, concise natural language answer.

Question: {question}

Computed Result:
{_result_to_repr(df)}

Instructions:
- Answer the question directly using the data above.
- Be specific: include actual numbers, names, or values.
- Keep it to 1-3 sentences.
- Do not mention pandas, code, plans, or DataFrames.
- If the table is empty, say no records match the question's conditions.
"""
    response = client.responses.create(model=model_name, input=prompt, temperature=0)
    return response.output_text.strip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_plan(state: AgentState) -> dict:
    try:
        plan, raw = generate_plan(
            question=state["question"],
            csv_path=state["csv_path"],
            model_name=state["model_name"],
            api_key=state["api_key"],
        )
        return {"plan": plan, "plan_raw": raw, "plan_error": None}
    except Exception as exc:  # noqa: BLE001
        return {"plan": {}, "plan_raw": "", "plan_error": f"{type(exc).__name__}: {exc}"}


def node_execute(state: AgentState) -> dict:
    if state.get("plan_error"):
        # Planner failed; produce an empty execution result so downstream nodes
        # have a consistent shape to work with.
        empty = ExecutionResult(final_df=None, trace=[], error=state["plan_error"], failed_step=0)
        return {"execution": empty}
    exec_result = run_plan(state["plan"], state["csv_path"])
    return {"execution": exec_result}


def node_check(state: AgentState) -> dict:
    """Deterministic check: did the plan run, and is the result table usable?

    Pass conditions:
      - planner produced a plan,
      - executor completed without error,
      - final_df is a DataFrame with >=1 column.
    """
    if state.get("plan_error"):
        return {"check_status": "FAIL", "check_reason": f"planner error: {state['plan_error']}"}

    exec_result: ExecutionResult = state["execution"]
    if exec_result.error is not None:
        return {"check_status": "FAIL", "check_reason": exec_result.error}
    if exec_result.final_df is None:
        return {"check_status": "FAIL", "check_reason": "executor produced no DataFrame"}
    if exec_result.final_df.shape[1] == 0:
        return {"check_status": "FAIL", "check_reason": "final table has zero columns"}
    # An empty result table (0 rows) is allowed; it's a valid answer.
    return {"check_status": "PASS", "check_reason": "ok"}


def node_retry(state: AgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def node_answer(state: AgentState) -> dict:
    exec_result: ExecutionResult = state.get("execution")
    df = exec_result.final_df if exec_result else None
    err = state.get("check_reason") if state.get("check_status") == "FAIL" else None
    answer = _format_answer(
        question=state["question"],
        df=df,
        error=err,
        model_name=state["model_name"],
        api_key=state["api_key"],
    )
    return {"final_answer": answer}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

MAX_RETRIES = 1


def route_after_check(state: AgentState) -> str:
    if state.get("check_status") == "FAIL" and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "answer"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan",    node_plan)
    g.add_node("execute", node_execute)
    g.add_node("check",   node_check)
    g.add_node("retry",   node_retry)
    g.add_node("answer",  node_answer)

    g.set_entry_point("plan")
    g.add_edge("plan",    "execute")
    g.add_edge("execute", "check")
    g.add_conditional_edges("check", route_after_check,
                            {"retry": "retry", "answer": "answer"})
    g.add_edge("retry",   "plan")      # loop back to re-plan
    g.add_edge("answer",  END)
    return g.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_workflow(question: str, csv_path: str) -> dict:
    """Run the full plan-and-execute pipeline for one question.

    Returns a dict with:
      - question
      - plan              (dict)
      - plan_raw          (str, the raw LLM output)
      - execution_trace   (list of dicts)
      - final_table       (DataFrame or None)
      - check_status      ("PASS"|"FAIL")
      - check_reason
      - final_answer      (str)
    """
    config = load_env_file()
    api_key = config.get("OPENAI_API_KEY", "")
    model_name = config.get("MODEL_NAME", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing from .env")
    if not model_name:
        raise ValueError("MODEL_NAME missing from .env")

    initial: AgentState = {
        "question": question,
        "csv_path": csv_path,
        "model_name": model_name,
        "api_key": api_key,
        "retry_count": 0,
    }
    final = _graph.invoke(initial)

    exec_result: ExecutionResult | None = final.get("execution")
    return {
        "question":        question,
        "plan":            final.get("plan", {}),
        "plan_raw":        final.get("plan_raw", ""),
        "execution_trace": [t.as_dict() for t in (exec_result.trace if exec_result else [])],
        "trace_table":     exec_result.trace_as_table() if exec_result else "",
        "final_table":     exec_result.final_df if exec_result else None,
        "check_status":    final.get("check_status", "FAIL"),
        "check_reason":    final.get("check_reason", ""),
        "final_answer":    final.get("final_answer", ""),
    }


# ---------------------------------------------------------------------------
# CLI for ad-hoc testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the plan-and-execute agent on one question.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--csv", required=True, help="path to dataset CSV")
    args = parser.parse_args()

    result = run_workflow(args.question, args.csv)
    print("=" * 80)
    print("QUESTION:", result["question"])
    print("=" * 80)
    print("PLAN:")
    print(json.dumps(result["plan"], indent=2, ensure_ascii=False))
    print("=" * 80)
    print("EXECUTION TRACE:")
    print(result["trace_table"])
    print("=" * 80)
    print(f"CHECK: {result['check_status']} ({result['check_reason']})")
    print("=" * 80)
    print("FINAL ANSWER:")
    print(result["final_answer"])
