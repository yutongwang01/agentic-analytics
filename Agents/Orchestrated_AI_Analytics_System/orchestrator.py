"""
orchestrator.py 
Master orchestration node built with LangGraph.

Responsibilities:
  1. Classify question → "analytics" or "generic"
  2. Route to DuckDuckGo web search  OR  analytics agent + validator
  3. On SUSPICIOUS / INCOMPLETE / RETRY verdict → inject suggestion and re-run
  4. Hard cap: MAX_RETRIES rounds to prevent infinite loops
  5. Return unified final output dict

State machine (LangGraph StateGraph):

  START
    └─► classify
          ├─► [generic]  web_search ──────────────────────► END
          └─► [analytics] analytics_node
                              └─► validator_node
                                    ├─► [PASS]            ► END
                                    ├─► [retry budget left]► analytics_node  (loop)
                                    └─► [budget exhausted] ► END
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict

from analytics_agent import run_workflow as analytics_run
from validator_agent import run_validator_workflow

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3          # max analytics→validator loops before giving up
CLASSIFIER_MODEL = os.getenv("MODEL_NAME", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    # ── inputs ──────────────────────────────────────────────────────────────
    question: str
    csv_paths: list[str]          # dataset file paths for analytics agent
    model_name: str
    api_key: str

    # ── routing ─────────────────────────────────────────────────────────────
    question_type: str            # "analytics" | "generic"

    # ── analytics loop ──────────────────────────────────────────────────────
    analytics_output: dict        # result from analytics_agent.run_workflow()
    validator_output: dict        # result from validator_agent.run_validator_workflow()
    retry_count: int
    retry_hint: str               # suggestion injected into next analytics run

    # ── web search ──────────────────────────────────────────────────────────
    web_results: str              # raw DuckDuckGo snippets
    web_answer: str               # formatted answer with citations

    # ── final output ────────────────────────────────────────────────────────
    final_output: dict


# ---------------------------------------------------------------------------
# Node 1: classify
# ---------------------------------------------------------------------------

def node_classify(state: OrchestratorState) -> dict:
    """
    Use LLM to decide whether the question is analytics-related
    (needs dataset + code execution) or generic (needs web search).
    """
    client = OpenAI(api_key=state["api_key"])
    prompt = f"""Classify the following user question as either "analytics" or "generic".

"analytics" means:
- the question requires data analysis, computation, statistics, or insights
  from a structured dataset (CSV / financial data / tabular data)
- examples: "Find undervalued tech stocks", "What is the average ROE by sector?",
  "Compare revenue growth between 2014 and 2017"

"generic" means:
- the question is a general knowledge or factual question that does not require
  dataset analysis — it can be answered with web search
- examples: "What is retrieval-augmented generation?",
  "What caused the 2016 oil-price crash?",
  "What is NVIDIA's primary business segment?"

Reply with ONLY a JSON object on a single line:
{{"type": "analytics" | "generic", "reason": "<one sentence>"}}

Question: {state["question"]}"""

    response = client.chat.completions.create(
        model=state["model_name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    # Parse JSON; fallback to analytics on error (safer default)
    import json, re
    try:
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = json.loads(raw)
        q_type = parsed.get("type", "analytics")
    except Exception:
        match = re.search(r'"type"\s*:\s*"(\w+)"', raw)
        q_type = match.group(1) if match else "analytics"

    if q_type not in ("analytics", "generic"):
        q_type = "analytics"

    return {"question_type": q_type}


# ---------------------------------------------------------------------------
# Node 2a: web_search  (generic questions)
# ---------------------------------------------------------------------------

def node_web_search(state: OrchestratorState) -> dict:
    """Run DuckDuckGo search and format a grounded answer with citations."""
    search = DuckDuckGoSearchRun()
    try:
        raw_results = search.run(state["question"])
    except Exception as exc:
        raw_results = f"Search failed: {exc}"

    # Ask LLM to synthesize a grounded answer from search snippets
    client = OpenAI(api_key=state["api_key"])
    prompt = f"""You are a research assistant. Based on the web search results below,
write a clear and accurate answer to the user's question.

Requirements:
- Ground your answer in the search results provided.
- Mention relevant facts, figures, or context.
- At the end, list the key sources or topics found (as a "Sources:" section).
- Keep the answer to 3-5 sentences plus the sources section.

Question: {state["question"]}

Search Results:
{raw_results[:3000]}
"""
    response = client.chat.completions.create(
        model=state["model_name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    web_answer = response.choices[0].message.content.strip()

    final_output = {
        "question_type": "generic",
        "final_answer": web_answer,
        "plots": [],
        "traces": [],
        "validator": None,
        "web_results_raw": raw_results,
    }
    return {"web_results": raw_results, "web_answer": web_answer, "final_output": final_output}


# ---------------------------------------------------------------------------
# Node 2b: analytics  (analytics questions)
# ---------------------------------------------------------------------------

def node_analytics(state: OrchestratorState) -> dict:
    """
    Run the iterative analytics agent.
    On retry rounds, prepend the validator's suggestion to the question
    so the agent knows what to fix.
    """
    question = state["question"]
    hint = state.get("retry_hint", "").strip()

    # Inject retry hint into the question context on subsequent rounds
    if hint and state.get("retry_count", 0) > 0:
        question = (
            f"{question}\n\n"
            f"[RETRY INSTRUCTION — address this before finalising your answer]: {hint}"
        )

    result = analytics_run(
        question=question,
        csv_paths=state["csv_paths"],
        model_name=state["model_name"],
        api_key=state["api_key"],
    )
    return {"analytics_output": result}


# ---------------------------------------------------------------------------
# Node 3: validator
# ---------------------------------------------------------------------------

def node_validator(state: OrchestratorState) -> dict:
    result = run_validator_workflow(
        question=state["question"],
        csv_paths=state["csv_paths"],
        analytics_output=state["analytics_output"],
        model_name=state["model_name"],
        api_key=state["api_key"],
    )
    return {"validator_output": result}


# ---------------------------------------------------------------------------
# Node 4: retry_increment  (bumps counter, stores suggestion)
# ---------------------------------------------------------------------------

def node_retry_increment(state: OrchestratorState) -> dict:
    """Increment retry counter and extract suggestion for next analytics run."""
    validator = state.get("validator_output", {})
    suggestion = validator.get("suggestion", "")
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "retry_hint": suggestion,
    }


# ---------------------------------------------------------------------------
# Node 5: build_final_output  (analytics path)
# ---------------------------------------------------------------------------

def node_build_final(state: OrchestratorState) -> dict:
    """Package analytics + validator results into the unified output dict."""
    analytics = state.get("analytics_output", {})
    validator = state.get("validator_output", {})

    final_output = {
        "question_type": "analytics",
        "final_answer": analytics.get("final_answer", ""),
        "plots": analytics.get("plots", []),
        "traces": analytics.get("traces", []),
        "total_iterations": analytics.get("total_iterations", 0),
        "success": analytics.get("success", False),
        "validator": {
            "verdict": validator.get("verdict", "UNKNOWN"),
            "reason": validator.get("reason", ""),
            "suggestion": validator.get("suggestion", ""),
            "execution_trace": validator.get("execution_trace", ""),
        },
        "retry_count": state.get("retry_count", 0),
    }
    return {"final_output": final_output}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_classify(state: OrchestratorState) -> str:
    return state["question_type"]   # "analytics" | "generic"


def route_after_validator(state: OrchestratorState) -> str:
    """
    PASS                      → build_final
    INCOMPLETE/SUSPICIOUS     → retry if budget left, else build_final
    RETRY                     → retry if budget left, else build_final
    """
    verdict = state.get("validator_output", {}).get("verdict", "PASS")
    retry_count = state.get("retry_count", 0)

    if verdict == "PASS":
        return "done"
    if retry_count < MAX_RETRIES:
        return "retry"
    # Budget exhausted — accept current result
    return "done"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    builder = StateGraph(OrchestratorState)

    # Register nodes
    builder.add_node("classify",         node_classify)
    builder.add_node("web_search",       node_web_search)
    builder.add_node("analytics",        node_analytics)
    builder.add_node("validator",        node_validator)
    builder.add_node("retry_increment",  node_retry_increment)
    builder.add_node("build_final",      node_build_final)

    # Entry point
    builder.set_entry_point("classify")

    # classify → web_search or analytics
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "generic":   "web_search",
            "analytics": "analytics",
        },
    )

    # Generic path: web_search → END
    builder.add_edge("web_search", END)

    # Analytics path: analytics → validator
    builder.add_edge("analytics", "validator")

    # validator → done or retry
    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "done":  "build_final",
            "retry": "retry_increment",
        },
    )

    # retry loop: retry_increment → analytics (back to top of loop)
    builder.add_edge("retry_increment", "analytics")

    # Final packaging → END
    builder.add_edge("build_final", END)

    return builder.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_orchestrator(
    question: str,
    csv_paths: list[str],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """

    Args:
        question   : user's natural language question
        csv_paths  : list of CSV dataset file paths (used for analytics questions)
        model_name : OpenAI model name (default: MODEL_NAME from .env)
        api_key    : OpenAI API key   (default: OPENAI_API_KEY from .env)

    Returns dict with keys:
        question_type : "analytics" | "generic"
        final_answer  : str
        plots         : list of {"title": str, "plotly_json": str}
        traces        : list of iteration trace dicts  (analytics only)
        validator     : dict with verdict/reason/suggestion  (analytics only)
        retry_count   : int  (analytics only)
        web_results_raw: str  (generic only)
    """
    model_name = model_name or os.getenv("MODEL_NAME", "gpt-4.1-mini")
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    initial_state: OrchestratorState = {
        "question":        question,
        "csv_paths":       csv_paths,
        "model_name":      model_name,
        "api_key":         api_key,
        "question_type":   "",
        "analytics_output": {},
        "validator_output": {},
        "retry_count":     0,
        "retry_hint":      "",
        "web_results":     "",
        "web_answer":      "",
        "final_output":    {},
    }

    final_state = _graph.invoke(initial_state)
    return final_state["final_output"]
