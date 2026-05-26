"""CSV QA agent built with LangGraph."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from nodes import (
    load_env_file,
    generate_code,
    execute_code,
    evaluate_result,
    format_final_answer,
    SessionMemory,
    run_visualization_agent,
    VisualizationResult,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # inputs
    question: str
    csv_path: str
    model_name: str
    api_key: str
    # intermediate
    generated_code: str
    execution_result: Any
    execution_error: str | None
    evaluation: str
    retry_count: int
    final_answer: str
    memory: Any          # SessionMemory
    is_followup: bool
    viz_result: Any      # VisualizationResult | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_generate(state: AgentState) -> AgentState:
    memory: SessionMemory = state.get("memory")
    memory_context = memory.get_context_for_prompt() if memory else ""
    is_followup = state.get("is_followup", False)

    code = generate_code(
        question=state["question"],
        csv_path=state["csv_path"],
        model_name=state["model_name"],
        api_key=state["api_key"],
        memory_context=memory_context,
        is_followup=is_followup,
    )
    return {"generated_code": code}


def node_execute(state: AgentState) -> AgentState:
    memory: SessionMemory = state.get("memory")
    prior = memory.last_result if (memory and state.get("is_followup")) else None

    result, error = execute_code(
        code=state["generated_code"],
        csv_path=state["csv_path"],
        prior_result=prior,
        question=state["question"],
        turn_id=len(memory.turns) if memory else 0,
    )
    return {"execution_result": result, "execution_error": error}


def node_evaluate(state: AgentState) -> AgentState:
    ev = evaluate_result(
        execution_error=state["execution_error"],
        result=state["execution_result"],
        question=state["question"],
        model_name=state["model_name"],
        api_key=state["api_key"],
    )
    return {"evaluation": ev}


def node_answer(state: AgentState) -> AgentState:
    answer = format_final_answer(
        result=state["execution_result"],
        evaluation=state["evaluation"],
        question=state["question"],
        model_name=state["model_name"],
        api_key=state["api_key"],
        error_message=state["execution_error"],
    )
    memory: SessionMemory = state.get("memory")
    if memory:
        memory.add_turn(
            question=state["question"],
            result=state["execution_result"],
            answer=answer,
        )
    return {"final_answer": answer}


def node_visualize(state: AgentState) -> AgentState:
    """Run the visualization agent on the current execution result."""
    # Skip visualization if evaluation failed
    if state.get("evaluation") != "PASS":
        return {"viz_result": VisualizationResult(
            should_visualize=False,
            chart_type=None,
            plotly_code=None,
            figure_json=None,
            error="Skipped: evaluation did not pass.",
        )}

    viz = run_visualization_agent(
        question=state["question"],
        result=state["execution_result"],
        model_name=state["model_name"],
        api_key=state["api_key"],
    )
    return {"viz_result": viz}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_evaluate(state: AgentState) -> str:
    """Retry once on FAIL, then proceed to answer."""
    if state["evaluation"] == "FAIL" and state.get("retry_count", 0) < 1:
        return "retry"
    return "answer"


def node_retry(state: AgentState) -> AgentState:
    """Increment retry counter; the graph loops back to generate."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("generate", node_generate)
    builder.add_node("execute", node_execute)
    builder.add_node("evaluate", node_evaluate)
    builder.add_node("retry", node_retry)
    builder.add_node("answer", node_answer)
    builder.add_node("visualize", node_visualize)

    builder.set_entry_point("generate")
    builder.add_edge("generate", "execute")
    builder.add_edge("execute", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"retry": "retry", "answer": "answer"},
    )
    builder.add_edge("retry", "generate")
    builder.add_edge("answer", "visualize")   # answer → visualize → END
    builder.add_edge("visualize", END)

    return builder.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_workflow(
    question: str,
    csv_path: str,
    memory: SessionMemory | None = None,
) -> dict:
    """Run the LangGraph CSV-QA workflow and return the result dictionary."""
    config = load_env_file()
    api_key = config.get("OPENAI_API_KEY", "")
    model_name = config.get("MODEL_NAME", "")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")
    if not model_name:
        raise ValueError("MODEL_NAME is missing from .env")

    is_followup = memory.is_followup(question) if memory else False

    initial_state: AgentState = {
        "question": question,
        "csv_path": csv_path,
        "model_name": model_name,
        "api_key": api_key,
        "generated_code": "",
        "execution_result": None,
        "execution_error": None,
        "evaluation": "",
        "retry_count": 0,
        "final_answer": "",
        "memory": memory,
        "is_followup": is_followup,
        "viz_result": None,
    }

    final_state = _graph.invoke(initial_state)

    return {
        "generated_code": final_state["generated_code"],
        "execution_result": final_state["execution_result"],
        "evaluation": final_state["evaluation"],
        "final_answer": final_state["final_answer"],
        "viz_result": final_state.get("viz_result"),
    }
