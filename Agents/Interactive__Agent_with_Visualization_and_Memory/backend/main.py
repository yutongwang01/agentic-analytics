"""
FastAPI backend for the QA Agent.

Endpoints
---------
POST /chat          — SSE stream: pushes progress events then the final result
GET  /history       — return conversation turns for the current session
POST /reset         — clear session memory
GET  /health        — liveness check

SSE event types (field: event)
-------------------------------
progress            — status update string (e.g. "Generating code…")
result              — JSON payload with final_answer + viz_result
error               — error message string
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup: backend/ sits one level below project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from nodes import load_env_file, SessionMemory          # noqa: E402
from agent import run_workflow                          # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_env = load_env_file(str(ROOT / ".env"))
DATASETS_DIR = ROOT / "datasets"
AVAILABLE_DATASETS = {
    p.name: str(p)
    for p in DATASETS_DIR.glob("*.csv")
}
DEFAULT_DATASET = "housing.csv"

# One shared memory instance per server process.
# For multi-user deployments you would key this by session cookie/token.
_memory = SessionMemory(max_turns=5)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="CSV QA Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    dataset: str = DEFAULT_DATASET


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: str) -> str:
    """Format a single SSE frame."""
    # Escape newlines inside data so the frame stays valid
    safe_data = data.replace("\n", "\\n")
    return f"event: {event}\ndata: {safe_data}\n\n"


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking function in a thread pool so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

async def _chat_stream(question: str, dataset: str) -> AsyncGenerator[str, None]:
    csv_path = AVAILABLE_DATASETS.get(dataset, AVAILABLE_DATASETS[DEFAULT_DATASET])
    """
    Yields SSE frames:
      progress → progress → … → result  (happy path)
      progress → … → error              (failure path)
    """
    try:
        yield _sse("progress", "Received question — checking conversation context…")
        await asyncio.sleep(0)          # flush to client

        yield _sse("progress", "Generating analysis code…")
        await asyncio.sleep(0)

        # run_workflow is blocking (LLM calls + exec); run in thread pool
        def _blocking():
            return run_workflow(
                question=question,
                csv_path=csv_path,
                memory=_memory,
            )

        yield _sse("progress", "Executing code and evaluating result…")
        await asyncio.sleep(0)

        result = await _run_in_thread(_blocking)

        yield _sse("progress", "Formatting answer…")
        await asyncio.sleep(0)

        viz = result.get("viz_result")

        # Serialize viz_result for JSON transport
        viz_payload = None
        if viz and viz.should_visualize:
            viz_payload = {
                "should_visualize": viz.should_visualize,
                "chart_type": viz.chart_type,
                "figure_json": viz.figure_json,   # already a JSON string
                "error": viz.error,
            }
            yield _sse("progress", f"Visualization ready ({viz.chart_type} chart).")
        else:
            yield _sse("progress", "No visualization needed for this question.")
        await asyncio.sleep(0)

        # Build history snapshot for the frontend
        history_snapshot = [
            {
                "question": t["question"],
                "answer": t["answer"],
                "data_type": t["result"].data_type,
                "summary": t["result"].summary,
            }
            for t in _memory.turns
        ]

        payload = {
            "final_answer": result["final_answer"],
            "evaluation": result["evaluation"],
            "viz": viz_payload,
            "history": history_snapshot,
        }

        yield _sse("result", json.dumps(payload))

    except Exception as exc:  # noqa: BLE001
        yield _sse("error", str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
_current_dataset = DEFAULT_DATASET

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Stream the agent's response as Server-Sent Events.

    The client should open an EventSource (or fetch with ReadableStream)
    and listen for:
      - event: progress  → show as a loading status message
      - event: result    → parse JSON, render answer + chart
      - event: error     → show error message
    """
    global _memory, _current_dataset
    if not req.question.strip():
        return JSONResponse({"error": "Question must not be empty."}, status_code=400)

    # Reset memory if dataset changed
    if req.dataset != _current_dataset:
        _memory = SessionMemory(max_turns=5)
        _current_dataset = req.dataset
    return StreamingResponse(
        _chat_stream(req.question.strip(), req.dataset),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable Nginx buffering if behind proxy
        },
    )


@app.get("/history")
async def get_history():
    """Return the current session's conversation turns."""
    turns = [
        {
            "turn_id": i,
            "question": t["question"],
            "answer": t["answer"],
            "data_type": t["result"].data_type,
            "summary": t["result"].summary,
        }
        for i, t in enumerate(_memory.turns)
    ]
    return JSONResponse({"turns": turns, "count": len(turns)})


@app.post("/reset")
async def reset_session():
    """Clear session memory and start a fresh conversation."""
    global _memory
    _memory = SessionMemory(max_turns=5)
    return JSONResponse({"status": "ok", "message": "Session reset."})

@app.get("/datasets")
async def list_datasets():
    return JSONResponse({
        "datasets": list(AVAILABLE_DATASETS.keys()),
        "default": DEFAULT_DATASET,
    })

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "available_datasets": list(AVAILABLE_DATASETS.keys()),
        "current_dataset": _current_dataset,
        "model": _env.get("MODEL_NAME", ""),
        "memory_turns": len(_memory.turns),
    })
