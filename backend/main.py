"""
FastAPI backend for Orchestrated AI Analytics System.

Endpoints
---------
POST /chat      — SSE stream with progress events + final result
GET  /history   — conversation turns for current session
POST /reset     — clear session memory
GET  /datasets  — list available CSV datasets
GET  /health    — liveness check

SSE event types
---------------
progress    — status string shown in loading indicator
result      — JSON: {final_answer, viz, validator, question_type, retry_count}
error       — error message string
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup: backend/main.py → project root is one level up
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from nodes import SessionMemory, load_env_file   
from orchestrator import run_orchestrator       

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_env = load_env_file(str(ROOT / ".env"))

DATASETS_DIR = ROOT / "datasets"
AVAILABLE_DATASETS: dict[str, str] = {
    p.name: str(p)
    for p in sorted(DATASETS_DIR.glob("*.csv"))
}

# All CSV paths passed to the orchestrator for analytics questions
ALL_CSV_PATHS: list[str] = sorted(AVAILABLE_DATASETS.values())

MODEL_NAME = _env.get("MODEL_NAME", "gpt-4.1-mini")
API_KEY    = _env.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Orchestrated Analytics", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session state  (single-user; key by session token for multi-user)
# ---------------------------------------------------------------------------
_memory = SessionMemory(max_turns=10)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: str) -> str:
    """Format an SSE frame.

    SSE forbids raw newlines in a single `data:` line, but allows multiple
    `data:` lines per event — the browser rejoins them with '\n'. We split on
    newlines and emit one `data:` line per segment. This keeps JSON payloads
    intact (no manual escaping that would corrupt the JSON).
    """
    lines = data.split("\n")
    body = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{body}\n"


async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# ---------------------------------------------------------------------------
# SSE generator — core of the backend
# ---------------------------------------------------------------------------

async def _chat_stream(question: str) -> AsyncGenerator[str, None]:
    """
    Yields SSE frames while the orchestrator runs.

    Progress events give the user live feedback; the final result frame
    carries everything the frontend needs to render the answer + charts.
    """
    try:
        # ── 1. memory context ────────────────────────────────────────────
        yield _sse("progress", "Checking conversation context…")
        await asyncio.sleep(0)

        memory_context = _memory.get_context_for_prompt()

        # Enrich question with conversation context if it's a follow-up
        effective_question = question
        if memory_context and _memory.is_followup(question):
            effective_question = (
                f"{question}\n\n"
                f"[CONVERSATION CONTEXT]:\n{memory_context}"
            )

        # ── 2. classify ──────────────────────────────────────────────────
        yield _sse("progress", "Classifying question…")
        await asyncio.sleep(0)

        # ── 3. run orchestrator (blocking → thread pool) ─────────────────
        def _blocking():
            return run_orchestrator(
                question=effective_question,
                csv_paths=ALL_CSV_PATHS,
                model_name=MODEL_NAME,
                api_key=API_KEY,
            )

        q_type_hint = "web search" if any(
            kw in question.lower()
            for kw in ["what is", "what caused", "explain", "who is", "why did"]
        ) else "data analysis"
        yield _sse("progress", f"Running {q_type_hint}…")
        await asyncio.sleep(0)

        result = await _run_in_thread(_blocking)
        print("VALIDATOR FULL:", json.dumps(result.get("validator", {}), ensure_ascii=False), flush=True)

        # ── 4. post-process ──────────────────────────────────────────────
        question_type = result.get("question_type", "analytics")

        if question_type == "analytics":
            yield _sse("progress", "Validating analysis results…")
            await asyncio.sleep(0)

            verdict = result.get("validator", {}).get("verdict", "")
            retry_count = result.get("retry_count", 0)
            if retry_count > 0:
                yield _sse("progress",
                    f"Validator triggered {retry_count} retry(ies) — verdict: {verdict}")
                await asyncio.sleep(0)

        yield _sse("progress", "Formatting answer…")
        await asyncio.sleep(0)

        # ── 5. build plots payload ───────────────────────────────────────
        plots = result.get("plots", [])
        # first plot, plus the full list for multi-plot support.
        first_viz = None
        if plots:
            first_viz = {
                "should_visualize": True,
                "chart_type":       plots[0].get("title", "chart"),
                "figure_json":      plots[0].get("plotly_json"),
                "error":            None,
            }
            yield _sse("progress",
                f"{len(plots)} visualization(s) ready.")
            await asyncio.sleep(0)
        else:
            yield _sse("progress", "No visualization for this question.")
            await asyncio.sleep(0)

        # ── 6. update memory ─────────────────────────────────────────────
        from nodes import ExecutionResult as Result
        import pandas as pd

        fake_result = Result(
            data=result.get("final_answer", ""),
            data_type="scalar",
            summary=result.get("final_answer", "")[:200],
            schema=None,
            question=question,
        )
        _memory.add_turn(
            question=question,
            result=fake_result,
            answer=result.get("final_answer", ""),
        )

        # ── 7. emit result ───────────────────────────────────────────────
        payload = {
            "final_answer":  result.get("final_answer", ""),
            "question_type": question_type,
            "viz":           first_viz,
            "all_plots":     plots,
            "validator": result.get("validator") or {},
            "retry_count":   result.get("retry_count", 0),
            "total_iterations": result.get("total_iterations", 0),
            # history snapshot for frontend sidebar
            "history": [
                {"question": t["question"], "answer": t["answer"]}
                for t in _memory.turns
            ],
        }
        yield _sse("result", json.dumps(payload))

    except Exception as exc:  # noqa: BLE001
        import traceback
        yield _sse("error", f"{exc}\n{traceback.format_exc(limit=3)}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        return JSONResponse({"error": "Question must not be empty."}, status_code=400)
    return StreamingResponse(
        _chat_stream(req.question.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history")
async def get_history():
    turns = [
        {"turn_id": i, "question": t["question"], "answer": t["answer"]}
        for i, t in enumerate(_memory.turns)
    ]
    return JSONResponse({"turns": turns, "count": len(turns)})


@app.post("/reset")
async def reset_session():
    global _memory
    _memory = SessionMemory(max_turns=10)
    return JSONResponse({"status": "ok", "message": "Session reset."})


@app.get("/datasets")
async def list_datasets():
    return JSONResponse({
        "datasets": list(AVAILABLE_DATASETS.keys()),
        "default":  next(iter(AVAILABLE_DATASETS), ""),
        "all_paths": ALL_CSV_PATHS,
    })


@app.get("/health")
async def health():
    return JSONResponse({
        "status":      "ok",
        "model":       MODEL_NAME,
        "datasets":    list(AVAILABLE_DATASETS.keys()),
        "memory_turns": len(_memory.turns),
        "csv_paths":   ALL_CSV_PATHS,
    })
