"""FastAPI app: CSV upload, in-memory session storage, and the streaming
natural-language query endpoint that ties agent.py + charter.py together."""
from __future__ import annotations

import io
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # must run before importing agent — it reads GROQ_MODEL from the environment

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import agent
import charter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulse")

app = FastAPI(title="PULSE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_COLUMNS = 200
SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours


@dataclass
class Session:
    df: pd.DataFrame
    filename: str
    created_at: float = field(default_factory=time.time)


SESSIONS: dict[str, Session] = {}


def _evict_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s.created_at > SESSION_TTL_SECONDS]
    for sid in expired:
        del SESSIONS[sid]


def _get_session(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please re-upload your CSV.")
    return session


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    seen: dict[str, int] = {}
    deduped = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            deduped.append(c)
    df.columns = deduped
    return df


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _parse_upload(contents: bytes, filename: str) -> dict:
    """CPU-bound CSV parsing + dtype inference. Runs in a worker thread (see
    `run_in_threadpool` below) — for a large file this can take a while, and
    running it directly in the async handler would block the event loop for
    every other request (including unrelated users' queries) until it finishes."""
    df = None
    parse_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            parse_error = exc
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df is None:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {parse_error}")

    if df.empty or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows or columns")
    if df.shape[1] > MAX_COLUMNS:
        raise HTTPException(status_code=400, detail=f"CSV has too many columns (max {MAX_COLUMNS})")

    df = _clean_columns(df)
    df = df.convert_dtypes()
    for col in df.columns:
        if df[col].dtype == "string":
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9:
                df[col] = parsed

    _evict_expired_sessions()
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = Session(df=df, filename=filename)

    return {
        "session_id": session_id,
        "filename": filename,
        "rows": len(df),
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "sample_rows": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty")

    return await run_in_threadpool(_parse_upload, contents, file.filename)


class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=500)


def _sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _query_stream(df: pd.DataFrame, question: str):
    try:
        subquestions = agent.split_into_subquestions(df, question)
    except Exception:  # LLM/network failure
        logger.exception("Question splitting failed")
        yield _sse("error", {"message": "Couldn't reach the analysis engine. Please try again."})
        return

    # A single-item split may have been lightly reworded by the LLM — use the
    # user's original wording verbatim in that (by far the most common) case.
    if len(subquestions) <= 1:
        subquestions = [question]

    labeled_results: list[tuple[str, Any]] = []
    for index, subq in enumerate(subquestions):
        try:
            code = agent.generate_pandas_code(df, subq)
            result = agent.safe_execute(code, df)
            # No title here — in multi-chart mode the frontend already renders
            # `subq` as a label above each chart, so an in-figure title would repeat it.
            figure = charter.generate_chart(result, question=subq)
        except agent.UnsafeQueryError as exc:
            logger.warning("Rejected unsafe query %r: %s", subq, exc)
            yield _sse("chart_error", {"index": index, "label": subq, "message": "That part couldn't be answered safely."})
            continue
        except Exception:
            logger.exception("Sub-question failed: %r", subq)
            yield _sse("chart_error", {"index": index, "label": subq, "message": "That part didn't compute."})
            continue

        yield _sse("chart", {"index": index, "label": subq, "figure": figure})
        labeled_results.append((subq, result))

    if not labeled_results:
        yield _sse("error", {"message": "That didn't compute — try rephrasing your question."})
        yield _sse("done", {})
        return

    try:
        for token in agent.stream_insight(question, labeled_results):
            yield _sse("insight", {"token": token})
    except Exception:
        logger.exception("Insight streaming failed")

    yield _sse("done", {})


@app.post("/api/query")
async def query(request: QueryRequest) -> StreamingResponse:
    session = _get_session(request.session_id)
    question = request.question.strip()
    return StreamingResponse(
        _query_stream(session.df, question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
