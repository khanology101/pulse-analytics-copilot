# PULSE

AI-powered business analytics copilot. Upload any CSV, ask a question in plain English, and get back a Pandas query, one or more interactive Plotly charts, and a plain-English insight — auto-generated and executed against your data.

## Stack

- **Backend**: FastAPI + LangChain (`langchain-groq`) + Pandas + Plotly
- **LLM**: Groq, model set via `GROQ_MODEL` in `backend/.env` (default: `openai/gpt-oss-120b`)
- **Frontend**: React + Vite + Plotly.js
- **Storage**: uploaded CSVs are parsed into an in-memory session (no disk, no database)

> Groq periodically retires models. If questions start failing immediately with a "model not found" style error, the configured `GROQ_MODEL` has likely been deprecated — list what's currently available with the Groq client (`client.models.list()`) or check [console.groq.com](https://console.groq.com), then update `backend/.env`.

## Features

- **Eight chart types, auto-selected from the shape of the query result**: bar, grouped bar, line, donut, histogram, scatter, box plot, and correlation heatmap. A chart type named explicitly in the question (e.g. "...as a pie chart") overrides the auto-pick when the data supports it.
- **Multi-chart dashboards**: a compound question ("give me a complete workforce analysis — attrition by department, salary distribution, and a correlation heatmap") is split into standalone sub-questions, each executed and charted independently and laid out in a grid, with one synthesized insight covering all of them. If one part fails, the rest still render.
- **Large file support**: uploads up to 500MB (`MAX_UPLOAD_BYTES` in `backend/main.py`). Parsing runs in a worker thread so the server stays responsive to other requests while a large file is processed, and the upload UI shows real byte-level progress followed by a "processing" state.
- **Sandboxed query execution**: see [Safety](#safety) below.

## Project structure

```
pulse/
├── backend/
│   ├── main.py          # FastAPI app: upload, session storage, SSE query endpoint
│   ├── agent.py          # NL -> pandas codegen, question splitting, AST-sandboxed execution, insight streaming
│   ├── charter.py        # Result -> themed Plotly figure, chart-type auto-selection
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx        # Layout + session state
│       ├── Upload.jsx      # CSV drag-and-drop upload with progress
│       ├── Chat.jsx        # Query input + streamed response (single- and multi-chart)
│       ├── Chart.jsx       # Plotly renderer
│       └── api.js          # Upload + SSE query helpers
└── README.md
```

## Setup

### Backend

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Create `backend\.env` with:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Get an API key at [console.groq.com](https://console.groq.com). Adjust `GROQ_MODEL` if the default has been deprecated (see the note under [Stack](#stack)).

### Frontend

```powershell
cd frontend
npm install
```

The frontend talks to the backend at whatever `VITE_API_BASE` is set to in `frontend/.env` (defaults to `http://127.0.0.1:8000`) — edit that file to point it elsewhere.

## Running

**Backend** (from `backend/`):

```powershell
.\.venv\Scripts\python -m uvicorn main:app --port 8000
```

**Frontend** (from `frontend/`, separate terminal):

```powershell
npm run dev
```

Open `http://localhost:5173`, upload a CSV, and start asking questions.

## Safety

User questions are never executed as arbitrary code. The LLM is prompted to return a single pandas expression, which is then:

1. Parsed with `ast.parse` and walked against an allow-list of node types, identifiers (`df`, `pd`, `np`, lambda parameters, and a small builtin whitelist), and attribute names — dunder access, imports, and known-dangerous methods (`eval`, `exec`, `to_csv`, `read_csv`, `os`/`sys` access, matplotlib plotting calls, etc.) are rejected before anything runs.
2. Evaluated with `__builtins__` replaced by that same small whitelist, in a namespace exposing only `df`, `pd`, and `np` — nothing else is reachable by name, even if a node type slips past the first check.

See `backend/agent.py` for the implementation.
