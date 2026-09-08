"""LangChain agent: turns a natural-language question + dataframe schema into a
safe pandas expression, executes it in a sandboxed namespace, and streams a
plain-English insight about the result.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import re
from typing import Any, Iterator

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
MAX_QUERY_CHARS = 500
SAMPLE_ROWS = 5
MAX_SUBQUESTIONS = 6


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.0, streaming: bool = False) -> ChatGroq:
    # Read env vars at call time, not import time — a module-level constant
    # would freeze in whatever GROQ_MODEL was set (or unset) before this
    # module was first imported, ignoring any later load_dotenv() call.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(model=model, api_key=api_key, temperature=temperature, streaming=streaming)


CODEGEN_SYSTEM_PROMPT = """You are a data analyst that translates a natural-language question into a \
single pandas expression that operates on a dataframe named `df`.

Rules (follow all of them exactly):
- Respond with ONE pandas expression only. No import statements, no assignments, no comments, \
no explanations, no markdown fences, no multiple statements separated by `;` or newlines.
- The expression must start with `df`.
- Only use `df`, `pd`, and `np` — nothing else is available.
- Never call: eval, exec, query, apply with file/network side effects, to_csv, to_excel, to_pickle, \
to_sql, read_csv, or anything touching the filesystem, network, or the `os`/`sys` modules.
- NEVER call `.plot`, `.plot.hist()`, `.plot.bar()`, `.hist()`, `.boxplot()`, or any other \
matplotlib-backed plotting method — even if the question asks for a "histogram", "box plot", \
"pie chart", "scatter plot", etc. A separate system renders the chart from whatever data you \
return, so just return the underlying DataFrame/Series.
- A "box plot" / "distribution" / "spread" / "scatter" / "relationship between" question wants \
RAW, UN-GROUPED rows — do NOT `groupby` or aggregate. Just select the relevant columns, e.g. \
`df[['department', 'salary']]` for "salary spread by department", or `df[['age', 'income']]` \
for "age vs income". Grouping first collapses each category to one row, which destroys the \
spread/relationship the question is asking about.
- A "by <category>" total/average/count question (with no "spread"/"distribution"/"relationship" \
wording) DOES want it aggregated — prefer groupby + agg, sort_values, value_counts, pivot_table, \
resetting the index afterward so category labels become a column.
- Reset the index with `.reset_index()` after a groupby/agg so category labels become a column.
- Only convert a column with `pd.to_numeric(...)` when the column IS meant to be a numeric \
quantity (a price, a percentage, a count) but got stored messily as text — e.g. "$1,234", "45%". \
NEVER apply `pd.to_numeric` to a genuinely categorical/label column (a grade like "A"-"G", a type \
like "RENT"/"OWN", an intent/category name) just because its dtype is text — stripping digits out \
of "MORTGAGE" or "PERSONAL" turns every row into NaN and destroys the column. If a column's dtype \
is already numeric (int64/float64) in the schema below, it does NOT need `pd.to_numeric` at all — \
only reach for it on text/object columns whose sample values look like numbers with junk attached. \
When you do need it, since the expression must start with `df`, wrap it in `.assign(...)`, e.g. \
`df.assign(price=pd.to_numeric(df['price'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce'))['price']`.
- For "correlation matrix" / "correlation heatmap" / "correlation between all numeric columns" \
questions, NEVER call bare `df.corr()` — if the dataframe has any non-numeric column, pandas raises \
instead of skipping it. Always restrict to numeric columns first: \
`df.select_dtypes('number').corr()`, or pass `numeric_only=True`: `df.corr(numeric_only=True)`.
- If the question cannot be answered from the given columns, return exactly: df.head(5)
"""


def _schema_context(df: pd.DataFrame) -> str:
    dtypes = "\n".join(f"- {col}: {dtype}" for col, dtype in df.dtypes.astype(str).items())
    sample = df.head(SAMPLE_ROWS).to_csv(index=False)
    return (
        f"Dataframe `df` has {len(df)} rows and {len(df.columns)} columns.\n\n"
        f"Columns and dtypes:\n{dtypes}\n\n"
        f"Sample rows (CSV):\n{sample}"
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _strip_code_fences(text: str) -> str:
    text = _strip_fences(text)
    # The model is instructed to return one line, but take the first
    # non-empty line anyway in case it adds stray commentary.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def generate_pandas_code(df: pd.DataFrame, question: str) -> str:
    llm = _get_llm()
    messages = [
        SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
        HumanMessage(content=f"{_schema_context(df)}\n\nQuestion: {question}\n\nExpression:"),
    ]
    response = llm.invoke(messages)
    return _strip_code_fences(response.content)


# ---------------------------------------------------------------------------
# Question splitting — a question that asks for several distinct analyses
# ("attrition by department, salary distribution, and a correlation heatmap")
# becomes several standalone sub-questions, each answered with its own
# pandas expression and chart. A single-focus question splits into a list
# of exactly one (itself, unchanged).
# ---------------------------------------------------------------------------

SPLIT_SYSTEM_PROMPT = f"""You split a user's data question into the distinct analyses it's asking for.

Rules:
- If the question asks for ONE thing, respond with a JSON array containing exactly that one \
question, unchanged: ["<the original question>"]
- If the question asks for SEVERAL distinct things (e.g. joined by commas, "and", or phrased as \
a list — "attrition by department, salary distribution, and a correlation heatmap"), respond with \
a JSON array of short standalone questions, one per distinct thing, each answerable with a single \
chart. Rephrase each into a complete, self-contained question — don't just copy sentence fragments.
- One metric broken out across every value of a SINGLE dimension is still ONE thing, not one part \
per value — "average X by department", "X across different age groups", "X for each category" all \
stay as ONE question (the resulting chart has one bar/point per group). NEVER split it into a \
separate question per department/age-bracket/category value.
- "A correlation matrix" / "a correlation heatmap of all numeric columns" is ONE thing (one \
heatmap chart) — NEVER split it into a separate question per pair of columns.
- Preserve the original wording/intent for each part; don't invent new analyses that weren't asked for.
- At most {MAX_SUBQUESTIONS} items. If more were asked for, keep the first {MAX_SUBQUESTIONS} and drop the rest.
- Respond with ONLY the JSON array. No markdown fences, no explanation, no other text.

Example:
Question: "Give me a complete workforce analysis — attrition by department, salary distribution, \
and a correlation heatmap of all key metrics"
["Attrition by department", "Salary distribution", "Correlation heatmap of all key metrics"]

Example (single dimension broken out by group — stays ONE question, do not split per group):
Question: "Plot average loan amount across different age groups"
["Plot average loan amount across different age groups"]

Example (correlation matrix — stays ONE question, do not split per column pair):
Question: "Generate a correlation matrix for all numeric columns"
["Generate a correlation matrix for all numeric columns"]

Example:
Question: "average salary by department"
["average salary by department"]
"""


def _parse_json_string_list(text: str) -> list[str] | None:
    text = _strip_fences(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    items = [str(item).strip() for item in parsed if str(item).strip()]
    return items or None


def split_into_subquestions(df: pd.DataFrame, question: str) -> list[str]:
    """Break a compound question into standalone sub-questions, one per
    distinct analysis. Always returns at least [question] — any failure to
    reach the LLM or parse its response degrades to the single-question
    behavior rather than blocking the query."""
    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=SPLIT_SYSTEM_PROMPT),
            HumanMessage(content=f"{_schema_context(df)}\n\nQuestion: {question}"),
        ]
        response = llm.invoke(messages)
        items = _parse_json_string_list(response.content)
    except Exception:
        items = None

    if not items:
        return [question]
    return items[:MAX_SUBQUESTIONS]


# ---------------------------------------------------------------------------
# Safety: AST allow-listing + sandboxed execution
#
# The LLM output is untrusted input. Two independent layers protect execution:
#   1. `_validate_ast` parses the expression and rejects anything outside a
#      small allow-list of node types, identifiers, and attribute names —
#      this blocks dunder access, imports, and known-dangerous methods
#      (eval/exec/to_csv/read_csv/os/...) before any code runs.
#   2. `safe_execute` runs the validated expression with `__builtins__`
#      replaced by a tiny whitelist and a namespace exposing only
#      `df`, `pd`, `np` — so even a node type we failed to anticipate has
#      nothing dangerous reachable by name.
# ---------------------------------------------------------------------------

_ALLOWED_NAMES = {"df", "pd", "np"}
_ALLOWED_BUILTIN_CALLS = {
    "len", "sum", "min", "max", "sorted", "abs", "round", "list", "dict",
    "set", "tuple", "range", "str", "int", "float", "bool",
}
_BLOCKED_ATTRS = {
    "eval", "query", "exec", "compile", "to_csv", "to_excel", "to_pickle",
    "to_sql", "to_parquet", "to_hdf", "to_feather", "to_json", "to_html",
    "to_clipboard", "to_markdown", "read_csv", "read_excel", "read_sql",
    "read_pickle", "read_json", "read_parquet", "read_html", "open",
    "pipe", "system", "popen", "__import__",
    # matplotlib-backed plotting — not installed, and charting is handled
    # separately by charter.py from whatever data the expression returns.
    "plot", "hist", "boxplot",
}
_ALLOWED_AST_NODES = (
    ast.Expression, ast.Call, ast.Attribute, ast.Subscript, ast.Name,
    ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.List, ast.Tuple, ast.Dict, ast.Slice, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Lambda, ast.arguments, ast.arg, ast.Starred,
)

_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_CALLS}


class UnsafeQueryError(ValueError):
    pass


def _lambda_param_names(tree: ast.AST) -> set[str]:
    """Names bound by a `lambda ...: ...` parameter list — these are locally
    scoped, not global references, so they don't belong in `_ALLOWED_NAMES`
    but are still safe to allow (a lambda param can never grant access to
    anything beyond what its own expression body already reaches)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            a = node.args
            names.update(p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs))
            if a.vararg:
                names.add(a.vararg.arg)
            if a.kwarg:
                names.add(a.kwarg.arg)
    return names


def _validate_ast(code: str) -> ast.Expression:
    if not code or len(code) > MAX_QUERY_CHARS:
        raise UnsafeQueryError("Generated query is empty or too long")
    if not code.startswith("df"):
        raise UnsafeQueryError("Generated query must start with `df`")

    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as exc:
        raise UnsafeQueryError(f"Generated query is not valid Python: {exc}") from exc

    allowed_names = _ALLOWED_NAMES | _lambda_param_names(tree)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise UnsafeQueryError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names and node.id not in _ALLOWED_BUILTIN_CALLS:
            raise UnsafeQueryError(f"Disallowed identifier: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _BLOCKED_ATTRS:
                raise UnsafeQueryError(f"Disallowed attribute access: {node.attr}")

    return tree


def safe_execute(code: str, df: pd.DataFrame) -> Any:
    """Validate `code` against an AST allow-list, then evaluate it against `df`
    with no builtins beyond a small whitelist and no access to the
    filesystem, network, `os`, or `sys`."""
    tree = _validate_ast(code)
    compiled = compile(tree, "<pulse-query>", "eval")
    namespace = {"df": df, "pd": pd, "np": np}
    return eval(compiled, {"__builtins__": _SAFE_BUILTINS}, namespace)  # noqa: S307


# ---------------------------------------------------------------------------
# Insight generation (streamed)
# ---------------------------------------------------------------------------

INSIGHT_SYSTEM_PROMPT = """You are a sharp business analyst. You'll be given a question and one or \
more labeled results. In plain English, summarize what the result(s) show and call out the most \
notable takeaway from each (e.g. the top value, a trend, an outlier). One short sentence per result \
— if there's only one result, that's 1-2 sentences total. No preamble, no markdown, no repeating \
the question, no result labels/headers in your answer."""


def _result_summary(result: Any) -> str:
    if isinstance(result, pd.DataFrame):
        return result.head(10).to_csv(index=False)
    if isinstance(result, pd.Series):
        return result.head(10).to_csv()
    return str(result)


def stream_insight(question: str, labeled_results: list[tuple[str, Any]]) -> Iterator[str]:
    """`labeled_results` is a list of (label, result) pairs — one per chart.
    A single-chart question passes a list of length 1."""
    blocks = "\n\n".join(f"### {label}\n{_result_summary(result)}" for label, result in labeled_results)
    llm = _get_llm(temperature=0.3, streaming=True)
    messages = [
        SystemMessage(content=INSIGHT_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nResult(s):\n{blocks}\n\nInsight:"),
    ]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
