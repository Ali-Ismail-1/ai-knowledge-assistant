# app/services/tools.py
from langchain.agents import Tool
from typing import Callable

# Web Search
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    _ddg = DuckDuckGoSearchRun()
    def _web_search(query: str) -> str:
        return _ddg.run(query)
except Exception:
    # Falllback if package is missing. Robust agent in local dev
    def _web_search(query: str) -> str:
        return "(Web search not available - please install langchain-community)"

# Calculator
from math import sin, cos, tan, sqrt, log, pi, e, pow, floor, ceil, abs, max, min, round, exp
def _calc(expression: str) -> str:
    try:
        allowed = {k: v for k, v in globals().items() if k in {
        'sin','cos','tan','sqrt','log','pi','e','pow','floor','ceil','abs','max','min','round','exp'
        }}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"calc error: {e}"

# File read tool (read-only)
import pathlib

def _read_file(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists() or not p.is_file():
        return f"File not found: {path}"
    if p.suffix not in {'.txt', '.md', '.pdf'}:
        return f"Unsupported file type: {p.suffix}"
    return p.read_text(encoding="utf-8")[:8000]

def make_tools(retrieval_fn: Callable[[str], str]) -> list[Tool]:
    return [
        Tool(
            name="retrieval",
            func=retrieval_fn,
            description=(
                "Use to answer questions from the user's document corpus."
                "Input: a natural-language question. Output: an answer string."
            ),
        ),
        Tool(
            name="web_search",
            func=_web_search,
            description=(
                "Use to search the web when the docs don't have the answer or to update facts."
            ),
        ),
        Tool(name="calculator", func=_calc, description="Evaluate a math expression."),
        Tool(name="read_file", func=_read_file, description="Read a local file for reference."),
    ]
