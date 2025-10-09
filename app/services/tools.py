# app/services/tools.py
from langchain.agents import Tool
from typing import Callable

# Lightweight web search using DuckDuckGo (no API key required)


try:
    from langchain_community.tools import DuckDuckGoSearchRun
    _ddg = DuckDuckGoSearchRun()
    def _web_search(query: str) -> str:
        return _ddg.run(query)
except Exception:
    # Falllback if package is missing. Robust agent in local dev
    def _web_search(query: str) -> str:
        return "(Web search not available - please install langchain-community)"

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
    ]
