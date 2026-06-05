"""Pure tool implementations.

Kept free of any framework imports so the SAME logic backs both the MCP server
(`mcp_server.py`) and the LangChain tool wrappers (`graph.py`), and so the unit
tests can exercise them with no LLM and no network.
"""
from __future__ import annotations

# A tiny built-in knowledge base — stands in for a retrieval/RAG source.
NOTES: dict[str, str] = {
    "mcp": (
        "The Model Context Protocol (MCP) is an open standard that lets AI apps connect "
        "to tools and data sources through a uniform client/server interface."
    ),
    "langgraph": (
        "LangGraph models an agent as a graph of nodes and edges, giving explicit, "
        "inspectable control over multi-step tool-calling loops."
    ),
    "rag": (
        "Retrieval-Augmented Generation grounds an LLM's answer in retrieved documents "
        "to reduce hallucination and support citations."
    ),
}


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def search_notes(query: str) -> str:
    """Keyword-search the built-in knowledge base; return the best matching note."""
    q = query.lower().strip()
    if not q:
        return "No matching note found."
    # Prefer a direct key hit, then fall back to token overlap with note text.
    for key, text in NOTES.items():
        if key in q:
            return text
    tokens = [t for t in q.split() if len(t) > 2]
    for text in NOTES.values():
        lowered = text.lower()
        if any(tok in lowered for tok in tokens):
            return text
    return "No matching note found."
