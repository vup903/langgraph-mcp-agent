"""Verify the MCP server registers and executes its tools (no LLM, no network)."""
import asyncio

from langgraph_mcp_agent.mcp_server import mcp


def _result_text(result) -> str:
    """Tolerantly extract text from FastMCP.call_tool's return value.

    Across mcp SDK versions call_tool returns either a list of content objects
    or a (content_list, structured) tuple; handle both.
    """
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        return " ".join(getattr(item, "text", str(item)) for item in result)
    return str(result)


def test_server_lists_all_tools():
    listed = asyncio.run(mcp.list_tools())
    names = {t.name for t in listed}
    assert {"add", "multiply", "search_notes", "list_topics"} <= names
    # every tool advertises a description (good MCP hygiene)
    assert all(t.description for t in listed)


def test_call_add_over_mcp():
    text = _result_text(asyncio.run(mcp.call_tool("add", {"a": 2, "b": 3})))
    assert "5" in text


def test_call_search_notes_over_mcp():
    text = _result_text(asyncio.run(mcp.call_tool("search_notes", {"query": "mcp"})))
    assert "Model Context Protocol" in text


def test_call_list_topics_over_mcp():
    text = _result_text(asyncio.run(mcp.call_tool("list_topics", {})))
    assert "mcp" in text and "rag" in text
