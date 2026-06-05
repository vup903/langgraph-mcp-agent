"""MCP server: exposes the agent's tools over the Model Context Protocol.

Run as a standalone stdio server:
    python -m langgraph_mcp_agent.mcp_server

The LangGraph agent (see `mcp_client.py` + `graph.py`) connects to this server
and consumes these tools — i.e. the agent is decoupled from tool implementations
via the MCP boundary, exactly as a production agent would integrate external tools.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from langgraph_mcp_agent import tools

mcp = FastMCP("langgraph-mcp-agent-tools")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return tools.add(a, b)


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return tools.multiply(a, b)


@mcp.tool()
def search_notes(query: str) -> str:
    """Search the built-in knowledge base and return the most relevant note."""
    return tools.search_notes(query)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
