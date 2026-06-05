"""Load the MCP server's tools as LangChain tools, via langchain-mcp-adapters.

This is what makes the agent consume tools *through MCP* rather than importing
them directly: it spawns `mcp_server.py` over stdio and adapts its tools into
LangChain tools that the LangGraph agent can call.
"""
from __future__ import annotations

import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


def make_client(python_executable: str | None = None) -> MultiServerMCPClient:
    """Build a client configured to launch our stdio MCP server."""
    py = python_executable or sys.executable
    return MultiServerMCPClient(
        {
            "agent-tools": {
                "command": py,
                "args": ["-m", "langgraph_mcp_agent.mcp_server"],
                "transport": "stdio",
            }
        }
    )


async def load_mcp_tools(python_executable: str | None = None):
    """Return the MCP server's tools as a list of LangChain tools."""
    client = make_client(python_executable)
    return await client.get_tools()
