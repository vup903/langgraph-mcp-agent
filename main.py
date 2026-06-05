"""Runnable demo: a LangGraph ReAct agent using tools served over MCP.

Requires an LLM API key (the unit tests do NOT). Configure via .env or env vars:

    MODEL=google_genai:gemini-2.0-flash   GOOGLE_API_KEY=...
    MODEL=openai:gpt-4o-mini              OPENAI_API_KEY=...
    MODEL=anthropic:claude-3-5-haiku-latest  ANTHROPIC_API_KEY=...

Run:  python main.py
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from langgraph_mcp_agent.graph import build_agent
from langgraph_mcp_agent.mcp_client import load_mcp_tools


async def main() -> None:
    load_dotenv()
    model = init_chat_model(os.environ.get("MODEL", "google_genai:gemini-2.0-flash"))

    tools = await load_mcp_tools()  # tools come from the MCP server, not direct imports
    agent = build_agent(model, tools=tools)

    question = "Search your notes for what MCP is, then compute 21 * 2."
    result = await agent.ainvoke({"messages": [HumanMessage(question)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
