# langgraph-mcp-agent

A small but complete **LangGraph** ReAct agent whose tools are served over the
**Model Context Protocol (MCP)**. The agent is decoupled from its tools by the
MCP boundary — the same pattern production agents use to integrate external
tools and data sources.

```
                         ┌─────────────────────────┐
  HumanMessage ─▶ model ─┤ tools_condition          │
        ▲                │  • tool calls? ─▶ tools ──┼─▶ (executes) ─┐
        │                │  • else ─▶ END            │               │
        └────────────────┴──────────────────────────┘ ◀─────────────┘
                                  ▲
                     tools loaded over MCP (stdio)
                                  │
                       ┌──────────┴───────────┐
                       │  mcp_server.py        │  add · multiply · search_notes
                       │  (FastMCP, stdio)     │
                       └───────────────────────┘
```

## What it demonstrates
- **LangGraph**: an explicit `StateGraph` ReAct loop (`model → tools_condition → tools → model`) instead of a black-box agent — every step is inspectable. See [`graph.py`](src/langgraph_mcp_agent/graph.py).
- **MCP**: a `FastMCP` server exposing tools ([`mcp_server.py`](src/langgraph_mcp_agent/mcp_server.py)) and a client that adapts them into LangChain tools via `langchain-mcp-adapters` ([`mcp_client.py`](src/langgraph_mcp_agent/mcp_client.py)).
- **Testing/CI**: hermetic `pytest` suite that runs the full tool-calling loop with a scripted fake model — **no API key, no network** — wired to GitHub Actions ([ci.yml](.github/workflows/ci.yml)).
- **Containerized**: a [`Dockerfile`](Dockerfile) for reproducible runs.

## Run the tests (no API key)
```bash
pip install -e ".[dev]"
pytest -q
```

## Run the agent (needs an LLM key)
```bash
cp .env.example .env        # set MODEL + provider key
pip install -e ".[google]"  # or .[openai]
python main.py
```
`main.py` launches the MCP server, loads its tools over MCP, and runs the
LangGraph agent on: *"Search your notes for what MCP is, then compute 21 * 2."*

## Layout
```
src/langgraph_mcp_agent/
  tools.py        # pure tool logic (shared by server + agent + tests)
  mcp_server.py   # FastMCP server (stdio) exposing the tools over MCP
  mcp_client.py   # load MCP tools as LangChain tools (langchain-mcp-adapters)
  graph.py        # LangGraph StateGraph ReAct agent
main.py           # runnable demo (agent consumes tools via MCP)
tests/            # hermetic unit tests (no key)
```
