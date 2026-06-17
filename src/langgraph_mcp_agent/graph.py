"""The LangGraph agent: a ReAct loop built as an explicit StateGraph.

Topology:

    START -> model -> (tools_condition) -> tools -> model -> ... -> END

`model` calls the LLM (bound to the tools); `tools_condition` routes to the
`tools` node whenever the model emits tool calls, otherwise to END. The `tools`
node executes the calls and feeds results back to `model`. This is the canonical
tool-calling agent, written out as a graph so every step is inspectable.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool as as_tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_mcp_agent import tools as impl


# LangChain tool wrappers around the shared pure functions. These are the
# fallback tools used when the agent is NOT loading tools from the MCP server
# (e.g. in the hermetic unit tests).
@as_tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return impl.add(a, b)


@as_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return impl.multiply(a, b)


@as_tool
def search_notes(query: str) -> str:
    """Search the built-in knowledge base and return the most relevant note."""
    return impl.search_notes(query)


@as_tool
def list_topics() -> str:
    """List the topics available in the built-in knowledge base."""
    return impl.list_topics()


LOCAL_TOOLS = [add, multiply, search_notes, list_topics]


def _on_tool_error(exc: Exception) -> str:
    """Turn a tool exception into a message the model can recover from."""
    return (
        f"Tool error: {exc}. Do not retry blindly — continue without this tool's "
        "result and tell the user the tool failed."
    )


def build_agent(model, tools=None, checkpointer=None, system_prompt=None):
    """Compile a ReAct agent graph.

    Args:
        model: a LangChain chat model that supports ``bind_tools``.
        tools: optional list of LangChain tools. Defaults to the local wrappers;
            pass MCP-loaded tools (see ``mcp_client.load_mcp_tools``) to run the
            agent against the MCP server.
        checkpointer: optional LangGraph checkpointer (e.g. ``MemorySaver``) to
            persist conversation state. When provided, invoke with a
            ``{"configurable": {"thread_id": ...}}`` config and the agent
            remembers prior turns on that thread (multi-turn memory).
        system_prompt: optional instruction prepended (as a SystemMessage) to
            the messages sent to the model on each step, to steer its behavior.

    Returns:
        A compiled LangGraph graph with ``.invoke`` / ``.ainvoke``.
    """
    tools = tools or LOCAL_TOOLS
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState):
        messages = state["messages"]
        if system_prompt and not (messages and messages[0].type == "system"):
            messages = [SystemMessage(system_prompt), *messages]
        return {"messages": [model_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=_on_tool_error))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=checkpointer)
