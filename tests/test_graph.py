"""Exercise the full LangGraph tool-calling loop with a scripted fake model.

No API key required: a tiny fake chat model emits a tool call, the graph routes
to the ToolNode (which runs the real `add` tool), then the model emits the final
answer. This validates the graph wiring end-to-end and keeps CI hermetic.
"""
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph_mcp_agent.graph import build_agent


class FakeToolCallingModel(BaseChatModel):
    """Returns scripted AIMessages in order; supports bind_tools (a no-op)."""

    responses: List[AIMessage]

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Pop until the last response, then keep returning the final one.
        msg = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-model"


def test_agent_runs_tool_loop():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage(content="2 + 3 = 5"),
    ]
    agent = build_agent(FakeToolCallingModel(responses=responses))

    out = agent.invoke({"messages": [HumanMessage("please add 2 and 3")]})

    # The real `add` tool ran inside the graph and produced a ToolMessage of "5".
    assert any(getattr(m, "type", None) == "tool" and "5" in m.content for m in out["messages"])
    # The model's final answer is the last message.
    assert "5" in out["messages"][-1].content


def test_agent_without_tool_calls_terminates():
    agent = build_agent(FakeToolCallingModel(responses=[AIMessage(content="hello, no tools needed")]))
    out = agent.invoke({"messages": [HumanMessage("just say hi")]})
    assert out["messages"][-1].content == "hello, no tools needed"
