"""A failing tool is handled gracefully: the agent gets an error ToolMessage and continues."""
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool as as_tool

from langgraph_mcp_agent.graph import build_agent


@as_tool
def boom(x: int) -> int:
    """A tool that always raises (to test error handling)."""
    raise ValueError("kaboom")


class _Caller(BaseChatModel):
    responses: List[AIMessage]

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self):
        return "caller"


def test_tool_error_is_handled_gracefully():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "boom", "args": {"x": 1}, "id": "c1"}]),
        AIMessage(content="Sorry, the tool failed."),
    ]
    agent = build_agent(_Caller(responses=responses), tools=[boom])

    # Should NOT raise — the ToolNode catches the error and feeds it back.
    out = agent.invoke({"messages": [HumanMessage("use boom")]})

    tool_msgs = [m for m in out["messages"] if getattr(m, "type", None) == "tool"]
    assert tool_msgs, "expected an error ToolMessage"
    assert any("kaboom" in m.content for m in tool_msgs)
    assert out["messages"][-1].content == "Sorry, the tool failed."
