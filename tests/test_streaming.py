"""Streaming surfaces each step (model / tools) as the agent runs (hermetic)."""
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph_mcp_agent.graph import build_agent, stream_steps


class _FakeToolThenAnswer(BaseChatModel):
    responses: List[AIMessage]

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self):
        return "fake-stream"


def test_stream_yields_model_and_tool_steps():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        AIMessage(content="2 + 3 = 5"),
    ]
    agent = build_agent(_FakeToolThenAnswer(responses=responses))
    steps = list(stream_steps(agent, [HumanMessage("add 2 and 3")]))

    nodes = [node for node, _ in steps]
    assert "model" in nodes and "tools" in nodes
    assert nodes.count("model") >= 2  # model ran before and after the tool


def test_stream_no_tool_single_model_step():
    agent = build_agent(_FakeToolThenAnswer(responses=[AIMessage(content="hi")]))
    nodes = [node for node, _ in stream_steps(agent, [HumanMessage("hi")])]
    assert nodes == ["model"]
