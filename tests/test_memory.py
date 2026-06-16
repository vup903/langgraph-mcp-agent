"""Multi-turn memory: with a checkpointer, the agent remembers prior turns on a thread.

Hermetic — uses a trivial fake model (no tool calls, no API key).
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from langgraph_mcp_agent.graph import build_agent


class _FakeModel(BaseChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    @property
    def _llm_type(self):
        return "fake"


def test_memory_persists_across_turns():
    agent = build_agent(_FakeModel(), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    agent.invoke({"messages": [HumanMessage("first question")]}, cfg)
    out = agent.invoke({"messages": [HumanMessage("second question")]}, cfg)

    contents = [m.content for m in out["messages"]]
    # The earlier turn is still in the thread's state -> memory works.
    assert "first question" in contents
    assert "second question" in contents


def test_threads_are_isolated():
    agent = build_agent(_FakeModel(), checkpointer=MemorySaver())
    agent.invoke({"messages": [HumanMessage("alpha")]}, {"configurable": {"thread_id": "a"}})
    out_b = agent.invoke({"messages": [HumanMessage("beta")]}, {"configurable": {"thread_id": "b"}})

    contents = [m.content for m in out_b["messages"]]
    assert "beta" in contents
    assert "alpha" not in contents  # separate thread does not leak
