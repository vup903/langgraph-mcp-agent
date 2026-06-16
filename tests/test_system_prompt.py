"""The optional system_prompt is prepended to what the model sees (hermetic)."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph_mcp_agent.graph import build_agent


class _EchoFirstMessage(BaseChatModel):
    """Echoes the first message it receives as '<type>:<content>' (no tool calls)."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        first = messages[0]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"{first.type}:{first.content}"))])

    @property
    def _llm_type(self):
        return "echo-first"


def test_system_prompt_is_prepended():
    agent = build_agent(_EchoFirstMessage(), system_prompt="You are concise.")
    out = agent.invoke({"messages": [HumanMessage("hi")]})
    assert out["messages"][-1].content == "system:You are concise."


def test_no_system_prompt_by_default():
    agent = build_agent(_EchoFirstMessage())
    out = agent.invoke({"messages": [HumanMessage("hi")]})
    assert out["messages"][-1].content == "human:hi"
