from langgraph_mcp_agent import tools


def test_add():
    assert tools.add(2, 3) == 5


def test_multiply():
    assert tools.multiply(4, 5) == 20


def test_search_notes_key_hit():
    assert "Model Context Protocol" in tools.search_notes("what is mcp?")


def test_search_notes_token_overlap():
    assert "Retrieval-Augmented" in tools.search_notes("tell me about retrieved documents")


def test_search_notes_miss():
    assert tools.search_notes("zzzz") == "No matching note found."
