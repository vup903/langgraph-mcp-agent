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


def test_search_notes_ranks_by_overlap():
    # "standard" hits only the mcp note (1 token); "retrieved"+"documents" hit
    # the rag note (2 tokens). The old "first overlap wins" logic returned mcp
    # because it comes first; ranking by overlap correctly returns rag instead.
    result = tools.search_notes("standard retrieved documents")
    assert "Retrieval-Augmented" in result


def test_list_topics():
    topics = tools.list_topics()
    assert "mcp" in topics and "rag" in topics and "langgraph" in topics
