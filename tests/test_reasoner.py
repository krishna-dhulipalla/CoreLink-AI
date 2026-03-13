import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.nodes.reasoner import _executor_max_tokens


def test_executor_max_tokens_is_raised_only_for_legal_tasks():
    assert _executor_max_tokens("legal") == 1500
    assert _executor_max_tokens("options") == 1000
    assert _executor_max_tokens("quantitative") == 1000
    assert _executor_max_tokens("general") == 1000
