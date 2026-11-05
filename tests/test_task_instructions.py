import types

import pytest

from mteb_gym.judge import judge_system
from mteb_gym.llm import MockLLM
from mteb_gym.queries import QueryGenerator
from mteb_gym.run import resolve_instruction
from mteb_gym.task_instructions import TASK_INSTRUCTIONS


def test_descriptions_cover_the_paper_tasks():
    """Every evaluated task has a sentence, each names a real mteb task, and a corpus without one is an error."""
    import mteb

    for name in TASK_INSTRUCTIONS:
        assert mteb.get_task(name).metadata.name == name


def test_description_is_required():
    named = types.SimpleNamespace(name="ArguAna")
    assert resolve_instruction(None, named) == (TASK_INSTRUCTIONS["ArguAna"], "mteb_gym:task_instructions")
    assert resolve_instruction("Prefer replies that resolve the ticket", named)[1] == "config:task_instruction"
    with pytest.raises(ValueError, match="task_instruction"):
        resolve_instruction(None, types.SimpleNamespace(name="my-corpus"))
    gen = QueryGenerator(MockLLM(), task_instruction="Given a claim, find documents that refute the claim")
    assert "refute the claim" in gen.system and gen.params["task_instruction"]  # part of the query cache key
    assert "The retrieval task is: Given a claim" in judge_system("Given a claim, find documents that refute the claim")
