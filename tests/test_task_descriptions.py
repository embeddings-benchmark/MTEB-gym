import types

import pytest

from mteb_gym.judge import judge_system
from mteb_gym.llm import MockLLM
from mteb_gym.queries import QueryGenerator
from mteb_gym.run import resolve_description
from mteb_gym.task_descriptions import TASK_DESCRIPTIONS


def test_descriptions_cover_the_paper_tasks():
    """Every evaluated task has a sentence, each names a real mteb task, and a corpus without one is an error."""
    mteb = pytest.importorskip("mteb")
    assert len(TASK_DESCRIPTIONS) == 28
    for name in TASK_DESCRIPTIONS:
        assert mteb.get_task(name).metadata.name == name


def test_description_is_required():
    named = types.SimpleNamespace(name="ArguAna")
    assert resolve_description(None, named) == (TASK_DESCRIPTIONS["ArguAna"], "mteb_gym:task_descriptions")
    assert resolve_description("Prefer replies that resolve the ticket", named)[1] == "config:task_description"
    with pytest.raises(ValueError, match="task_description"):
        resolve_description(None, types.SimpleNamespace(name="my-corpus"))
    gen = QueryGenerator(MockLLM(), task_description="Given a claim, find documents that refute the claim")
    assert "refute the claim" in gen.system and gen.params["task_description"]  # part of the query cache key
    assert "The retrieval task is: Given a claim" in judge_system("Given a claim, find documents that refute the claim")
