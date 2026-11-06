from mteb_gym.judge import judge_system
from mteb_gym.llm import MockLLM
from mteb_gym.queries import QueryGenerator
from mteb_gym.task_instructions import TASK_INSTRUCTIONS


def test_every_instruction_names_a_real_task():
    """Every key in the table is an mteb task."""
    import mteb

    for name in TASK_INSTRUCTIONS:
        assert mteb.get_task(name).metadata.name == name


def test_the_sentence_is_optional_and_used_only_when_given():
    plain = QueryGenerator(MockLLM())
    assert "retrieval task" not in plain.system and plain.params["task_instruction"] is None
    gen = QueryGenerator(MockLLM(), task_instruction="Given a claim, find documents that refute the claim")
    assert gen.system.count("The retrieval task is: Given a claim") == 1 and gen.params["task_instruction"]
    assert plain.prompts != gen.prompts
    assert "retrieval task" not in judge_system(None)
    assert "The retrieval task is: Given a claim" in judge_system("Given a claim, find documents that refute the claim")


def test_a_generators_filter_is_told_the_sentence_only_when_there_is_one():
    assert "retrieval task" not in QueryGenerator(MockLLM()).prompts[2]
    assert "find it" in QueryGenerator(MockLLM(), task_instruction="find it").prompts[2]
