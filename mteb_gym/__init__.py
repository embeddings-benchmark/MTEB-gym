"""MTEB Gym: evaluate embedding models on any corpus, without relevance labels."""

from importlib.metadata import PackageNotFoundError, version

from .cache import cache_files
from .judge import Grade, Verdict
from .llm import LLM, ChatClient, DecisionClient, DecisionModel, MockLLM
from .queries import Query
from .rank import ModelRating
from .results import Result, Results, load_results
from .run import predict, run
from .submit import submit
from .task_instructions import TASK_INSTRUCTIONS

try:
    __version__ = version("mteb-gym")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "run",
    "predict",
    "cache_files",
    "submit",
    "LLM",
    "DecisionModel",
    "MockLLM",
    "ChatClient",
    "DecisionClient",
    "Query",
    "Grade",
    "Verdict",
    "ModelRating",
    "Result",
    "Results",
    "load_results",
    "TASK_INSTRUCTIONS",
]
