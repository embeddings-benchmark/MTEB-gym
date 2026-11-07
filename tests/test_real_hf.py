"""Direct loading against grades a server stored. Needs a GPU and those grades: MTEB_GYM_STORED_GRADES names their folder."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from mteb_gym import corpus
from mteb_gym.judge import _cut
from mteb_gym.llm import load_model
from mteb_gym.task_instructions import TASK_INSTRUCTIONS

STORED = os.environ.get("MTEB_GYM_STORED_GRADES")
QUESTION = (
    "How relevant is the document to the query, for this retrieval task? "
    "0 = not relevant at all, 5 = perfectly relevant: it fully satisfies the information need."
)


@pytest.mark.skipif(not STORED, reason="needs a GPU and stored grades")
@pytest.mark.parametrize(
    ("name", "file"),
    [("autotrust/JEV-9B", "jev-pointwise-jev9b.jsonl"), ("AutoTrust/jev-27b", "jev-pointwise-locked.jsonl")],
)
def test_a_model_loaded_directly_grades_as_it_does_served(name, file):
    rows = [json.loads(line) for line in (Path(STORED) / file).open()]
    random.Random(0).shuffle(rows)
    rows = rows[:200]
    task = corpus.load("SciFact")
    judge = load_model(name)
    grades = []
    for row in rows:
        query, doc = row["key"].split("\t")
        state = f"Retrieval task: {TASK_INSTRUCTIONS['SciFact']}\nQuery: {task.queries[query]}\nDocument: {_cut(task.docs[doc], 16000)}"
        grades.append(sum(int(k) * p for k, p in judge.decide("score", state, QUESTION).items()))
    grades, stored = np.array(grades), np.array([row["grade"] for row in rows])
    difference, pearson = np.abs(grades - stored), np.corrcoef(grades, stored)[0, 1]
    print(f"{name}: mean difference {difference.mean():.4f}, largest {difference.max():.4f}, pearson {pearson:.5f}")
    assert pearson > 0.999 and difference.max() < 0.1
