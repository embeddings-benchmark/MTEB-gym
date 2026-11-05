"""Shared test doubles and data."""

from mteb_gym.retrieval import Ranked


def make_corpus(n=40):
    topics = ["heart disease and statins", "vitamin D and asthma", "gut microbiome and fiber", "telomeres and stress"]
    return {f"D{i}": f"Document {i} about {topics[i % 4]}. " + "clinical evidence " * (i % 5 + 1) for i in range(n)}


def fake_ranked(seed, queries, k=5):
    return [
        Ranked(q.qid, q.text, [f"D{seed}{j}" for j in range(k)], [f"result {seed}-{q.qid}-{j}" for j in range(k)])
        for q in queries
    ]


class Deciding:
    """A decision-only client: decide() and no chat(), as DecisionModel; grades by word overlap with the query."""

    model = "mock-decider"

    def __init__(self):
        self.calls = 0

    def decide(self, kind, state, question, options=None):
        self.calls += 1
        query = state.split("Query: ", 1)[1].split("\n", 1)[0].lower().split()
        doc = state.split("Document: ", 1)[1].lower()
        g = min(5, sum(w in doc for w in query))
        return {str(i): 1.0 if i == g else 0.0 for i in range(6)}
