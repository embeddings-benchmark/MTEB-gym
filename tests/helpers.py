"""Shared test doubles and data."""

from mteb_gym.retrieval import Ranked


def make_corpus(n=40):
    topics = ["heart disease and statins", "vitamin D and asthma", "gut microbiome and fiber", "telomeres and stress"]
    return {f"D{i}": f"Document {i} about {topics[i % 4]}. " + "clinical evidence " * (i % 5 + 1) for i in range(n)}


def metas(names):
    """mteb's ModelMeta for each of `names`, in their order, as run() takes them."""
    import mteb

    return [mteb.get_model_meta(name) for name in names]


def fake_ranked(seed, queries, k=5):
    return [
        Ranked(q.qid, q.text, [f"D{seed}{j}" for j in range(k)], [f"result {seed}-{q.qid}-{j}" for j in range(k)])
        for q in queries
    ]


class Deciding:
    """A decision-only client: decide() and no chat(), as DecisionModel; grades, and picks a list, by word overlap with the query."""

    model = "mock-decider"

    def __init__(self):
        self.calls = 0

    def decide(self, kind, state, question, options=None):
        self.calls += 1
        if kind == "choice":
            query = state.split("Query: ", 1)[1].split("\n", 1)[0].lower().split()
            a, b = state.split("System B results:", 1)
            a, b = (sum(w in side.split("System A results:")[-1].lower() for w in query) for side in (a, b))
            return dict(zip(options, (1.0, 0.0, 0.0) if a > b else (0.0, 1.0, 0.0) if b > a else (0.0, 0.0, 1.0)))
        query = state.split("Query: ", 1)[1].split("\n", 1)[0].lower().split()
        doc = state.split("Document: ", 1)[1].lower()
        g = min(5, sum(w in doc for w in query))
        return {str(i): 1.0 if i == g else 0.0 for i in range(6)}
