"""MTEB-Gym leaderboard: label-free embedding-model rankings with a reliability report.

No server-side inference; every ranking ships with the judge's chance-corrected agreement against the
corpus's own human labels. A ranking without its kappa is not publishable here. The data file is
built by `python -m leaderboard.export` from gym records (see README.md).
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd

try:  # ZeroGPU hosting requires a registered @spaces.GPU function; never invoked.
    import spaces

    @spaces.GPU
    def _zerogpu_probe():
        return "ok"
except ImportError:  # local runs have no spaces package
    pass

DATA = json.loads((Path(__file__).parent / "data" / "leaderboard_export.json").read_text(encoding="utf-8"))

# Contract: every ranked corpus carries a reliability row. Kappa-only corpora (scored but not ranked)
# are allowed and appear in the Reliability tab.
_MISSING = sorted(set(DATA["corpora"]) - set(DATA["reliability"]))
assert not _MISSING, f"ranked corpora without reliability rows: {_MISSING}"

LOW_KAPPA = 0.20


def corpus_table(task: str):
    info = DATA["corpora"][task]
    rows = sorted(info["models"], key=lambda m: -m["rating"])
    df = pd.DataFrame(rows)
    df.insert(0, "rank", df["rating"].rank(method="min", ascending=False).astype(int))
    kappa = DATA["reliability"].get(task, {}).get("kappa")
    if kappa is None:
        banner = f"**{task}**: {info['n_queries']} frozen queries. No reliability row."
    elif kappa < LOW_KAPPA:
        banner = (
            f"⚠️ **{task}**: judge–qrels κ = {kappa} (near chance). This ranking should **not** be used "
            f"for model selection; it is shown for completeness. {info['n_queries']} frozen queries."
        )
    else:
        banner = (
            f"**{task}**: judge–qrels κ = {kappa} on this corpus's own human queries. "
            f"{info['n_queries']} frozen queries."
        )
    return banner, df


def reliability_table() -> pd.DataFrame:
    rows = [
        {"corpus": c, **{("validation_queries" if k == "n_queries" else k): x for k, x in v.items()}}
        for c, v in sorted(DATA["reliability"].items(), key=lambda kv: -kv[1]["kappa"])
    ]
    return pd.DataFrame(rows)


RELIABILITY_NOTE = (
    "κ is chance-corrected judge–qrels agreement on each corpus's own human queries "
    "(κ = 2p − 1 on committed verdicts over objectively decisive pairs; Bennett's S, not Cohen's κ). "
    "Rankings on corpora with κ near 0 carry no signal against human labels. A corpus listed here "
    "without a ranked tab was scored for reliability but not ranked. The validation_queries column "
    "counts the human queries behind the κ study, a separate set from the frozen synthetic queries "
    "behind the rankings."
)

_first = sorted(DATA["corpora"])[0]
_b0, _d0 = corpus_table(_first)
_commit = DATA["meta"]["experiment_commit"]

with gr.Blocks(title="MTEB-Gym leaderboard") as demo:
    gr.Markdown(
        "# MTEB-Gym: label-free embedding-model leaderboard\n"
        f"Judge `{DATA['meta']['judge']}` · frozen inputs at commit `{_commit}` · Bradley–Terry over "
        "both-order pairwise verdicts on frozen synthetic queries. Companion to the MTEB-Gym paper."
    )
    with gr.Tab("Rankings"):
        task = gr.Dropdown(sorted(DATA["corpora"]), value=_first, label="Corpus")
        banner = gr.Markdown(_b0)
        table = gr.Dataframe(value=_d0)
        task.change(corpus_table, inputs=task, outputs=[banner, table])
    with gr.Tab("Reliability"):
        gr.Markdown(RELIABILITY_NOTE)
        gr.Dataframe(value=reliability_table())

if __name__ == "__main__":
    demo.launch()
