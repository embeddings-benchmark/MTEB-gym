#!/bin/bash
# Analysis pass over one corpus of a regen run, on the login node (CPU only; nothing here needs a GPU).
# Usage: bash run_analysis.sh <TaskName> [results dir] [judge] [out name]
#   e.g. bash run_analysis.sh NFCorpus                                          (27B judge, writes analysis_out/NFCorpus/)
#        bash run_analysis.sh NFCorpus $RUNS/results MiniMax-M2.7 NFCorpus_m27   (MiniMax record, writes analysis_out/NFCorpus_m27/)
# The synthetic-arm record is picked by task and judge name, so a second judge's record for the same task is not
# mistaken for the first's. Writes to tmp_regen/analysis_out/<out name>/: truth.json, original_queries.json, docs.json,
# scaling.{json,md}, seed_baseline.{json,md}, query_stats.{json,md}. Read-only on everything else. The level check
# (level_check.py) runs once over SUMMARY.jsonl from the README, not per task.
set -uo pipefail
TASK=${1:?task name}
OUT=${2:-/data/home/niklas/tejas/tmp_regen/results}
JUDGE=${3:-Qwen3.6-27B}
NAME=${4:-$TASK}
RUNS=/data/home/niklas/tejas/tmp_regen
PY=/data/home/niklas/adnan/gym-runs/venv-gpu/bin/python
export PYTHONPATH=$RUNS/MTEB-gym-v2 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error
A=$RUNS/analysis_out/$NAME; mkdir -p "$A"
REC=$(ls "$OUT"/records/"$TASK"__"$JUDGE"__*__q*-s*-*.json 2>/dev/null | grep -v original-queries | head -1)
[ -n "$REC" ] || { echo "no synthetic-arm record for $TASK judged by $JUDGE under $OUT/records"; exit 2; }
echo "record: $REC"

# official scores for the roster (mteb results cache, already warm), the corpus's own queries, and the documents
"$PY" - "$REC" "$A" <<'EOF'
import json, sys
from pathlib import Path
rec = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
from mteb_gym.agreement import fetch_truth
truth = fetch_truth(rec["config"]["models"], rec["task_name"])
if isinstance(truth, tuple):  # (scores, missing) in some versions; the analysis scripts want {model: score}
    truth = truth[0]
(out / "truth.json").write_text(json.dumps(truth, indent=1))
print("truth:", len(truth), "of", len(rec["config"]["models"]), "models anchored")
from mteb_gym.corpus import load
corp = load(rec["task_name"])
(out / "original_queries.json").write_text(json.dumps(corp.queries or {}, indent=1))
(out / "docs.json").write_text(json.dumps(corp.docs))
print("original queries:", len(corp.queries or {}), "| docs:", len(corp.docs))
EOF

cd "$RUNS/MTEB-gym-v2"
"$PY" -m analysis.scaling --output-folder "$OUT" --record "$REC" --truth "$A/truth.json" --out "$A/scaling.json" --draws 200 > "$A/scaling.md" 2> "$A/scaling.err"; echo "scaling exit $?"
"$PY" -m analysis.seed_baseline --output-folder "$OUT" --record "$REC" --truth "$A/truth.json" --out "$A/seed_baseline.json" > "$A/seed_baseline.md" 2> "$A/seed_baseline.err"; echo "seed_baseline exit $?"
"$PY" -m analysis.query_stats --output-folder "$OUT" --record "$REC" --original-queries "$A/original_queries.json" --docs "$A/docs.json" --out "$A/query_stats.json" > "$A/query_stats.md" 2> "$A/query_stats.err"; echo "query_stats exit $?"
echo "outputs:"; ls -la "$A" | awk '{print "  "$5" "$9}' | tail -n +2
