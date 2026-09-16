#!/bin/bash
#SBATCH --job-name=gym-regen-397b
#SBATCH --partition=slinky
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/niklas/tejas/tmp_regen/%x-%j.log
# Stronger-judge pass over the SAME nano-tier queries and predictions as gym-regen-nano: judge
# Qwen3.5-397B-A17B-FP8 (tensor parallel over 8 GPUs), no generator server, no embedders, because
# both are served from the caches the 27B run wrote. Submit only after gym-regen-nano has finished:
# the driver exits 2 if the query cache is incomplete. $0: cluster weights, cluster GPUs.
set -uo pipefail
RUNS=/data/home/niklas/tejas/tmp_regen
PY=/data/home/niklas/adnan/gym-runs/venv-gpu/bin/python
CONDA_ENV=/home/niklas/.local/bin/miniconda3/envs/vllm-dsv4b
VLLM=$CONDA_ENV/bin/vllm
export JUDGE_MODEL=Qwen/Qwen3.5-397B-A17B-FP8
export GEN_MODEL=openai/gpt-oss-20b          # identity only: names the cached query sets, never called
export JUDGE_ONLY=1
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1
export PYTHONPATH=$RUNS/MTEB-gym-v2
# the driver uses the shared HF cache (datasets downloaded there on the login node; job 14253 showed that a
# symlinked HF_HOME breaks sentence-transformers model loading); only the generator server uses tmp_regen/hf
export PATH=$CONDA_ENV/bin:$PATH                                             # ninja + nvcc for kernel JIT
export VLLM_USE_FLASHINFER_SAMPLER=0                                         # sampling JIT needs curand.h, absent from the env
export LIBRARY_PATH=$CONDA_ENV/lib:${LIBRARY_PATH:-}                         # -lcudart at the GDN kernel link step
export LD_LIBRARY_PATH=$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}
echo "job $SLURM_JOB_ID on $(hostname)  $(date -Iseconds)"
cd "$RUNS/MTEB-gym-v2" && echo "gym at $(git rev-parse --short HEAD) ($(git branch --show-current))" && cd "$RUNS"
test -f "$RUNS/results/SUMMARY.jsonl" || { echo "FATAL: no results/SUMMARY.jsonl; run gym-regen-nano first"; exit 2; }

IFS=',' read -ra G <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-}}"
[ "${#G[@]}" -lt 8 ] && { echo "FATAL: need 8 GPUs, got: ${G[*]:-none}"; exit 1; }
echo "gpus: ${G[*]}"

"$VLLM" serve "$JUDGE_MODEL" --port 8000 --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 64 \
    --served-model-name "$JUDGE_MODEL" > "$RUNS/vllm-judge-$SLURM_JOB_ID.log" 2>&1 &
JP=$!
cleanup() { echo "tearing down vllm"; kill $JP 2>/dev/null; sleep 5; kill -9 $JP 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 120); do   # a 400 GB load takes a while
  curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1 && { echo "judge ready after $((i*10))s"; break; }
  kill -0 "$JP" 2>/dev/null || { echo "FATAL: judge process died; see its log"; exit 1; }
  sleep 10
  [ "$i" = 120 ] && { echo "FATAL: judge did not come up within 1200s"; exit 1; }
done
resp=$(curl -s -m 900 -w '\nHTTP %{http_code}' "http://localhost:8000/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
echo "  probe -> $(echo "$resp" | tail -1); body: $(echo "$resp" | head -c 300 | tr '\n' ' ')"
echo "$resp" | tail -1 | grep -q 'HTTP 200' || { echo "FATAL: probe call failed"; exit 1; }

export JUDGE_URL=http://localhost:8000/v1 GENERATOR_URL=http://127.0.0.1:9/v1
CUDA_VISIBLE_DEVICES="" "$PY" "$RUNS/regen_nano.py" "$RUNS/results"
echo "driver exit: $?  $(date -Iseconds)"
