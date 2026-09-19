#!/bin/bash
#SBATCH --job-name=gym-regen-nano
#SBATCH --partition=slinky
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00                      # job 14359 ran with 6 h and hit the limit; job 14372 resumed from its caches
#SBATCH --output=/data/home/niklas/tejas/tmp_regen/%x-%j.log
# Nano-tier regeneration under mteb_gym at commit 71b2cf4 (branch tejas/analysis-scripts = main + reliability + analysis/).
# GPU 0: judge Qwen3.6-27B (shared HF cache). GPU 1: generator openai/gpt-oss-20b (own HF_HOME, a
# different family from the judge and from every entrant) plus the embedders, which is why the
# generator server leaves most of its GPU free. Everything written stays under tmp_regen/.
set -uo pipefail
RUNS=/data/home/niklas/tejas/tmp_regen
PY=/data/home/niklas/adnan/gym-runs/venv-gpu/bin/python                    # torch cu128 (base conda is cu130, too new for the driver)
CONDA_ENV=/home/niklas/.local/bin/miniconda3/envs/vllm-dsv4b
VLLM=$CONDA_ENV/bin/vllm                                                    # vllm 0.22.1, torch cu128
export JUDGE_MODEL=Qwen/Qwen3.6-27B
export GEN_MODEL=openai/gpt-oss-20b
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1
export PYTHONPATH=$RUNS/MTEB-gym-v2
# the driver uses the shared HF cache (datasets downloaded there on the login node; job 14253 showed that a
# symlinked HF_HOME breaks sentence-transformers model loading); only the generator server uses tmp_regen/hf
# Toolchain for the kernels vllm JIT-builds on first use (three failed jobs taught each line):
export PATH=$CONDA_ENV/bin:$PATH                                             # ninja + nvcc (job 14196: 'ninja' not found)
export VLLM_USE_FLASHINFER_SAMPLER=0                                         # sampling JIT needs curand.h, absent from the env (job 14203)
export LIBRARY_PATH=$CONDA_ENV/lib:${LIBRARY_PATH:-}                         # -lcudart at the GDN kernel link step (job 14204)
export LD_LIBRARY_PATH=$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}
echo "job $SLURM_JOB_ID on $(hostname)  $(date -Iseconds)"
cd "$RUNS/MTEB-gym-v2" && echo "gym at $(git rev-parse --short HEAD) ($(git branch --show-current))" && cd "$RUNS"

VLLM_PY=$CONDA_ENV/bin/python
for pair in "gym:$PY" "vllm:$VLLM_PY"; do
  name=${pair%%:*}; interp=${pair#*:}
  "$interp" -c "import torch,sys; ok=torch.cuda.is_available(); print(f'  $name env: torch {torch.__version__} cuda_available={ok} devices={torch.cuda.device_count()}'); sys.exit(0 if ok else 1)" \
    || { echo "FATAL: $name env cannot initialise CUDA on this node"; exit 1; }
done

# slurm hands absolute GPU indices in CUDA_VISIBLE_DEVICES; the node is not cgroup-isolated, so never hardcode 0/1
IFS=',' read -ra G <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-}}"
[ "${#G[@]}" -lt 2 ] && { echo "FATAL: need 2 GPUs, got: ${G[*]:-none}"; exit 1; }
echo "gpus: judge=${G[0]} generator+embedders=${G[1]}"

CUDA_VISIBLE_DEVICES=${G[0]} "$VLLM" serve "$JUDGE_MODEL" --port 8000 \
    --gpu-memory-utilization 0.88 --max-model-len 8192 --max-num-seqs 64 \
    --served-model-name "$JUDGE_MODEL" > "$RUNS/vllm-judge-$SLURM_JOB_ID.log" 2>&1 &
JP=$!
CUDA_VISIBLE_DEVICES=${G[1]} HF_HOME=$RUNS/hf "$VLLM" serve "$GEN_MODEL" --port 8001 \
    --gpu-memory-utilization 0.40 --max-model-len 8192 --max-num-seqs 64 \
    --reasoning-parser openai_gptoss --served-model-name "$GEN_MODEL" > "$RUNS/vllm-gen-$SLURM_JOB_ID.log" 2>&1 &
GP=$!
cleanup() { echo "tearing down vllm"; kill $JP $GP 2>/dev/null; sleep 5; kill -9 $JP $GP 2>/dev/null; }
trap cleanup EXIT

wait_up() {  # wait_up <port> <pid> <name>
  for i in $(seq 1 90); do
    curl -sf "http://localhost:$1/v1/models" >/dev/null 2>&1 && { echo "$3 ready after $((i*10))s"; return 0; }
    kill -0 "$2" 2>/dev/null || { echo "FATAL: $3 process died; see its log"; return 1; }
    sleep 10
  done
  echo "FATAL: $3 did not come up within 900s"; return 1
}
wait_up 8000 $JP judge || exit 1
wait_up 8001 $GP generator || exit 1

# one real call per endpoint before the run, printed with status and body, so a dead server fails here
# and not as 10,000 silent ties
probe() {  # probe <port> <model>
  local resp
  resp=$(curl -s -m 900 -w '\nHTTP %{http_code}' "http://localhost:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":64}")
  echo "  probe $2 -> $(echo "$resp" | tail -1); body: $(echo "$resp" | head -c 300 | tr '\n' ' ')"
  echo "$resp" | tail -1 | grep -q 'HTTP 200'
}
probe 8000 "$JUDGE_MODEL" || { echo "FATAL: probe call to $JUDGE_MODEL failed"; exit 1; }
probe 8001 "$GEN_MODEL" || { echo "FATAL: probe call to $GEN_MODEL failed"; exit 1; }

export JUDGE_URL=http://localhost:8000/v1 GENERATOR_URL=http://localhost:8001/v1
CUDA_VISIBLE_DEVICES=${G[1]} "$PY" "$RUNS/regen_nano.py" "$RUNS/results"
echo "driver exit: $?  $(date -Iseconds)"
