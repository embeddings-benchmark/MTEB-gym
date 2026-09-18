#!/bin/bash
#SBATCH --job-name=gym-regen-m27
#SBATCH --partition=slinky
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/data/home/niklas/tejas/tmp_regen/%x-%j.log
# Second-judge pass over the SAME queries and predictions as gym-regen-nano (both cached by identity):
# judge MiniMaxAI/MiniMax-M2.7 (FP8, 4 x H100, tensor parallel), a family different from the 27B judge,
# the gpt-oss generator and every entrant. No generator server, no embedders. The model always emits a
# <think> block before its answer; the gym takes the last JSON object in the reply, so the cap is raised
# to leave room for the reasoning. TASKS limits the pass (NFCorpus first: the level-check corpus).
set -uo pipefail
RUNS=/data/home/niklas/tejas/tmp_regen
PY=/data/home/niklas/adnan/gym-runs/venv-gpu/bin/python
CONDA_ENV=/home/niklas/.local/bin/miniconda3/envs/vllm-dsv4b
VLLM=$CONDA_ENV/bin/vllm
NV=$CONDA_ENV/lib/python3.12/site-packages/nvidia
export JUDGE_MODEL=MiniMaxAI/MiniMax-M2.7
export GEN_MODEL=openai/gpt-oss-20b                 # cache key only; never called (JUDGE_ONLY)
export JUDGE_ONLY=1 JUDGE_MAX_TOKENS=6144 TASKS=${TASKS:-NFCorpus} ARMS=${ARMS:-synthetic,original}
# 16 and 48 workers gave the same ~25 calls/min (generation bound), 48 just tripled the per-call latency and
# one call passed the client's 120 s timeout 4 times, which ends the arm (job 14710, original arm, 10.6 h in)
export WORKERS=${WORKERS:-24} JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-900}
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1
export PYTHONPATH=$RUNS/MTEB-gym-v2
# Toolchain for the kernels vllm JIT-builds on first use. The conda env has nvcc but not the CUDA headers;
# the torch cu128 wheels ship them under site-packages/nvidia (job 14708: cuda_fp16.h, curand_kernel.h,
# cublasLt.h not found while flashinfer built fp8_blockscale_gemm_90). Unversioned .so links for the linker
# live in tmp_regen/cudalibs (the wheels only carry .so.12 / .so.10).
export PATH=$CONDA_ENV/bin:$PATH
export CPATH=$NV/cuda_runtime/include:$NV/curand/include:$NV/cublas/include:$CONDA_ENV/targets/x86_64-linux/include:${CPATH:-}
export CPLUS_INCLUDE_PATH=$CPATH C_INCLUDE_PATH=$CPATH
export LIBRARY_PATH=$CONDA_ENV/lib:$RUNS/cudalibs:$NV/cuda_runtime/lib:$NV/curand/lib:$NV/cublas/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_ENV/lib:$RUNS/cudalibs:$NV/cuda_runtime/lib:$NV/curand/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
# and keep the FP8 block-scaled GEMMs on the kernels compiled into vllm (CUTLASS, Triton) instead of JIT ones
export VLLM_DISABLED_KERNELS=FlashInferFp8BlockScaledMMKernel,FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,FlashInferFP8ScaledMMLinearKernel
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_MOE_FP8=0
# per-job kernel caches under the run folder: on slinky-0 /tmp/triton holds entries owned by another account
# and the worker died on 'Permission denied' at the profile run (job 14731)
export TRITON_CACHE_DIR=$RUNS/kernel_cache/triton VLLM_CACHE_ROOT=$RUNS/kernel_cache/vllm TORCHINDUCTOR_CACHE_DIR=$RUNS/kernel_cache/inductor
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
echo "job $SLURM_JOB_ID on $(hostname)  $(date -Iseconds)  tasks=$TASKS"
cd "$RUNS/MTEB-gym-v2" && echo "gym at $(git rev-parse --short HEAD) ($(git branch --show-current))" && cd "$RUNS"

IFS=',' read -ra G <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-}}"
[ "${#G[@]}" -lt 4 ] && { echo "FATAL: need 4 GPUs, got: ${G[*]:-none}"; exit 1; }
echo "gpus: ${G[*]}"

# stale workers from an earlier job of mine on this node would hold GPU memory (the node is not cgroup-isolated);
# sweep only MiniMax servers (and their process group) before launch; never touch other users' workers
for p in $(pgrep -f "vllm serve $JUDGE_MODEL" 2>/dev/null); do [ "$p" != "$$" ] && { echo "killing stale server $p"; kill -9 -- -"$(ps -o pgid= -p "$p" | tr -d " ")" 2>/dev/null; kill -9 "$p" 2>/dev/null; }; done
sleep 3
setsid "$VLLM" serve "$JUDGE_MODEL" --port 8000 --tensor-parallel-size 4 --trust-remote-code \
    --gpu-memory-utilization 0.90 --max-model-len 12288 --max-num-seqs 64 \
    --served-model-name "$JUDGE_MODEL" > "$RUNS/vllm-judge-$SLURM_JOB_ID.log" 2>&1 &
JP=$!
cleanup() { echo "tearing down vllm (process group)"; kill -- -$JP 2>/dev/null; sleep 8; kill -9 -- -$JP 2>/dev/null; kill -9 $JP 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 360); do   # weights load in a minute once cached; CUDA graph capture stalled on slinky-0 (job 14733): allow 60 min
  curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1 && { echo "judge ready after $((i*10))s"; break; }
  kill -0 "$JP" 2>/dev/null || { echo "FATAL: judge process died; see vllm-judge-$SLURM_JOB_ID.log"; exit 1; }
  [ "$i" -eq 360 ] && { echo "FATAL: judge did not come up within 3600s"; exit 1; }
  sleep 10
done

# one real call, printed with status and body, so a dead or truncating server fails here and not as ties
resp=$(curl -s -m 900 -w '\nHTTP %{http_code}' "http://localhost:8000/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with strict JSON: {\\\"ok\\\": true}\"}],\"max_tokens\":$JUDGE_MAX_TOKENS}")
echo "  probe -> $(echo "$resp" | tail -1); body: $(echo "$resp" | head -c 600 | tr '\n' ' ')"
echo "$resp" | tail -1 | grep -q 'HTTP 200' || { echo "FATAL: probe call failed"; exit 1; }

export JUDGE_URL=http://localhost:8000/v1 GENERATOR_URL=http://127.0.0.1:9/v1
"$PY" "$RUNS/regen_nano.py" "$RUNS/results"
echo "driver exit: $?  $(date -Iseconds)"
