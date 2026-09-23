#!/usr/bin/env bash
# Unified launcher for CC-OPD training on local GPU servers.
#
# This is the single entry point: select method and hyperparameters
# via environment variables, then run this script.
#
# REQUIRED environment variables:
#   STUDENT_MODEL_PATH   - local model path or Hugging Face repo ID
#   TEACHER_MODEL_PATH   - local model path or Hugging Face repo ID
#   TRAIN_DATA           - path to the training parquet
#   VAL_DATA             - path to the validation parquet
#
# ALGORITHM CHOICE (optional):
#   OPD_METHOD               - loo | opd                (default: loo)
#   RHO                       - rubric sample ratio (0,1] (default: 1.0)
#   LAMBDA                    - lambda scaling on CC delta (default: 2.0)
#   DELTA_CLIP                - per-token clip on CC delta (default: 5.0)
#   CC_RUBRIC_SAMPLE_SEED     - rng seed for rubric subset sampling (default: 21)
#   CC_DELTA_KEY              - tensor batch key for CC deltas (default: cc_delta_log_probs)
#   CC_MAX_ONLINE_JOBS        - cap on counterfactual scoring jobs per batch (default: 8192)
#   CC_ONLINE_MICRO_BATCH_SIZE - micro-batch size for counterfactual teacher scoring (default: 512)
#   CC_OPD_ENV_NAME           - rollout env name (default: hir16k_instruction)
#
# TRAINING SETTINGS (optional):
#   TOTAL_TRAINING_STEPS - hard step cap       (default: 1500 for loo, 6000 for opd)
#   TOTAL_EPOCHS         - dataloader capacity (default: 30 for loo, 60 for opd)
#   SAVE_FREQ            - checkpoint save interval       (default: 20)
#   TEST_FREQ            - validation interval            (default: 20)
#   TRAIN_BATCH_SIZE     - prompts per training step      (default: 128)
#   PPO_MINI_BATCH_SIZE  - PPO mini-batch                 (default: 128)
#   PPO_MICRO_BATCH_SIZE_PER_GPU                          (default: 4)
#   ENV_ROLLOUT_N        - rollouts per prompt            (default: 8)
#   MAX_RESPONSE_LENGTH  - max response token length      (default: 4096)
#   VAL_N                - avg-at-N validation samples    (default: 3)
#
# CLUSTER TOPOLOGY (optional, single-node defaults):
#   WORLD_SIZE           - number of nodes  (default: 1)
#   GPUS_PER_NODE        - GPUs per node    (default: 8)
#   MASTER_ADDR          - head node IP     (default: 127.0.0.1)
#   RANK                 - this node's rank (default: 0)
#
# OUTPUT PATHS (optional):
#   CC_OPD_DATA_BASE     - intermediate data staging dir  (default: ${HOME}/cc_opd/data)
#   CC_OPD_EXP_BASE      - checkpoints + logs dir         (default: ${HOME}/cc_opd/experiments)
#   BENCHMARK_ROOT       - benchmark root (required only if benchmark eval enabled)
#   CC_OPD_BENCHMARK_EVAL_ENABLED - true|false            (default: false)
#
# JOB IDENTITY (optional):
#   EXPERIMENT_NAME      - human-readable name            (default: derived from method)
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${LAUNCH_DIR}/.." && pwd)"
export CODE_DIR="${CODE_DIR:-${RELEASE_ROOT}/code}"
export LAUNCH_DIR

# Required paths for a real training run.
if [[ "${CC_OPD_PRINT_CONFIG_ONLY:-false}" != "true" ]]; then
    for required_var in STUDENT_MODEL_PATH TEACHER_MODEL_PATH TRAIN_DATA VAL_DATA; do
        if [[ -z "${!required_var:-}" ]]; then
            echo "[FATAL] required environment variable ${required_var} is not set." >&2
            echo "        Either edit env.example.sh / source env.sh, or export it manually." >&2
            exit 1
        fi
    done
fi

# Method choice and LOO knobs
export OPD_METHOD="${OPD_METHOD:-loo}"
case "${OPD_METHOD}" in
    loo|opd) ;;
    *)
        echo "[FATAL] OPD_METHOD='${OPD_METHOD}' is not supported." >&2
        echo "        Valid choices: loo | opd" >&2
        exit 1
        ;;
esac
if [[ "${OPD_METHOD}" == "loo" ]]; then
    export CC_ENABLED="true" CC_ONLINE_ENABLED="true"
    DEFAULT_TRAINING_STEPS=1500
    DEFAULT_EPOCHS=30
else
    export CC_ENABLED="false" CC_ONLINE_ENABLED="false"
    DEFAULT_TRAINING_STEPS=6000
    DEFAULT_EPOCHS=60
fi

export CC_RUBRIC_SAMPLE_RATIO="${RHO:-${CC_RUBRIC_SAMPLE_RATIO:-1.0}}"
export CC_LAMBDA="${LAMBDA:-${CC_LAMBDA:-2.0}}"
export CC_DELTA_CLIP="${DELTA_CLIP:-${CC_DELTA_CLIP:-5.0}}"
export CC_RUBRIC_SAMPLE_SEED="${CC_RUBRIC_SAMPLE_SEED:-21}"
export CC_DELTA_KEY="${CC_DELTA_KEY:-cc_delta_log_probs}"
export CC_MAX_ONLINE_JOBS="${CC_MAX_ONLINE_JOBS:-8192}"
export CC_ONLINE_MICRO_BATCH_SIZE="${CC_ONLINE_MICRO_BATCH_SIZE:-512}"
export CC_OPD_ENV_NAME="${CC_OPD_ENV_NAME:-hir16k_instruction}"

# Training settings
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${DEFAULT_TRAINING_STEPS}}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-${DEFAULT_EPOCHS}}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
export ENV_ROLLOUT_N="${ENV_ROLLOUT_N:-8}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export VAL_N="${VAL_N:-3}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
export VAL_TOP_P="${VAL_TOP_P:-1}"
export VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-4096}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-false}"

# Cluster topology
WORLD_SIZE="${WORLD_SIZE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# Output paths
export CC_OPD_DATA_BASE="${CC_OPD_DATA_BASE:-${HOME}/cc_opd/data}"
export CC_OPD_EXP_BASE="${CC_OPD_EXP_BASE:-${HOME}/cc_opd/experiments}"
export CC_OPD_BENCHMARK_EVAL_ENABLED="${CC_OPD_BENCHMARK_EVAL_ENABLED:-false}"
export CC_OPD_PREPARE_DATA="${CC_OPD_PREPARE_DATA:-false}"
export CC_OPD_PREPARE_BENCHMARKS="${CC_OPD_PREPARE_BENCHMARKS:-false}"
export CC_OPD_RESOLVE_MODELS="${CC_OPD_RESOLVE_MODELS:-false}"

if [[ "${CC_OPD_BENCHMARK_EVAL_ENABLED}" == "true" && -z "${BENCHMARK_ROOT:-}" ]]; then
    echo "[FATAL] CC_OPD_BENCHMARK_EVAL_ENABLED=true but BENCHMARK_ROOT is not set." >&2
    exit 1
fi
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"

# Job identity
GPU_NUMS=$((WORLD_SIZE * GPUS_PER_NODE))
DATE="$(date +%Y%m%d%H%M)"
if [[ "${OPD_METHOD}" == "loo" ]]; then
    DEFAULT_NAME="cc_opd_loo_rho$(printf '%03d' $(awk -v r="${CC_RUBRIC_SAMPLE_RATIO}" 'BEGIN{printf "%d", r*100}'))_lam${CC_LAMBDA}_clip${CC_DELTA_CLIP}"
else
    DEFAULT_NAME="opd_baseline"
fi
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_NAME}}"
export VERSION="${VERSION:-${OPD_METHOD}_${TOTAL_TRAINING_STEPS}step}"
export JOB_NAME="${EXPERIMENT_NAME}__gpus__${GPU_NUMS}__${DATE}__${VERSION}"

export WORLD_SIZE GPUS_PER_NODE RANK MASTER_ADDR

echo "=========================================="
echo "  OPD local training"
echo "=========================================="
echo "JOB_NAME       : ${JOB_NAME}"
echo "OPD_METHOD     : ${OPD_METHOD}"
echo "CC_ENABLED     : ${CC_ENABLED}"
echo "CC_ONLINE_ENABLED: ${CC_ONLINE_ENABLED}"
if [[ "${OPD_METHOD}" == "loo" ]]; then
    echo "rho (sample ratio): ${CC_RUBRIC_SAMPLE_RATIO}"
    echo "lambda         : ${CC_LAMBDA}"
    echo "delta_clip     : ${CC_DELTA_CLIP}"
fi
echo "STEPS / EPOCHS : ${TOTAL_TRAINING_STEPS} / ${TOTAL_EPOCHS}"
echo "Batch          : train=${TRAIN_BATCH_SIZE}  ppo_mini=${PPO_MINI_BATCH_SIZE}  micro/gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
echo "Rollout N      : ${ENV_ROLLOUT_N}"
echo "Topology       : WORLD_SIZE=${WORLD_SIZE} nodes x GPUS_PER_NODE=${GPUS_PER_NODE} = ${GPU_NUMS} GPU"
echo "Rank/Master    : RANK=${RANK}  MASTER_ADDR=${MASTER_ADDR}"
echo "Student model  : ${STUDENT_MODEL_PATH:-<unset>}"
echo "Teacher model  : ${TEACHER_MODEL_PATH:-<unset>}"
echo "TRAIN / VAL    : ${TRAIN_DATA:-<unset>} / ${VAL_DATA:-<unset>}"
echo "Benchmark eval : ${CC_OPD_BENCHMARK_EVAL_ENABLED}  (root: ${BENCHMARK_ROOT:-<unset>})"
echo "Exp dir        : ${CC_OPD_EXP_BASE}/${JOB_NAME}"
echo "=========================================="

if [[ "${CC_OPD_PRINT_CONFIG_ONLY:-false}" == "true" ]]; then
    exit 0
fi

exec python3 "${LAUNCH_DIR}/entry.py" \
    --world_size "${WORLD_SIZE}" \
    --gpus_per_node "${GPUS_PER_NODE}" \
    --job_name "${JOB_NAME}"
