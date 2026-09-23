#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export RAY_memory_usage_threshold=0.99
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT_MS=7200000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_DEBUG=WARN
export NCCL_IBEXT_DISABLE=${NCCL_IBEXT_DISABLE:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1}

JOB_NAME="${JOB_NAME:?JOB_NAME required}"
WORLD_SIZE="${WORLD_SIZE:-1}"
umask 077
LAUNCH_DIR="${LAUNCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_DIR="${CODE_DIR:-$(cd "${LAUNCH_DIR}/../code" && pwd)}"

DATA_BASE="${CC_OPD_DATA_BASE:-/tmp/cc_opd_hir16k_data/${JOB_NAME}}"
EXP_BASE="${CC_OPD_EXP_BASE:-/tmp/cc_opd_experiments}"
CC_OPD_ENV_NAME="${CC_OPD_ENV_NAME:-hir16k_instruction}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${DATA_BASE}/Benchmark}"
BENCHMARK_VAL_DATA="${BENCHMARK_VAL_DATA:-${DATA_BASE}/instruction_benchmarks_val.parquet}"
EVAL_BENCHMARKS_CONFIG="${EVAL_BENCHMARKS_CONFIG:-${LAUNCH_DIR}/config/eval_benchmarks.json}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-4096}"
CC_OPD_BENCHMARK_EVAL_ENABLED="${CC_OPD_BENCHMARK_EVAL_ENABLED:-false}"
BENCHMARK_EVAL_RUN_EVALUATORS="${BENCHMARK_EVAL_RUN_EVALUATORS:-${CC_OPD_BENCHMARK_EVAL_ENABLED}}"
BENCHMARK_EVAL_FAIL_ON_ERROR="${BENCHMARK_EVAL_FAIL_ON_ERROR:-true}"
BENCHMARK_EVAL_STRIP_THINKING="${BENCHMARK_EVAL_STRIP_THINKING:-true}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
ENV_ROLLOUT_N="${ENV_ROLLOUT_N:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-false}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
OPD_METHOD="${OPD_METHOD:-loo}"
case "${OPD_METHOD}" in
  loo) CC_ENABLED=true; CC_ONLINE_ENABLED=true; DEFAULT_TRAINING_STEPS=1500; DEFAULT_EPOCHS=30 ;;
  opd) CC_ENABLED=false; CC_ONLINE_ENABLED=false; DEFAULT_TRAINING_STEPS=6000; DEFAULT_EPOCHS=60 ;;
  *) echo "[FATAL] OPD_METHOD must be loo or opd, got '${OPD_METHOD}'" >&2; exit 1 ;;
esac
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${DEFAULT_TRAINING_STEPS}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${DEFAULT_EPOCHS}}"

if [[ "${CC_OPD_PRINT_CONFIG_ONLY:-false}" == "true" ]]; then
  echo "OPD_METHOD=${OPD_METHOD}"
  echo "algorithm.adv_estimator=opd"
  echo "algorithm.opd.cc.enabled=${CC_ENABLED}"
  echo "algorithm.opd.cc.online.enabled=${CC_ONLINE_ENABLED}"
  echo "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  echo "trainer.total_epochs=${TOTAL_EPOCHS}"
  exit 0
fi

STUDENT_MODEL_ID="${STUDENT_MODEL_ID:-Qwen/Qwen2.5-1.5B}"
TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_ENV_FILE="${DATA_BASE}/model_paths.env"

TRAIN_DATA="${TRAIN_DATA:-${DATA_BASE}/hir16k_train.parquet}"
VAL_DATA="${VAL_DATA:-${BENCHMARK_VAL_DATA}}"

CKPTS_DIR="${EXP_BASE}/${JOB_NAME}"
LOG_DIR="${CKPTS_DIR}/logs"
TB_DIR="${CKPTS_DIR}/tensorboard"
VAL_GENERATION_DIR="${VAL_GENERATION_DIR:-${CKPTS_DIR}/validation_generations}"
BENCHMARK_EVAL_OUTPUT_DIR="${BENCHMARK_EVAL_OUTPUT_DIR:-${CKPTS_DIR}/benchmark_eval}"
HYDRA_RUN_DIR="/tmp/hydra_run/${JOB_NAME}_$$_$(date +%s%N)"

mkdir -p -m 700 "${DATA_BASE}" "${CKPTS_DIR}" "${LOG_DIR}" "${TB_DIR}" "${VAL_GENERATION_DIR}" "${BENCHMARK_EVAL_OUTPUT_DIR}" "${HYDRA_RUN_DIR}"
chmod 700 "${DATA_BASE}" "${CKPTS_DIR}" "${LOG_DIR}" "${TB_DIR}" "${VAL_GENERATION_DIR}" "${BENCHMARK_EVAL_OUTPUT_DIR}" "${HYDRA_RUN_DIR}"
export TENSORBOARD_DIR="${TB_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

if [[ "${CC_OPD_PREPARE_DATA:-true}" == "true" && ! -e "${TRAIN_DATA}" ]]; then
  echo "[data] preparing HIR-16K into ${DATA_BASE}"
  DATA_PREP_ARGS=(--output-dir "${DATA_BASE}" --val-size "${HIR16K_VAL_SIZE:-0}")
  if [[ -n "${HIR16K_LIMIT:-}" ]]; then
    DATA_PREP_ARGS+=(--limit "${HIR16K_LIMIT}")
  fi
  if [[ -n "${HIR16K_INPUT_JSONL:-}" ]]; then
    DATA_PREP_ARGS+=(--input-jsonl "${HIR16K_INPUT_JSONL}")
  fi
  python3 "${LAUNCH_DIR}/scripts/prepare_hir16k.py" "${DATA_PREP_ARGS[@]}"
fi

if [[ "${CC_OPD_PREPARE_BENCHMARKS:-true}" == "true" && ! -e "${BENCHMARK_VAL_DATA}" ]]; then
  if [[ ! -d "${BENCHMARK_ROOT}" ]]; then
    echo "[FATAL] BENCHMARK_ROOT is required to prepare benchmark validation data: ${BENCHMARK_ROOT}" >&2
    echo "        Expected a benchmark root directory (e.g. IFEval/IFBench/MulDimIF) at this path." >&2
    exit 1
  fi
  echo "[data] preparing IFEval/IFBench/MulDimIF benchmark validation into ${DATA_BASE}"
  BENCHMARK_PREP_ARGS=(--benchmark-root "${BENCHMARK_ROOT}" --output-dir "${DATA_BASE}")
  if [[ -n "${BENCHMARK_LIMIT_PER_BENCHMARK:-}" ]]; then
    BENCHMARK_PREP_ARGS+=(--limit-per-benchmark "${BENCHMARK_LIMIT_PER_BENCHMARK}")
  fi
  python3 "${LAUNCH_DIR}/scripts/prepare_instruction_benchmarks.py" "${BENCHMARK_PREP_ARGS[@]}"
fi

if [[ -n "${STUDENT_MODEL_PATH:-}" && -n "${TEACHER_MODEL_PATH:-}" ]]; then
  echo "[models] resolving student=${STUDENT_MODEL_PATH} teacher=${TEACHER_MODEL_PATH}"
  python3 "${LAUNCH_DIR}/scripts/resolve_models.py" \
    --student "${STUDENT_MODEL_PATH}" \
    --teacher "${TEACHER_MODEL_PATH}" \
    --output-env "${MODEL_ENV_FILE}"
elif [[ ! -e "${MODEL_ENV_FILE}" || "${CC_OPD_RESOLVE_MODELS:-true}" == "true" ]]; then
  echo "[models] resolving via OpenLM/HF: student=${STUDENT_MODEL_ID} teacher=${TEACHER_MODEL_ID}"
  python3 "${LAUNCH_DIR}/scripts/resolve_models.py" \
    --student "${STUDENT_MODEL_ID}" \
    --teacher "${TEACHER_MODEL_ID}" \
    --output-env "${MODEL_ENV_FILE}"
fi
MODEL_JSON_FILE="${MODEL_ENV_FILE}.json"
readarray -t MODEL_PATH_LINES < <(python3 - "${MODEL_ENV_FILE}" "${MODEL_JSON_FILE}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
allowed = {"STUDENT_MODEL_PATH", "TEACHER_MODEL_PATH"}

def normalize_model_path(value: str) -> str:
    normalized = str(value).strip()
    if normalized != "/":
        normalized = normalized.rstrip("/")
    if not normalized:
        raise SystemExit("empty model path")
    return normalized

if json_path.exists():
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    values = {
        "STUDENT_MODEL_PATH": normalize_model_path(payload["student"]),
        "TEACHER_MODEL_PATH": normalize_model_path(payload["teacher"]),
    }
else:
    values = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = shlex.split(line, comments=False, posix=True)
        if len(parts) != 2 or parts[0] != "export" or "=" not in parts[1]:
            raise SystemExit(f"unsafe model env line: {line!r}")
        key, value = parts[1].split("=", 1)
        if key not in allowed:
            raise SystemExit(f"unexpected model env key: {key}")
        values[key] = normalize_model_path(value)

missing = sorted(allowed - values.keys())
if missing:
    raise SystemExit(f"missing model env keys: {missing}")
print(values["STUDENT_MODEL_PATH"])
print(values["TEACHER_MODEL_PATH"])
PY
)
STUDENT_MODEL="${MODEL_PATH_LINES[0]}"
MATH_TEACHER="${MODEL_PATH_LINES[1]}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL}"
export TEACHER_MODEL_PATH="${MATH_TEACHER}"

for required in "${CODE_DIR}" "${STUDENT_MODEL}" "${MATH_TEACHER}" "${TRAIN_DATA}" "${VAL_DATA}" "${EVAL_BENCHMARKS_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[FATAL] missing required path: ${required}" >&2
    exit 1
  fi
done

if [[ "${CC_OPD_BENCHMARK_EVAL_ENABLED}" == "true" ]]; then
  for required in "${BENCHMARK_ROOT}" "${LAUNCH_DIR}/scripts/run_instruction_benchmark_eval.py"; do
    if [[ ! -e "${required}" ]]; then
      echo "[FATAL] benchmark eval enabled but missing required path: ${required}" >&2
      exit 1
    fi
  done
fi

cd "${CODE_DIR}"

LOG_FILE="${LOG_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=========================================="
    echo "JOB_NAME: $JOB_NAME"
    echo "METHOD: ${OPD_METHOD}"
    echo "OPD_METHOD: $OPD_METHOD"
    echo "ENV: $CC_OPD_ENV_NAME"
    echo "CODE_DIR: $CODE_DIR"
    echo "TRAIN: $TRAIN_DATA"
    echo "VAL: $VAL_DATA"
    echo "BENCHMARK_ROOT: $BENCHMARK_ROOT"
    echo "BENCHMARK_EVAL_ENABLED: $CC_OPD_BENCHMARK_EVAL_ENABLED"
    echo "VAL_GENERATION_DIR: $VAL_GENERATION_DIR"
    echo "VAL_TEMPERATURE: $VAL_TEMPERATURE"
    echo "VAL_MAX_RESPONSE_LENGTH: $VAL_MAX_RESPONSE_LENGTH"
    echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
    echo "ENV_ROLLOUT_N: $ENV_ROLLOUT_N"
    echo "PPO_MINI_BATCH_SIZE: $PPO_MINI_BATCH_SIZE"
    echo "PPO_MICRO_BATCH_SIZE_PER_GPU: $PPO_MICRO_BATCH_SIZE_PER_GPU"
    echo "ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU: $ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    echo "REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU: $REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    echo "ACTOR_PARAM_OFFLOAD: $ACTOR_PARAM_OFFLOAD"
    echo "ACTOR_OPTIMIZER_OFFLOAD: $ACTOR_OPTIMIZER_OFFLOAD"
    echo "REF_PARAM_OFFLOAD: $REF_PARAM_OFFLOAD"
    echo "TOTAL_TRAINING_STEPS: $TOTAL_TRAINING_STEPS"
    echo "STUDENT: $STUDENT_MODEL"
    echo "TEACHER: $MATH_TEACHER"
    echo "CKPT: $CKPTS_DIR"
    echo "WORLD_SIZE: $WORLD_SIZE (${GPUS_PER_NODE:-8} GPU/node = $((WORLD_SIZE * ${GPUS_PER_NODE:-8})) GPU total)"
    echo "Start: $(date)"
    echo "=========================================="
} | tee "$LOG_FILE"

set +e
python3 -m verl.trainer.main_ppo_multitask \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  algorithm.adv_estimator=opd \
  actor_rollout_ref.actor.kl_loss_type=k1 \
  +actor_rollout_ref.actor.opd_mask_special_tokens=False \
  algorithm.opd.cc.enabled="${CC_ENABLED}" \
  algorithm.opd.cc.delta_key="${CC_DELTA_KEY:-cc_delta_log_probs}" \
  algorithm.opd.cc.lambda="${CC_LAMBDA:-1.0}" \
  algorithm.opd.cc.delta_clip="${CC_DELTA_CLIP:-5.0}" \
  algorithm.opd.cc.online.enabled="${CC_ONLINE_ENABLED}" \
  algorithm.opd.cc.online.rubric_sample_ratio="${CC_RUBRIC_SAMPLE_RATIO:-1.0}" \
  algorithm.opd.cc.online.seed="${CC_RUBRIC_SAMPLE_SEED:-21}" \
  algorithm.opd.cc.online.max_jobs_per_batch="${CC_MAX_ONLINE_JOBS:-256}" \
  algorithm.opd.cc.online.micro_batch_size="${CC_ONLINE_MICRO_BATCH_SIZE:-64}" \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.ref.model.path="${MATH_TEACHER}" \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length=2048 \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
  data.filter_overlong_prompts=True \
  data.truncation=middle \
  data.return_raw_chat=True \
  +data.batching_mode=sequential \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-6144}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-18432}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
  algorithm.use_kl_in_reward=False \
  env.env_name="${CC_OPD_ENV_NAME}" \
  env.seed=21 \
  env.max_steps=30 \
  env.rollout.n="${ENV_ROLLOUT_N}" \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','tensorboard'] \
  trainer.project_name=cc_opd_rubric \
  trainer.experiment_name="${JOB_NAME}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE:-8}" \
  trainer.nnodes="${WORLD_SIZE}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}" \
  trainer.val_only=False \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.val_generation_dir="${VAL_GENERATION_DIR}" \
  trainer.val_max_response_length="${VAL_MAX_RESPONSE_LENGTH}" \
  trainer.benchmark_eval.enabled="${CC_OPD_BENCHMARK_EVAL_ENABLED}" \
  trainer.benchmark_eval.script_path="${LAUNCH_DIR}/scripts/run_instruction_benchmark_eval.py" \
  trainer.benchmark_eval.benchmark_root="${BENCHMARK_ROOT}" \
  trainer.benchmark_eval.output_dir="${BENCHMARK_EVAL_OUTPUT_DIR}" \
  trainer.benchmark_eval.run_evaluators="${BENCHMARK_EVAL_RUN_EVALUATORS}" \
  trainer.benchmark_eval.fail_on_error="${BENCHMARK_EVAL_FAIL_ON_ERROR}" \
  trainer.benchmark_eval.strip_thinking="${BENCHMARK_EVAL_STRIP_THINKING}" \
  trainer.resume_mode="${TRAINER_RESUME_MODE:-auto}" \
  trainer.resume_from_path="${TRAINER_RESUME_FROM_PATH:-null}" \
  +trainer.visualize_distribution=false \
  +trainer.visualize_distribution_freq=1 \
  +trainer.visualize_distribution_samples=2 \
  +trainer.visualize_distribution_dir="${CKPTS_DIR}/visualizations" \
  +trainer.visualize_distribution_ref_tokens=3 \
  2>&1 | tee -a "${LOG_FILE}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

echo "==========================================" | tee -a "${LOG_FILE}"
echo "End time: $(date)" | tee -a "${LOG_FILE}"
echo "Training exit code: $TRAIN_RC" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

exit "$TRAIN_RC"
