#!/usr/bin/env bash
# Environment template for CC-OPD training.
#
#   cp env.example.sh env.sh
#   $EDITOR env.sh        # fill in paths and choose algorithm
#   source env.sh
#   bash run_cc_opd.sh

# === REQUIRED: model and data paths ===
export STUDENT_MODEL_PATH="<path-to-student-model-or-hf-repo-id>"
export TEACHER_MODEL_PATH="<path-to-teacher-model-or-hf-repo-id>"
export TRAIN_DATA="<path-to-train.parquet>"
export VAL_DATA="<path-to-val.parquet>"

# === Method choice: loo (CC-OPD) or opd (vanilla sampled-token OPD) ===
export OPD_METHOD="loo"
export RHO="1.0"                  # rubric sample ratio in (0, 1]
export LAMBDA="2.0"               # scaling on the CC delta
export DELTA_CLIP="5.0"           # per-token clip on the CC delta

# === Training schedule ===
# Defaults: loo=1500 steps/30 epochs; opd=6000 steps/60 epochs.
# Override only if you intend to change the training horizon.
# export TOTAL_TRAINING_STEPS="1500"
# export TOTAL_EPOCHS="30"
export SAVE_FREQ="20"
export TEST_FREQ="20"

# === Batch and rollout ===
export TRAIN_BATCH_SIZE="128"
export PPO_MINI_BATCH_SIZE="128"
export PPO_MICRO_BATCH_SIZE_PER_GPU="4"
export ENV_ROLLOUT_N="8"
export MAX_RESPONSE_LENGTH="4096"

# === Cluster topology ===
export WORLD_SIZE="1"             # number of nodes
export GPUS_PER_NODE="8"
# Multi-node only:
# export MASTER_ADDR="<head-node-hostname-or-ip>"
# export RANK="0"                 # 0 .. WORLD_SIZE-1, set per node

# === Output paths ===
export CC_OPD_DATA_BASE="${HOME}/cc_opd/data"
export CC_OPD_EXP_BASE="${HOME}/cc_opd/experiments"

# === Optional in-training benchmark eval (off by default) ===
# When enabled, the trainer runs rule-based IFEval / IFBench / MulDimIF
# scoring on saved validation generations after each test_freq interval.
# No external LLM judge is involved.
# export CC_OPD_BENCHMARK_EVAL_ENABLED="true"
# export BENCHMARK_ROOT="<path-to-benchmark-root>"
