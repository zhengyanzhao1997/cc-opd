# Training CC-OPD and OPD

The launcher runs either the paper's leave-one-out CC-OPD method or the vanilla sampled-token OPD baseline. Both use the same student, frozen teacher, training data, and validation pipeline.

| `OPD_METHOD` | Teacher scoring | Default training horizon |
|---|---|---|
| `loo` (default) | Full instruction plus one prompt with each sampled constraint removed. | 1,500 steps |
| `opd` | Full instruction only; no counterfactual teacher jobs. | 6,000-step maximum horizon; the paper reports the selected 3,000-step checkpoint. |

The LOO method uses inline removal of each constraint, with `RHO=1.0`, `LAMBDA=2.0`, and `DELTA_CLIP=5.0` by default. These settings have no effect when `OPD_METHOD=opd`. Other counterfactual variants are not part of this release.

Both methods retain the experimental actor update implementation, including its dual-clip PPO setting `clip_ratio_c=3.0`; the release does not change that optimizer behavior.

## Installation

Install in a Python 3.10 environment on a CUDA server with a compatible PyTorch build:

```bash
pip install --no-deps --no-build-isolation -e ../code
pip install vllm==0.8.5
pip install flash-attn==2.7.0.post2 --no-build-isolation --no-cache-dir
pip install transformers==4.51.3 tokenizers==0.21.1 --no-deps --force-reinstall
pip install -r requirements.txt
```

The 7B teacher used in the paper generally needs 80 GB class GPUs. The launcher uses Ray for single- and multi-node training. Dependencies and model weights are not bundled.

## Prepare data

Training requires a parquet file containing prompts and per-example constraints (`TRAIN_DATA`), and a validation parquet file (`VAL_DATA`). The same files are used for both methods. To match the paper's training prompts, start from the teacher-aligned HIR JSONL, whose `messages[0].content` contains the inline constraints used to train the teacher:

```bash
python3 scripts/prepare_hir16k_teacher_aligned.py \
  --input-jsonl /path/to/HIR_trainv1_rubrics_processed.jsonl \
  --output-dir /path/to/data
python3 scripts/prepare_instruction_benchmarks.py \
  --benchmark-root /path/to/benchmark-sources \
  --output-dir /path/to/data
```

Set `TRAIN_DATA=/path/to/data/hir16k_train_teacher_aligned.parquet`. The teacher-aligned JSONL and benchmark source datasets are not bundled. `scripts/prepare_hir16k.py` can instead download public `sastpg/HIR-16K` and build a schema-compatible parquet for vanilla OPD, but its bullet-block prompts do not reproduce the paper's inline LOO counterfactuals.

## Configure and run

```bash
cp env.example.sh env.sh
# Edit model and data paths in env.sh, then:
source env.sh
bash run_cc_opd.sh                    # LOO CC-OPD
OPD_METHOD=opd bash run_cc_opd.sh     # vanilla OPD
```

`STUDENT_MODEL_PATH` and `TEACHER_MODEL_PATH` can be local model directories or Hugging Face repo IDs. `TRAIN_DATA` and `VAL_DATA` must point to existing parquet files. Checkpoints and logs are written under `CC_OPD_EXP_BASE`. The teacher follows an external fine-tuning recipe; this repository starts from a prepared teacher checkpoint.

For a single-node 8-GPU run, leave `WORLD_SIZE=1` and `GPUS_PER_NODE=8`. For multiple nodes, set the same `WORLD_SIZE` and `MASTER_ADDR` on all nodes, and set `RANK=0,1,...` per node. Rank 0 starts the Ray head and launches the trainer.

To preview method selection without starting Ray or loading models, set `CC_OPD_PRINT_CONFIG_ONLY=true`:

```bash
CC_OPD_PRINT_CONFIG_ONLY=true OPD_METHOD=loo bash run_cc_opd.sh
CC_OPD_PRINT_CONFIG_ONLY=true OPD_METHOD=opd bash run_cc_opd.sh
```

To inspect the exact method flags passed into the trainer, run `JOB_NAME=check CC_OPD_PRINT_CONFIG_ONLY=true OPD_METHOD=opd bash scripts/train.sh` (or use `loo`).

## CPU checks

After installing the Python dependencies, run:

```bash
python3 scripts/smoke_check_cc_opd.py
python3 scripts/smoke_check_instruction_benchmarks.py
```

The first check covers sampled-token OPD advantage, LOO prompt construction, counterfactual scoring batches, delta aggregation, and the CC advantage. The second checks instruction benchmark adapters with synthetic inputs. Neither runs a GPU training job.

## Validation and checkpoints

The default validation interval is 20 steps with three sampled generations per prompt. Rule-based IFEval, IFBench, and MulDimIF evaluation can be enabled with `CC_OPD_BENCHMARK_EVAL_ENABLED=true` and `BENCHMARK_ROOT=/path/to/benchmark-sources`. It requires the original benchmark files and evaluators under that root; no external LLM judge is used.

`TOTAL_TRAINING_STEPS` and `TOTAL_EPOCHS` may override the method-specific training horizon. LOO defaults to 1,500 steps and 30 epochs; OPD defaults to a 6,000-step maximum and 60 epochs so the epoch capacity does not stop it early. `SAVE_FREQ` and `TEST_FREQ` both default to 20. Select the desired checkpoint from validation, rather than assuming the final step is best.

`model_merger.py` can merge FSDP checkpoint shards for standalone inference after training. See the script's `--help` for arguments.
