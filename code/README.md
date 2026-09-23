# CC-OPD Method Implementation

This directory contains a verl-based training framework with two supported methods: leave-one-out CC-OPD (`OPD_METHOD=loo`) and vanilla sampled-token OPD (`OPD_METHOD=opd`). Both use the same student rollout, frozen teacher, training data, and advantage pipeline. LOO adds a constraint-conditioned teacher signal; OPD disables that signal and the extra teacher scoring jobs.

## Method files

| Path | Role |
|---|---|
| `verl/trainer/ppo/cc_opd.py` | Builds leave-one-out counterfactual prompts, batches teacher scoring, and aggregates token-level deltas. |
| `verl/trainer/ppo/core_algos.py` | Computes sampled-token OPD advantage, optionally adding the clipped CC delta. |
| `verl/trainer/ppo/ray_trainer_multitask.py` | Runs the student rollout, teacher scoring, and advantage computation. |
| `verl/trainer/main_ppo_multitask.py` | Training entry point. |
| `verl/trainer/config/ppo_trainer.yaml` | FSDP training configuration for both methods. |
| `verl/workers/multitask_ref_worker.py` | Frozen-teacher log-probability worker. |
| `verl/utils/dataset/multitask_rl_dataset.py` | Task-aware instruction-following dataset. |
| `agent_system/environments/env_package/instruction_following/` | Instruction-following environment. |
| `agent_system/multi_turn_rollout/rollout_loop.py` | Preserves the prompt and constraint metadata needed for LOO scoring. |

Use `../training/run_cc_opd.sh` to select a method. The launcher passes `algorithm.adv_estimator=opd` in both cases. For `OPD_METHOD=loo`, it sets `algorithm.opd.cc.enabled=true` and `algorithm.opd.cc.online.enabled=true`; for `OPD_METHOD=opd`, both flags are false.

The LOO configuration in `verl/trainer/config/ppo_trainer.yaml` includes the delta scale (`lambda`), symmetric clip (`delta_clip`), rubric sampling ratio, and counterfactual teacher-scoring batch sizes. The method constructs leave-one-out prompts and sums the per-constraint deltas before clipping.

The config retains inactive critic and reward-model fields for compatibility with the inherited trainer; neither method trains or loads those models.

The underlying verl code is distributed under Apache 2.0. See `LICENSE` and `Notice.txt` for its license and attribution.
