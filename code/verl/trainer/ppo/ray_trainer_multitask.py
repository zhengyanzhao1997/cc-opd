# ray trainer for multitask training
# dataload, rollout, reference model(along with metrics log) are different

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.cc_opd import (
    aggregate_online_cc_deltas,
    build_counterfactual_scoring_batch,
    build_online_cc_rubric_jobs,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


def _get_cc_delta_key(config) -> str:
    opd_cfg = config.algorithm.get("opd", {})
    cc_cfg = opd_cfg.get("cc", {}) if opd_cfg is not None else {}
    return str(cc_cfg.get("delta_key", "cc_delta_log_probs"))


def _route_key(task_type) -> tuple[str, str]:
    if task_type is None:
        return ("none", "")
    return (type(task_type).__name__, str(task_type))


def _chunk_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0 or chunk_size >= len(indices):
        return [indices]
    return [indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size)]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _as_list(value, length: int, default):
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif torch.is_tensor(value):
        items = value.detach().cpu().tolist()
    elif isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    else:
        items = [value]
    if len(items) == length:
        return items
    if len(items) == 1 and length != 1:
        return items * length
    return (items + [default for _ in range(length)])[:length]


def _metadata_items(non_tensor_batch: dict[str, object], input_texts: list[str]) -> list[dict[str, object]]:
    length = len(input_texts)
    extra_infos = _as_list(non_tensor_batch.get("extra_info"), length, {})
    data_sources = _as_list(non_tensor_batch.get("data_source"), length, "unknown")
    task_types = _as_list(non_tensor_batch.get("task_type"), length, "unknown")
    metadata: list[dict[str, object]] = []
    for index, input_text in enumerate(input_texts):
        raw_extra = _json_safe(extra_infos[index])
        extra_info = raw_extra if isinstance(raw_extra, dict) else {"value": raw_extra}
        original_prompt = extra_info.get("original_prompt") or extra_info.get("base_prompt") or input_text
        metadata.append(
            {
                "extra_info": extra_info,
                "original_prompt": original_prompt,
                "data_source": _json_safe(data_sources[index]),
                "task_type": _json_safe(task_types[index]),
            }
        )
    return metadata


def _compute_entropy_top_mask(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    top_ratio: float,
) -> torch.Tensor:
    """Keep only the top-`top_ratio` highest-entropy tokens per sample.

    Args:
        entropys: (batch_size, response_length) per-token entropy.
        response_mask: (batch_size, response_length) binary mask for valid tokens.
        top_ratio: fraction in (0, 1] of valid tokens to keep per sample.

    Returns:
        entropy_mask: (batch_size, response_length) binary mask, a subset of response_mask.
    """
    masked_entropy = entropys * response_mask + (-1e9) * (1 - response_mask)

    num_valid = response_mask.sum(dim=-1)  # (batch_size,)
    num_keep = torch.clamp((num_valid * top_ratio).long(), min=1)

    sorted_entropy, _ = masked_entropy.sort(dim=-1, descending=True)
    # threshold: the entropy value at the num_keep-th position per sample
    thresholds = sorted_entropy.gather(1, (num_keep - 1).unsqueeze(-1)).squeeze(-1)  # (batch_size,)

    entropy_mask = ((entropys >= thresholds.unsqueeze(-1)) & response_mask.bool()).float()
    return entropy_mask


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """Supported advantage estimator for both release methods."""

    OPD = "opd"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")



def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    *,
    cc_enabled: bool = False,
    cc_delta_key: str = "cc_delta_log_probs",
    cc_lambda: float = 1.0,
    cc_delta_clip: float = 0.0,
    metrics_out: dict | None = None,
) -> DataProto:
    """Compute the sampled-token OPD advantage, optionally with LOO shaping."""
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    advantages, returns = core_algos.compute_opd_advantage(
        data=data,
        cc_enabled=cc_enabled,
        cc_delta_key=cc_delta_key,
        cc_lambda=cc_lambda,
        cc_delta_clip=cc_delta_clip,
        metrics_out=metrics_out,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        if self.config.algorithm.adv_estimator != AdvantageEstimator.OPD:
            raise ValueError("Only LOO CC-OPD and vanilla sampled-token OPD are supported")
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            raise ValueError("The supported OPD methods use the sampled-token reward without an additional KL loss")
        if not self.use_reference_policy:
            raise ValueError("A frozen teacher reference worker is required for OPD")
        self.use_critic = False

        if OmegaConf.select(self.config, 'multitask'):
            with open_dict(self.config.actor_rollout_ref):
                self.config.actor_rollout_ref.multitask = self.config.multitask 

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        # V15 PATCH: env-multitask trainer uses env.rollout.n (NOT actor_rollout_ref.rollout.n which is forced to 1).
        # Original line was wrong for multitask trainer; without this patch 32 GPU fails (16*1=16 not divisible by 32).
        try:
            env_rollout_n = config.env.rollout.n
        except Exception:
            env_rollout_n = 1
        rollout_n_for_check = max(env_rollout_n, config.actor_rollout_ref.rollout.n)
        real_train_batch_size = config.data.train_batch_size * rollout_n_for_check
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus}). env.rollout.n={env_rollout_n}, actor_rollout_ref.rollout.n={config.actor_rollout_ref.rollout.n}"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # Tool-driven multi-turn rollouts are outside this release.
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            raise ValueError("Tool-driven multi-turn rollout is not supported by these OPD recipes")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.dataset_helpers import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        # check if we need a multi-task sampler
        batching_mode = self.config.data.get("batching_mode", None)
        if batching_mode == "sequential":
            from verl.utils.dataset.multitask_rl_dataset import (
                MultiTaskRLHFDataset,
                SequentialTaskSampler,
            )

            if isinstance(self.val_dataset, MultiTaskRLHFDataset):
                val_sampler = SequentialTaskSampler(
                    dataset=self.val_dataset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    drop_last=False,
                    seed=self.config.data.get("seed", 42),
                )
            else:
                val_sampler = None
        else:
            val_sampler = None

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=val_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w", encoding="utf-8") as f:
            for i in range(n):
                entry = {k: _json_safe(v[i]) for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")
        return filename

    def _run_instruction_benchmark_eval(self, generation_jsonl: str) -> dict[str, float]:
        benchmark_cfg = self.config.trainer.get("benchmark_eval", {})
        if not benchmark_cfg or not benchmark_cfg.get("enabled", False):
            return {}

        script_path = benchmark_cfg.get("script_path", None)
        if not script_path:
            print("[benchmark_eval] enabled but trainer.benchmark_eval.script_path is empty; skipping")
            return {"val/benchmark_eval/skipped": 1.0}
        output_dir = benchmark_cfg.get("output_dir", None)
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(generation_jsonl), "benchmark_eval", str(self.global_steps))
        metrics_json = os.path.join(output_dir, "benchmark_metrics.json")
        command = [
            sys.executable,
            str(script_path),
            "--generation-jsonl",
            str(generation_jsonl),
            "--output-dir",
            str(output_dir),
            "--metrics-json",
            str(metrics_json),
        ]
        benchmark_root = benchmark_cfg.get("benchmark_root", None)
        if benchmark_root:
            command.extend(["--benchmark-root", str(benchmark_root)])
        if benchmark_cfg.get("run_evaluators", False):
            command.append("--run-evaluators")
        if benchmark_cfg.get("strip_thinking", True):
            command.append("--strip-thinking")
        else:
            command.append("--no-strip-thinking")

        print(f"[benchmark_eval] running: {' '.join(command)}")
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            message = f"benchmark evaluator failed with code {result.returncode}"
            if benchmark_cfg.get("fail_on_error", True):
                raise RuntimeError(message)
            print(f"[benchmark_eval] {message}; continuing because fail_on_error=false")
            return {"val/benchmark_eval/failed": 1.0}

        try:
            with open(metrics_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            if benchmark_cfg.get("fail_on_error", True):
                raise
            return {"val/benchmark_eval/missing_metrics": 1.0}
        raw_metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        metrics: dict[str, float] = {}
        for key, value in raw_metrics.items():
            if isinstance(value, (int, float)):
                metric_key = key if key.startswith("val/") else f"val/{key}"
                metrics[metric_key] = float(value)
        return metrics

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_reward_extra_infos: dict[str, list[object]] = {}
        sample_data_sources = []
        sample_extra_infos = []
        sample_original_prompts = []
        sample_task_types = []
        dumped_val_file = None

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            validation_metadata = _metadata_items(test_batch.non_tensor_batch, input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            if "task_type" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("task_type")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            val_max_response_length = self.config.trainer.get("val_max_response_length", None)
            if val_max_response_length is not None:
                test_gen_batch.meta_info["max_tokens"] = int(val_max_response_length)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            if len(validation_metadata) != len(output_texts):
                print(
                    f"[validation_metadata] metadata/output length mismatch: "
                    f"{len(validation_metadata)} vs {len(output_texts)}; using decoded inputs as fallback"
                )
                validation_metadata = _metadata_items(test_output_gen_batch.non_tensor_batch, input_texts)
            sample_extra_infos.extend([item["extra_info"] for item in validation_metadata])
            sample_original_prompts.extend([item["original_prompt"] for item in validation_metadata])
            sample_task_types.extend([item["task_type"] for item in validation_metadata])

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            batch_extra = result.get("reward_extra_info", {})
            for k, v in batch_extra.items():
                sample_reward_extra_infos.setdefault(k, []).extend(v if isinstance(v, list) else list(v))
            batch_ds = test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
            sample_data_sources.extend(batch_ds if isinstance(batch_ds, list) else list(batch_ds))

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(batch_ds)
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_dump_dir = self.config.trainer.get("val_generation_dir", None)
        if not val_dump_dir:
            val_dump_dir = self.config.trainer.get("validation_data_dir", None)
        extra_with_ds = {
            **sample_reward_extra_infos,
            "data_source": sample_data_sources,
            "extra_info": sample_extra_infos,
            "original_prompt": sample_original_prompts,
            "task_type": sample_task_types,
        }
        if val_dump_dir:
            dumped_val_file = self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=extra_with_ds,
                dump_path=val_dump_dir,
            )

        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,) # will lead to a bug if repo_len is not equal in the batch
        reward_tensor = torch.cat([r.sum(-1) for r in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        if dumped_val_file:
            metric_dict.update(self._run_instruction_benchmark_eval(dumped_val_file))

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _compute_online_cc_delta(self, batch: DataProto, cc_cfg) -> tuple[torch.Tensor, dict[str, float]]:
        start_time = time.perf_counter()
        if "ref_log_prob" not in batch.batch:
            raise KeyError("online CC-OPD requires full-rubric ref_log_prob before counterfactual scoring")
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        online_cfg = cc_cfg.get("online", {})
        rubric_sample_ratio = float(online_cfg.get("rubric_sample_ratio", 1.0))
        seed = int(online_cfg.get("seed", self.config.env.get("seed", 0)))
        max_jobs_per_batch = int(online_cfg.get("max_jobs_per_batch", 0) or 0)
        micro_batch_size = int(online_cfg.get("micro_batch_size", 0) or 0)
        jobs, build_stats = build_online_cc_rubric_jobs(
            non_tensor_batch=batch.non_tensor_batch,
            rubric_sample_ratio=rubric_sample_ratio,
            seed=seed,
            step=int(self.global_steps),
        )
        if max_jobs_per_batch > 0 and len(jobs) > max_jobs_per_batch:
            raise ValueError(
                f"online CC-OPD generated {len(jobs)} counterfactual jobs, exceeding "
                f"algorithm.opd.cc.online.max_jobs_per_batch={max_jobs_per_batch}. "
                "Lower rubric_sample_ratio or raise the cap deliberately."
            )

        full_log_probs = batch.batch["ref_log_prob"].detach().cpu()
        response_mask = batch.batch["response_mask"].detach().cpu()
        metrics = {
            "cc_opd/online_jobs": float(len(jobs)),
            "cc_opd/online_rubric_sample_ratio": rubric_sample_ratio,
            "cc_opd/online_samples_with_rubrics": float(len({job.sample_index for job in jobs})),
            "cc_opd/online_max_jobs_per_batch": float(max_jobs_per_batch),
            "cc_opd/online_micro_batch_size": float(micro_batch_size),
        }
        metrics.update(build_stats)
        if not jobs:
            metrics["cc_opd/online_delta_abs_mean"] = 0.0
            metrics["cc_opd/online_latency_sec"] = time.perf_counter() - start_time
            return torch.zeros_like(full_log_probs), metrics

        response_length = full_log_probs.shape[-1]
        counterfactual_log_probs = torch.empty((len(jobs), response_length), dtype=full_log_probs.dtype)
        grouped_job_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for job_index, job in enumerate(jobs):
            grouped_job_indices[_route_key(job.task_type)].append(job_index)

        apply_chat_template_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        ref_calls = 0
        counterfactual_tokens = 0.0
        for job_indices in grouped_job_indices.values():
            for chunk_indices in _chunk_indices(job_indices, micro_batch_size):
                group_jobs = [jobs[job_index] for job_index in chunk_indices]
                cf_batch = build_counterfactual_scoring_batch(
                    jobs=group_jobs,
                    responses=batch.batch["responses"].detach().cpu(),
                    response_mask=response_mask,
                    tokenizer=self.tokenizer,
                    max_prompt_length=int(self.config.data.max_prompt_length),
                    pad_token_id=int(self.tokenizer.pad_token_id),
                    truncation=str(self.config.data.truncation),
                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                )
                counterfactual_tokens += float(cf_batch.batch["attention_mask"].sum().item())
                if not self.ref_in_actor:
                    cf_ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(cf_batch)
                else:
                    cf_ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(cf_batch)
                ref_calls += 1
                cf_values = cf_ref_log_prob.batch["ref_log_prob"].detach().cpu().to(dtype=full_log_probs.dtype)
                if tuple(cf_values.shape) != (len(group_jobs), response_length):
                    raise ValueError(
                        f"counterfactual ref_log_prob shape {tuple(cf_values.shape)} does not match "
                        f"expected {(len(group_jobs), response_length)}"
                    )
                for local_index, job_index in enumerate(chunk_indices):
                    counterfactual_log_probs[job_index] = cf_values[local_index]

        cc_delta, delta_stats = aggregate_online_cc_deltas(
            full_log_probs=full_log_probs,
            counterfactual_log_probs=counterfactual_log_probs,
            jobs=jobs,
            response_mask=response_mask,
        )
        metrics.update(delta_stats)
        metrics.update(
            {
                "cc_opd/online_delta_abs_mean": cc_delta.abs().mean().item(),
                "cc_opd/online_delta_abs_max": cc_delta.abs().max().item(),
                "cc_opd/online_rubrics_total_mean": float(np.mean([job.total_rubrics for job in jobs])),
                "cc_opd/online_rubrics_selected_mean": float(np.mean([job.selected_count for job in jobs])),
                "cc_opd/online_scale_mean": float(np.mean([job.scale for job in jobs])),
                "cc_opd/online_ref_calls": float(ref_calls),
                "cc_opd/online_counterfactual_tokens": counterfactual_tokens,
                "cc_opd/online_latency_sec": time.perf_counter() - start_time,
            }
        )
        return cc_delta.to(device=batch.batch["responses"].device, dtype=batch.batch["ref_log_prob"].dtype), metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                cc_delta_key_for_gen = _get_cc_delta_key(self.config)
                if cc_delta_key_for_gen in batch.batch:
                    batch_keys_to_pop.append(cc_delta_key_for_gen)
                # print(f"[DBG] {batch.non_tensor_batch.keys()=}") # [DBG] batch.non_tensor_batch.keys()=dict_keys(['task_type', 'data_source', 'env_kwargs', 'extra_info', 'ability', 'reward_model', 'raw_prompt_ids', 'raw_prompt', 'index', 'tools_kwargs'])
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                if "task_type" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("task_type")
                if "extra_info" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("extra_info")
                if cc_delta_key_for_gen in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(cc_delta_key_for_gen)
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Reordering preserves per-trajectory tensors and metadata.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Recompute sampled-token log-probabilities for the rollout.
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                        # Visualize teacher vs student distributions
                        if self.config.trainer.get("visualize_distribution", False):
                            visualize_freq = self.config.trainer.get("visualize_distribution_freq", 100)
                            if self.global_steps % visualize_freq == 0:
                                try:
                                    from verl.utils.visualize_distribution import visualize_teacher_student_batch
                                    
                                    # Extract task_type for filename
                                    task_type = None
                                    if 'task_type' in batch.non_tensor_batch:
                                        task_type = str(batch.non_tensor_batch['task_type'][0])
                                    
                                    output_dir = self.config.trainer.get(
                                        "visualize_distribution_dir",
                                        f"{self.config.trainer.default_hdfs_dir}/visualizations"
                                    )
                                    
                                    visualize_teacher_student_batch(
                                        batch=batch,
                                        teacher_log_probs=batch.batch['ref_log_prob'],
                                        student_log_probs=batch.batch['old_log_probs'],
                                        tokenizer=self.tokenizer,
                                        global_step=self.global_steps,
                                        output_dir=output_dir,
                                        num_samples=self.config.trainer.get("visualize_distribution_samples", 2),
                                        task_type=task_type,
                                        num_tokens=self.config.trainer.get("visualize_distribution_ref_tokens", 1),
                                    )
                                except Exception as e:
                                    print(f"[Warning] Failed to generate distribution visualization: {e}")
                                    import traceback
                                    traceback.print_exc()
                        if self.config.trainer.get("visualize_tea_stu_diff", False):
                            try:
                                from verl.utils.visualize_distribution import visualize_teacher_student_diff
                                output_dir = self.config.trainer.get(
                                    "visualize_distribution_dir",
                                    f"{self.config.trainer.default_hdfs_dir}/visualizations"
                                )
                                visualize_teacher_student_diff(
                                    batch=batch,
                                    teacher_log_probs=batch.batch['ref_log_prob'],
                                    student_log_probs=batch.batch['old_log_probs'],
                                    global_step=self.global_steps,
                                    output_dir=output_dir,
                                )
                            except Exception as e:
                                print(f"[Warning] Failed to generate teacher-student difference visualization: {e}")
                                import traceback
                                traceback.print_exc()

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', False):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.adv_estimator != AdvantageEstimator.OPD:
                            raise ValueError("This release supports sampled-token OPD and LOO CC-OPD only")

                        opd_cfg = self.config.algorithm.get("opd", {})
                        cc_cfg = opd_cfg.get("cc", {})
                        cc_enabled = bool(cc_cfg.get("enabled", False))
                        cc_online_enabled = bool(cc_cfg.get("online", {}).get("enabled", False))
                        if cc_enabled != cc_online_enabled:
                            raise ValueError("LOO requires both algorithm.opd.cc.enabled and online.enabled")

                        cc_delta_key = str(cc_cfg.get("delta_key", "cc_delta_log_probs"))
                        if cc_enabled:
                            cc_delta, cc_metrics = self._compute_online_cc_delta(batch, cc_cfg)
                            batch.batch[cc_delta_key] = cc_delta
                            metrics.update(cc_metrics)

                        batch = compute_advantage(
                            batch,
                            cc_enabled=cc_enabled,
                            cc_delta_key=cc_delta_key,
                            cc_lambda=float(cc_cfg.get("lambda", 1.0)),
                            cc_delta_clip=float(cc_cfg.get("delta_clip", 0.0)),
                            metrics_out=metrics,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict[str, object] = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
