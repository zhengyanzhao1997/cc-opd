# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

"""Environment manager for the one-step instruction-following task."""

from typing import Any, Dict, List, Tuple

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.env_package.instruction_following import (
    build_instruction_following_envs,
    instruction_projection,
)


class InstructionFollowingEnvironmentManager(EnvironmentManagerBase):
    """Wrap prompt/response samples for OPD and leave-one-out CC-OPD."""

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        return {"text": obs, "image": None, "anchor": obs.copy()}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        observations = {
            "text": next_obs,
            "image": None,
            "anchor": next_obs.copy() if isinstance(next_obs, list) else None,
        }
        for info, valid in zip(infos, valids):
            info["is_action_valid"] = to_numpy(valid)
        return observations, to_numpy(rewards), to_numpy(dones), infos

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Benchmark scores are computed by the external evaluators."""
        return {}


def make_envs(config):
    """Construct the training and validation instruction-following environments."""
    env_name = str(config.env.env_name).lower()
    if "hir" not in env_name and "instruction" not in env_name:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")
    if hasattr(config, "multitask") and config.multitask and config.multitask.get("enable", False):
        raise ValueError("This release supports instruction-following training only")

    rollout_n = config.env.rollout.n
    if not isinstance(rollout_n, int):
        raise ValueError("config.env.rollout.n must be an integer")
    group_n = rollout_n if rollout_n > 0 else 1
    try:
        val_n = max(1, int(config.actor_rollout_ref.rollout.val_kwargs.n))
    except (AttributeError, KeyError, TypeError, ValueError):
        val_n = 1

    train_env = build_instruction_following_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
    )
    val_env = build_instruction_following_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=val_n,
        is_train=False,
    )
    return (
        InstructionFollowingEnvironmentManager(train_env, instruction_projection, config),
        InstructionFollowingEnvironmentManager(val_env, instruction_projection, config),
    )
