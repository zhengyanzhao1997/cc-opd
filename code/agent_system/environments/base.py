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

"""Small interface shared by the instruction-following environment manager."""

import numpy as np
import torch


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (int, float, bool, list, tuple)):
        return np.asarray(data)
    raise TypeError(f"Unsupported type: {type(data)!r}")


class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

    def reset(self, kwargs):
        raise NotImplementedError

    def step(self, text_actions):
        raise NotImplementedError

    def success_evaluator(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        self.envs.close()
