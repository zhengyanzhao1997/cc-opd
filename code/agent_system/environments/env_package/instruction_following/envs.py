from __future__ import annotations

from typing import Any, Dict, List


class InstructionFollowingEnv:
    """One-step prompt environment for HIR/instruction-following OPD.

    HIR-16K verifier metadata can contain executable checker snippets. This
    environment deliberately treats that metadata as opaque data and never
    executes it. OPD/CC-OPD uses teacher log-probabilities as the training
    signal; episode reward is a neutral zero unless an external evaluator is
    added later.
    """

    def __init__(self) -> None:
        self.question = ""
        self.data_source = "unknown"
        self.ground_truth = ""

    def reset(self, extras: Dict[str, Any]) -> str:
        question = extras.get("question") or extras.get("prompt") or extras.get("instruction") or ""
        self.question = str(question)
        self.data_source = str(extras.get("data_source", "unknown"))
        self.ground_truth = str(extras.get("ground_truth", ""))
        return self.question

    def step(self, action: str):
        del action
        done = True
        info = {
            "data_source": self.data_source,
            "won": False,
            "has_toolcall": False,
        }
        return None, 0.0, done, info

    def close(self) -> None:
        pass


class InstructionFollowingMultiProcessEnv:
    """Simple vector wrapper for parallel prompt/response episodes."""

    def __init__(self, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True) -> None:
        del seed, is_train
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.envs = [InstructionFollowingEnv() for _ in range(self.batch_size)]

    def reset(self, kwargs: List[Dict[str, Any]]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}")

        padded_kwargs = list(kwargs) + [{"question": "", "data_source": "unknown", "ground_truth": ""}] * (self.batch_size - len(kwargs))
        valid_mask = [True] * len(kwargs) + [False] * (self.batch_size - len(kwargs))

        results = [(env.reset(kw), {"data_source": env.data_source}) for env, kw in zip(self.envs, padded_kwargs)]
        obs_list, info_list = map(list, zip(*results))
        obs_list = [obs for obs, keep in zip(obs_list, valid_mask) if keep]
        info_list = [info for info, keep in zip(info_list, valid_mask) if keep]
        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}")

        padded_actions = list(actions) + [""] * (self.batch_size - len(actions))
        valid_mask = [True] * len(actions) + [False] * (self.batch_size - len(actions))
        results = [env.step(action) for env, action in zip(self.envs, padded_actions)]
        obs_list, reward_list, done_list, info_list = map(list, zip(*results))
        obs_list = [obs for obs, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [reward for reward, keep in zip(reward_list, valid_mask) if keep]
        done_list = [done for done, keep in zip(done_list, valid_mask) if keep]
        info_list = [info for info, keep in zip(info_list, valid_mask) if keep]
        return obs_list, reward_list, done_list, info_list

    def close(self) -> None:
        for env in self.envs:
            env.close()


def build_instruction_following_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True):
    return InstructionFollowingMultiProcessEnv(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train)
