#!/usr/bin/env python3
"""CPU smoke check for vanilla sampled-token OPD and leave-one-out CC-OPD."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(CODE_DIR))

from verl.trainer.ppo.cc_opd import (  # noqa: E402
    aggregate_online_cc_deltas,
    build_cf_instruction,
    build_counterfactual_scoring_batch,
    build_online_cc_rubric_jobs,
)
from verl.trainer.ppo.core_algos import compute_opd_advantage  # noqa: E402


class FakeBatch(dict[str, torch.Tensor]):
    batch_size = (1,)


class FakeData:
    def __init__(self, delta: torch.Tensor | None = None) -> None:
        self.batch = FakeBatch(
            responses=torch.tensor([[10, 11, 0]]),
            attention_mask=torch.tensor([[1.0, 1.0, 0.0]]),
            token_level_scores=torch.zeros(1, 3),
            old_log_probs=torch.tensor([[-2.0, -3.0, -9.0]]),
            ref_log_prob=torch.tensor([[-1.5, -2.0, -8.0]]),
        )
        if delta is not None:
            self.batch["cc_delta_log_probs"] = delta


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, chat, add_generation_prompt: bool, tokenize: bool, **kwargs) -> str:
        assert add_generation_prompt and not tokenize
        return chat[0]["content"] + "<assistant>"

    def __call__(self, text: str, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt" and not add_special_tokens
        ids = torch.tensor([[(ord(char) % 97) + 1 for char in text]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def main() -> None:
    # Vanilla OPD ignores CC deltas and scores only sampled response tokens.
    baseline, _ = compute_opd_advantage(FakeData())
    expected_baseline = torch.tensor([[0.5, 1.0, 0.0]])
    assert torch.allclose(baseline, expected_baseline), (baseline, expected_baseline)
    ignored_delta, _ = compute_opd_advantage(FakeData(torch.full((1, 3), 99.0)), cc_enabled=False)
    assert torch.allclose(ignored_delta, expected_baseline)

    base_prompt = "Write an update."
    criteria = ["Use JSON.", "Mention latency."]
    full_instruction = f"{base_prompt} {criteria[0]} {criteria[1]}"
    first_cf = build_cf_instruction(full_instruction, criteria, 0)
    second_cf = build_cf_instruction(full_instruction, criteria, 1)
    assert criteria[0] not in first_cf and criteria[1] in first_cf
    assert criteria[1] not in second_cf and criteria[0] in second_cf

    metadata = {
        "extra_info": np.array([{"criteria": criteria, "base_prompt": base_prompt}], dtype=object),
        "cc_full_instruction": np.array([full_instruction], dtype=object),
        "task_type": np.array(["instruction_following"], dtype=object),
    }
    jobs, _ = build_online_cc_rubric_jobs(metadata, 1.0, seed=21, step=0)
    assert len(jobs) == 2 and {job.rubric_index for job in jobs} == {0, 1}
    assert {job.instruction for job in jobs} == {first_cf, second_cf}

    response_mask = torch.tensor([[1.0, 1.0, 0.0]])
    responses = torch.tensor([[10, 11, 0]])
    scoring_batch = build_counterfactual_scoring_batch(
        jobs=jobs,
        responses=responses,
        response_mask=response_mask,
        tokenizer=FakeTokenizer(),
        max_prompt_length=32,
        pad_token_id=0,
        truncation="left",
        apply_chat_template_kwargs={},
    )
    assert tuple(scoring_batch.batch["responses"].shape) == (2, 3)
    assert torch.equal(scoring_batch.batch["input_ids"][:, -3:], responses.expand(2, -1))

    delta, _ = aggregate_online_cc_deltas(
        full_log_probs=torch.zeros((1, 3)),
        counterfactual_log_probs=torch.tensor([[-0.25, 0.5, 0.0], [-0.5, 0.25, 0.0]]),
        jobs=jobs,
        response_mask=response_mask,
    )
    expected_delta = torch.tensor([[0.75, -0.75, 0.0]])
    assert torch.allclose(delta, expected_delta), (delta, expected_delta)
    cc_advantage, _ = compute_opd_advantage(
        FakeData(delta), cc_enabled=True, cc_lambda=2.0, cc_delta_clip=1.0
    )
    expected_cc = torch.tensor([[2.0, -0.5, 0.0]])
    assert torch.allclose(cc_advantage, expected_cc), (cc_advantage, expected_cc)

    try:
        compute_opd_advantage(FakeData(), cc_enabled=True)
    except KeyError:
        pass
    else:
        raise AssertionError("LOO mode accepted a batch without counterfactual deltas")

    print(json.dumps({"vanilla_opd": baseline.tolist(), "loo_delta": delta.tolist(), "loo_advantage": cc_advantage.tolist()}))


if __name__ == "__main__":
    main()
