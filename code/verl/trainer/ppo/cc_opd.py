"""Utilities for sampled-token CC-OPD batch plumbing."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.position_ids import compute_position_id_with_mask
from verl.utils.torch_functional import postprocess_data


@dataclass(frozen=True)
class RubricScoringJob:
    sample_index: int
    rubric_index: int
    total_rubrics: int
    selected_count: int
    scale: float
    instruction: str
    task_type: object | None = None


_INLINE_REMOVE_DOUBLE_SPACE = re.compile(r" {2,}")
_INLINE_REMOVE_LEADING_SPACES_BEFORE_NL = re.compile(r" +\n")
_INLINE_REMOVE_TRAILING_SPACES_AFTER_NL = re.compile(r"\n +")
_INLINE_REMOVE_TRIPLE_NL = re.compile(r"\n{3,}")


def _normalize_inline_seam(text: str) -> str:
    text = _INLINE_REMOVE_DOUBLE_SPACE.sub(" ", text)
    text = _INLINE_REMOVE_LEADING_SPACES_BEFORE_NL.sub("\n", text)
    text = _INLINE_REMOVE_TRAILING_SPACES_AFTER_NL.sub("\n", text)
    text = _INLINE_REMOVE_TRIPLE_NL.sub("\n\n", text)
    return text.strip()


def build_cf_instruction(
    full_instruction: str,
    criteria: list[str],
    excluded_index: int,
) -> str:
    """Remove one exact constraint span from the full teacher instruction."""
    if not 0 <= excluded_index < len(criteria):
        raise IndexError(
            f"excluded_index {excluded_index} out of bounds for criteria of length {len(criteria)}"
        )

    target = criteria[excluded_index]
    if not isinstance(target, str) or not target.strip():
        raise ValueError(
            f"inline_remove requires non-empty rubric string at index {excluded_index}; got {target!r}"
        )
    if not isinstance(full_instruction, str) or not full_instruction.strip():
        raise ValueError("inline_remove requires non-empty full_instruction")
    idx = full_instruction.find(target)
    if idx == -1:
        raise ValueError(
            "inline_remove cannot locate rubric substring in full_instruction; "
            "ensure prepare data wrote extra_info.full_instruction verbatim from the "
            "teacher's training prompt and that extra_info.criteria are exact rubric descriptions"
        )
    seam = full_instruction[:idx] + full_instruction[idx + len(target) :]
    return _normalize_inline_seam(seam)


def _coerce_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Expected dict-like metadata for online CC-OPD, got {type(value).__name__}")


def _coerce_criteria(value: object, sample_index: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(
        "online CC-OPD requires extra_info['criteria'] to be a list/tuple/array; "
        f"sample {sample_index} has {type(value).__name__}"
    )


def _sequence_length(value: object, key: str) -> int:
    try:
        return len(value)
    except TypeError as exc:
        raise ValueError(f"online CC-OPD non_tensor_batch['{key}'] must be a per-sample sequence") from exc


def _extract_message_content(raw_prompt: object) -> str | None:
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, list) and raw_prompt:
        first_message = raw_prompt[0]
        if isinstance(first_message, dict) and isinstance(first_message.get("content"), str):
            return first_message["content"]
    if isinstance(raw_prompt, dict) and isinstance(raw_prompt.get("content"), str):
        return raw_prompt["content"]
    if isinstance(raw_prompt, str):
        return raw_prompt
    return None


def _full_instruction_for_sample(non_tensor_batch: dict[str, np.ndarray], sample_index: int, extra_info: dict[str, Any]) -> str:
    if "cc_full_instruction" in non_tensor_batch:
        value = non_tensor_batch["cc_full_instruction"][sample_index]
        if isinstance(value, str) and value.strip():
            return value
    if "cc_env_kwargs" in non_tensor_batch:
        env_kwargs = _coerce_mapping(non_tensor_batch["cc_env_kwargs"][sample_index])
        question = env_kwargs.get("question")
        if isinstance(question, str) and question.strip():
            return question
    if "env_kwargs" in non_tensor_batch:
        env_kwargs = _coerce_mapping(non_tensor_batch["env_kwargs"][sample_index])
        question = env_kwargs.get("question")
        if isinstance(question, str) and question.strip():
            return question
    if "raw_prompt" in non_tensor_batch:
        content = _extract_message_content(non_tensor_batch["raw_prompt"][sample_index])
        if content:
            return content
    for key in ("full_instruction", "question", "prompt"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Unable to find full instruction for online CC-OPD counterfactual scoring")


def _ratio_to_selected_count(total_rubrics: int, rubric_sample_ratio: float) -> int:
    if total_rubrics <= 0 or rubric_sample_ratio <= 0.0:
        return 0
    if rubric_sample_ratio >= 1.0:
        return total_rubrics
    return max(1, int(math.ceil(float(rubric_sample_ratio) * total_rubrics)))


def build_online_cc_rubric_jobs(
    non_tensor_batch: dict[str, np.ndarray],
    rubric_sample_ratio: float,
    seed: int,
    step: int,
) -> tuple[list[RubricScoringJob], dict[str, float]]:
    if not 0.0 <= float(rubric_sample_ratio) <= 1.0:
        raise ValueError(f"rubric_sample_ratio must be in [0, 1], got {rubric_sample_ratio}")
    if "extra_info" not in non_tensor_batch:
        raise KeyError("online CC-OPD requires non_tensor_batch['extra_info'] with a criteria list")

    jobs: list[RubricScoringJob] = []
    extra_infos = non_tensor_batch["extra_info"]
    sample_count = _sequence_length(extra_infos, "extra_info")
    task_types = non_tensor_batch.get("task_type")
    if task_types is not None and _sequence_length(task_types, "task_type") != sample_count:
        raise ValueError("online CC-OPD task_type length must match extra_info length")
    zero_rubric_samples = 0
    rubric_counts: list[int] = []
    for sample_index, raw_extra_info in enumerate(extra_infos):
        extra_info = _coerce_mapping(raw_extra_info)
        criteria = _coerce_criteria(extra_info.get("criteria"), sample_index=sample_index)
        total_rubrics = len(criteria)
        rubric_counts.append(total_rubrics)
        selected_count = _ratio_to_selected_count(total_rubrics, float(rubric_sample_ratio))
        if selected_count == 0:
            zero_rubric_samples += 1
            continue

        full_instruction = _full_instruction_for_sample(non_tensor_batch, sample_index, extra_info)
        if selected_count == total_rubrics:
            selected_indices = np.arange(total_rubrics, dtype=np.int64)
        else:
            rng_seed = int(seed) + int(step) * 1_000_003 + sample_index
            rng = np.random.default_rng(rng_seed)
            selected_indices = np.sort(rng.choice(total_rubrics, size=selected_count, replace=False))

        scale = float(total_rubrics) / float(selected_count)
        task_type = task_types[sample_index] if task_types is not None else None
        for rubric_index in selected_indices.tolist():
            cf_instruction = build_cf_instruction(
                full_instruction=full_instruction,
                criteria=criteria,
                excluded_index=int(rubric_index),
            )
            jobs.append(
                RubricScoringJob(
                    sample_index=sample_index,
                    rubric_index=int(rubric_index),
                    total_rubrics=total_rubrics,
                    selected_count=selected_count,
                    scale=scale,
                    instruction=cf_instruction,
                    task_type=task_type,
                )
            )

    rubric_counts_arr = np.asarray(rubric_counts, dtype=np.int64) if rubric_counts else np.zeros(0, dtype=np.int64)
    build_stats = {
        "cc_opd/online_zero_rubric_samples": float(zero_rubric_samples),
        "cc_opd/online_zero_rubric_frac": float(zero_rubric_samples) / float(sample_count) if sample_count else 0.0,
        "cc_opd/online_rubrics_per_sample_max": float(rubric_counts_arr.max()) if rubric_counts_arr.size else 0.0,
        "cc_opd/online_rubrics_per_sample_min": float(rubric_counts_arr.min()) if rubric_counts_arr.size else 0.0,
    }
    return jobs, build_stats


def build_counterfactual_scoring_batch(
    jobs: list[RubricScoringJob],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    max_prompt_length: int,
    pad_token_id: int,
    truncation: str,
    apply_chat_template_kwargs: dict[str, object],
) -> DataProto:
    if not jobs:
        raise ValueError("Cannot build counterfactual scoring batch with no jobs")
    if responses.ndim != 2:
        raise ValueError(f"responses must be rank-2, got shape {tuple(responses.shape)}")
    if tuple(response_mask.shape) != tuple(responses.shape):
        raise ValueError(
            f"response_mask shape {tuple(response_mask.shape)} must match responses shape {tuple(responses.shape)}"
        )

    prompt_ids_list: list[torch.Tensor] = []
    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []
    position_ids_list: list[torch.Tensor] = []
    responses_list: list[torch.Tensor] = []
    task_types: list[object] = []

    for job in jobs:
        if job.sample_index < 0 or job.sample_index >= responses.shape[0]:
            raise IndexError(
                f"online CC-OPD job sample_index {job.sample_index} is outside response batch size {responses.shape[0]}"
            )
        chat = [{"role": "user", "content": job.instruction}]
        raw_prompt = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        model_inputs = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids, prompt_mask = postprocess_data(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=truncation,
        )

        response = responses[job.sample_index].detach().cpu().long()
        sample_response_mask = response_mask[job.sample_index].detach().cpu().to(dtype=prompt_mask.dtype)
        input_ids = torch.cat([prompt_ids[0].long(), response], dim=-1)
        attention_mask = torch.cat([prompt_mask[0].to(dtype=sample_response_mask.dtype), sample_response_mask], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        prompt_ids_list.append(prompt_ids[0].long())
        responses_list.append(response)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
        task_types.append(job.task_type)

    tensors = {
        "prompts": torch.stack(prompt_ids_list, dim=0),
        "responses": torch.stack(responses_list, dim=0),
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attention_mask_list, dim=0),
        "position_ids": torch.stack(position_ids_list, dim=0),
    }
    non_tensors = {}
    if all(task_type is not None for task_type in task_types):
        non_tensors["task_type"] = np.array(task_types, dtype=object)
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, auto_padding=True)


def aggregate_online_cc_deltas(
    full_log_probs: torch.Tensor,
    counterfactual_log_probs: torch.Tensor,
    jobs: list[RubricScoringJob],
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tuple(full_log_probs.shape) != tuple(response_mask.shape):
        raise ValueError(
            f"full_log_probs shape {tuple(full_log_probs.shape)} must match response_mask shape {tuple(response_mask.shape)}"
        )
    if counterfactual_log_probs.ndim != full_log_probs.ndim:
        raise ValueError(
            f"counterfactual_log_probs rank {counterfactual_log_probs.ndim} must match full_log_probs rank {full_log_probs.ndim}"
        )
    if tuple(counterfactual_log_probs.shape[1:]) != tuple(full_log_probs.shape[1:]):
        raise ValueError(
            f"counterfactual_log_probs shape {tuple(counterfactual_log_probs.shape)} is incompatible with "
            f"full_log_probs shape {tuple(full_log_probs.shape)}"
        )
    if counterfactual_log_probs.shape[0] != len(jobs):
        raise ValueError(
            f"counterfactual rows {counterfactual_log_probs.shape[0]} do not match jobs {len(jobs)}"
        )
    delta = torch.zeros_like(full_log_probs)
    cf_log_probs = counterfactual_log_probs.to(device=full_log_probs.device, dtype=full_log_probs.dtype)
    mask = response_mask.to(device=full_log_probs.device, dtype=full_log_probs.dtype)
    per_job_token_deltas: list[torch.Tensor] = []
    for job_index, job in enumerate(jobs):
        sample_index = job.sample_index
        if sample_index < 0 or sample_index >= full_log_probs.shape[0]:
            raise IndexError(
                f"online CC-OPD job sample_index {sample_index} is outside full_log_probs batch size {full_log_probs.shape[0]}"
            )
        token_delta = full_log_probs[sample_index] - cf_log_probs[job_index]
        delta[sample_index] += float(job.scale) * token_delta
        per_job_token_deltas.append(token_delta * mask[sample_index])
    delta = delta * mask
    if not torch.isfinite(delta).all():
        raise ValueError("online CC-OPD produced non-finite deltas")

    masked_delta = delta[mask.bool()].float()
    if per_job_token_deltas:
        per_job_stack = torch.stack(per_job_token_deltas, dim=0)
        masked_per_job = per_job_stack[per_job_stack.abs() > 0].float() if per_job_stack.numel() else per_job_stack.float()
    else:
        masked_per_job = torch.zeros(0, dtype=torch.float32)

    if masked_delta.numel() > 0:
        stats = {
            "cc_opd/online_delta_mean_signed": float(masked_delta.mean().item()),
            "cc_opd/online_delta_std": float(masked_delta.std(unbiased=False).item()) if masked_delta.numel() > 1 else 0.0,
            "cc_opd/online_delta_p50": float(masked_delta.median().item()),
            "cc_opd/online_delta_p90": float(torch.quantile(masked_delta, 0.9).item()),
            "cc_opd/online_delta_p99": float(torch.quantile(masked_delta, 0.99).item()),
            "cc_opd/online_delta_pos_frac": float((masked_delta > 0).float().mean().item()),
            "cc_opd/online_delta_neg_frac": float((masked_delta < 0).float().mean().item()),
            "cc_opd/online_delta_zero_frac": float((masked_delta == 0).float().mean().item()),
            "cc_opd/online_per_job_delta_abs_mean": float(masked_per_job.abs().mean().item()) if masked_per_job.numel() else 0.0,
        }
    else:
        stats = {
            "cc_opd/online_delta_mean_signed": 0.0,
            "cc_opd/online_delta_std": 0.0,
            "cc_opd/online_delta_p50": 0.0,
            "cc_opd/online_delta_p90": 0.0,
            "cc_opd/online_delta_p99": 0.0,
            "cc_opd/online_delta_pos_frac": 0.0,
            "cc_opd/online_delta_neg_frac": 0.0,
            "cc_opd/online_delta_zero_frac": 0.0,
            "cc_opd/online_per_job_delta_abs_mean": 0.0,
        }
    return delta, stats
