# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import torch

import verl.utils.torch_functional as verl_F


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training. Consider a configurable normalizer if this mode is used.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_opd_advantage(
        data,
        cc_enabled: bool = False,
        cc_delta_key: str = "cc_delta_log_probs",
        cc_lambda: float = 1.0,
        cc_delta_clip: float = 0.0,
        metrics_out: dict | None = None,
):
    """Compute sampled-token OPD rewards with an optional LOO correction.

    With ``cc_enabled=False``, the token reward is the teacher log probability
    minus the student log probability on the sampled token. With CC-OPD enabled,
    add ``cc_lambda`` times the clipped, aggregated LOO teacher delta from
    ``data.batch[cc_delta_key]``.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        cc_enabled: bool, whether to add sampled-token CC-OPD delta correction
        cc_delta_key: str, batch key containing realized-token CC deltas
        cc_lambda: float, multiplier for the CC correction
        cc_delta_clip: float, symmetric clipping bound for CC deltas; <=0 disables clipping
        metrics_out: optional dictionary filled with CC-OPD diagnostics
    Returns:
        advantages and returns, each shaped (batch_size, response_length)
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]
    
    # Sampled-token reverse-KL estimator: log P_student(y_t) - log P_teacher(y_t).
    student_log_probs = data.batch["old_log_probs"]
    teacher_log_probs = data.batch["ref_log_prob"]
    kl_divergence = student_log_probs - teacher_log_probs
    kl_divergence = kl_divergence * response_mask

    cc_token_bonus = torch.zeros_like(kl_divergence)
    if cc_enabled:
        if not cc_delta_key:
            raise ValueError("cc_delta_key must be non-empty when cc_enabled=True")
        if cc_delta_key not in data.batch:
            raise KeyError(
                f"CC-OPD enabled but batch key '{cc_delta_key}' is missing. "
                "Provide realized-token leave-one-out teacher deltas or disable algorithm.opd.cc.enabled."
            )
        cc_delta_raw = data.batch[cc_delta_key].to(device=kl_divergence.device, dtype=kl_divergence.dtype)
        if cc_delta_raw.shape != kl_divergence.shape:
            raise ValueError(
                f"CC-OPD delta shape {tuple(cc_delta_raw.shape)} does not match OPD reward shape {tuple(kl_divergence.shape)}"
            )
        if cc_delta_clip and cc_delta_clip > 0.0:
            cc_delta = torch.clamp(cc_delta_raw, min=-float(cc_delta_clip), max=float(cc_delta_clip))
        else:
            cc_delta = cc_delta_raw
        cc_token_bonus = float(cc_lambda) * cc_delta * response_mask
        data.batch["cc_token_bonus"] = cc_token_bonus

        if metrics_out is not None:
            mask_bool = response_mask.bool()
            valid_count = float(response_mask.sum().item())
            denom = max(valid_count, 1.0)
            raw_masked = cc_delta_raw[mask_bool].float()
            clipped_masked = cc_delta[mask_bool].float()
            bonus_masked = cc_token_bonus[mask_bool].float()
            kl_masked = kl_divergence[mask_bool].float()
            if cc_delta_clip and cc_delta_clip > 0.0:
                clip_hit = ((cc_delta_raw.abs() >= float(cc_delta_clip)).to(dtype=response_mask.dtype) * response_mask).sum().item()
                clip_hit_frac = float(clip_hit) / denom
            else:
                clip_hit_frac = 0.0
            if raw_masked.numel() > 0:
                metrics_out["cc_opd/delta_pre_clip_abs_mean"] = float(raw_masked.abs().mean().item())
                metrics_out["cc_opd/delta_pre_clip_abs_max"] = float(raw_masked.abs().max().item())
                metrics_out["cc_opd/delta_pre_clip_p99"] = float(torch.quantile(raw_masked.abs(), 0.99).item())
                metrics_out["cc_opd/delta_post_clip_abs_mean"] = float(clipped_masked.abs().mean().item())
                metrics_out["cc_opd/delta_clip_hit_frac"] = float(clip_hit_frac)
                metrics_out["cc_opd/cc_token_bonus_abs_mean"] = float(bonus_masked.abs().mean().item())
                metrics_out["cc_opd/cc_token_bonus_abs_max"] = float(bonus_masked.abs().max().item())
                metrics_out["cc_opd/kl_divergence_abs_mean"] = float(kl_masked.abs().mean().item())
                metrics_out["cc_opd/kl_divergence_abs_max"] = float(kl_masked.abs().max().item())
                kl_abs_mean = max(metrics_out["cc_opd/kl_divergence_abs_mean"], 1e-8)
                metrics_out["cc_opd/cc_to_kl_ratio_mean"] = metrics_out["cc_opd/cc_token_bonus_abs_mean"] / kl_abs_mean
                cc_sign = torch.sign(bonus_masked)
                kl_sign = torch.sign(-kl_masked)
                metrics_out["cc_opd/cc_kl_sign_agreement"] = float((cc_sign == kl_sign).float().mean().item())
                metrics_out["cc_opd/cc_lambda"] = float(cc_lambda)
                metrics_out["cc_opd/cc_delta_clip"] = float(cc_delta_clip)
            else:
                metrics_out["cc_opd/delta_pre_clip_abs_mean"] = 0.0
                metrics_out["cc_opd/delta_pre_clip_abs_max"] = 0.0
                metrics_out["cc_opd/delta_pre_clip_p99"] = 0.0
                metrics_out["cc_opd/delta_post_clip_abs_mean"] = 0.0
                metrics_out["cc_opd/delta_clip_hit_frac"] = 0.0
                metrics_out["cc_opd/cc_token_bonus_abs_mean"] = 0.0
                metrics_out["cc_opd/cc_token_bonus_abs_max"] = 0.0
                metrics_out["cc_opd/kl_divergence_abs_mean"] = 0.0
                metrics_out["cc_opd/kl_divergence_abs_max"] = 0.0
                metrics_out["cc_opd/cc_to_kl_ratio_mean"] = 0.0
                metrics_out["cc_opd/cc_kl_sign_agreement"] = 0.0
                metrics_out["cc_opd/cc_lambda"] = float(cc_lambda)
                metrics_out["cc_opd/cc_delta_clip"] = float(cc_delta_clip)
     
    token_level_rewards = -kl_divergence + cc_token_bonus
    data.batch["token_level_rewards"] = token_level_rewards

    advantages = token_level_rewards * response_mask
    returns = advantages
    return advantages, returns
