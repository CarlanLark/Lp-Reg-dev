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

from collections import defaultdict
import copy
import numpy as np
import torch
import random


import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    whiten: bool = True,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        if whiten:
            advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
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
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (torch.sum(loss_mask, dim=-1) + 1e-8)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
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
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    # assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange


    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # pg_losses3 = -advantages * clip_ratio_c
    # clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    # pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl




def compute_policy_loss_gspo(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="seq-mean-token-mean",
    clip_ratio=0.0003,
    clip_ratio_low=None,
    clip_ratio_high=None,
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    clip_ratio_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
    clip_ratio_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_8020_split(
    old_log_prob,
    log_prob,
    entropy,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    # assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange


    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    
    # update response_mask, find top 20% entropy idx
    all_valid = (response_mask > 0)
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0].detach() 
    all_valid_entropy = entropy[all_valid].detach().reshape(-1).cpu()
    
    if len(all_valid_entropy) > 0:
        top_20_percent_num = max(1, int(len(all_valid_entropy) * 0.2))
        top_20_entropy_idx = torch.topk(all_valid_entropy, k=top_20_percent_num, largest=True).indices
        

        mask_all_tokens = torch.ones(len(all_valid_entropy), dtype=torch.bool)
        mask_all_tokens[top_20_entropy_idx] = False

        tokens_to_mask_idx = torch.nonzero(mask_all_tokens, as_tuple=True)[0]
        
        tokens_to_mask_original_idxs = all_valid_idx[tokens_to_mask_idx]
        
        if len(tokens_to_mask_original_idxs) > 0:
            response_mask[tokens_to_mask_original_idxs // advantages.shape[1], tokens_to_mask_original_idxs % advantages.shape[1]] = 0
    
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # pg_losses3 = -advantages * clip_ratio_c
    # clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    # pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    k_percent=0.2,
    ppo_kl_coef=1,
):
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob

    abs_kl = negative_approx_kl.abs()

    ratio = torch.exp(negative_approx_kl)

    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)

    pg_losses1 = -advantages * ratio

    pg_losses_kl = - advantages * ratio + ppo_kl_coef * abs_kl

    pg_losses = pg_losses1

    all_valid = (response_mask > 0)
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0] 
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()
    all_valid_prob = torch.exp(all_valid_logp)

    k = min(k_percent, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * k / 100))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices
        
        if len(large_cov_idxs) != 0:
            
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    return pg_loss, torch.tensor(0.), ppo_kl_abs


def compute_policy_loss_lp_reg(
    old_log_prob,
    log_prob,
    pos_tgt_log_prob,
    neg_tgt_log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    logp_pos_k_percent=0,
    logp_neg_k_percent=0.01,
    ppo_kl_coef=1,
    kl_type="low_var_kl",
):

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    with torch.no_grad():
        ppo_kl_mean = verl_F.masked_mean(kl_penalty(log_prob, old_log_prob, kl_type), response_mask)

    # calculate ratio
    ratio = torch.exp(negative_approx_kl)
    
    # calculate loss
    pg_losses1 = -advantages * ratio
    ratio_clipped = ratio.clamp(1 - cliprange_low, 1 + cliprange_high)
    pg_losses2 = -advantages * ratio_clipped
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    
    # calculate clipfrac
    with torch.no_grad():
        pg_clipfrac = verl_F.masked_mean((pg_losses2 > pg_losses1).float(), response_mask)
    
    # find valid indices
    all_valid_mask = response_mask > 0
    all_valid_flat_idx = all_valid_mask.reshape(-1).nonzero(as_tuple=True)[0]
    
    if len(all_valid_flat_idx) == 0:
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        return pg_loss, pg_clipfrac, ppo_kl_mean

    # get sequence length for index conversion
    _, seq_len = advantages.shape
    advantages_flat = advantages.reshape(-1)
    log_prob_flat = log_prob.reshape(-1)
    
    # find positive and negative sample indices
    pos_indices = all_valid_flat_idx[advantages_flat[all_valid_flat_idx] > 0]
    neg_indices = all_valid_flat_idx[advantages_flat[all_valid_flat_idx] < 0]

    def apply_kl_penalty_selective(target_flat_idx, tgt_log_prob):
        print(f"dev_debug: apply_kl_penalty_selective_len: {len(target_flat_idx)}")
        if len(target_flat_idx) == 0:
            return
            
        batch_idx, seq_idx = target_flat_idx // seq_len, target_flat_idx % seq_len

        reg_token_num = (log_prob[batch_idx, seq_idx].detach() < tgt_log_prob[batch_idx, seq_idx]).sum().cpu()
        print(f"dev_debug: reg_token_num: {reg_token_num}")

        mask = log_prob[batch_idx, seq_idx].detach() < tgt_log_prob[batch_idx, seq_idx]
        batch_idx, seq_idx = batch_idx[mask], seq_idx[mask]

        if len(batch_idx) == 0:
            return

        target_kl = kl_penalty(log_prob[batch_idx, seq_idx], tgt_log_prob[batch_idx, seq_idx], kl_type)
        pg_losses[batch_idx, seq_idx] = -advantages[batch_idx, seq_idx] * ratio[batch_idx, seq_idx] + ppo_kl_coef * target_kl
    
    # positive samples
    if len(pos_indices) > 0:
        pos_logp_k = max(1, int(len(pos_indices) * logp_pos_k_percent))
        if pos_logp_k > 1:
            with torch.no_grad():
                pos_logp_values = log_prob_flat[pos_indices]
                _, pos_logp_indices = pos_logp_values.topk(k=pos_logp_k, largest=False)
                pos_target_idx = pos_indices[pos_logp_indices]
            apply_kl_penalty_selective(pos_target_idx, pos_tgt_log_prob)
    
    # negative samples
    if len(neg_indices) > 0:
        neg_logp_k = max(1, int(len(neg_indices) * logp_neg_k_percent))
        if neg_logp_k > 1:
            with torch.no_grad():
                neg_logp_values = log_prob_flat[neg_indices]
                _, neg_logp_indices = neg_logp_values.topk(k=neg_logp_k, largest=False)
                neg_target_idx = neg_indices[neg_logp_indices]
            apply_kl_penalty_selective(neg_target_idx, neg_tgt_log_prob)
    
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    return pg_loss, pg_clipfrac, ppo_kl_mean

def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    clip_ratio=0.0002,
    clip_cov_lb=1.0,
    clip_cov_ub=5.0,
):
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)
    
    cov_all = (advantages- verl_F.masked_mean(advantages, response_mask)) * (log_prob- verl_F.masked_mean(log_prob.detach(), response_mask))
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf
    
    clip_num = max(int(clip_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)
    
    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[:min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)
    
    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0
    
    pg_clipfrac = verl_F.masked_mean((corr==0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl

def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        kld = logprob - ref_logprob
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "abs":
        kld = (logprob - ref_logprob).abs()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "mse":
        kld = 0.5 * (logprob - ref_logprob).square()
        return torch.clamp(kld, min=-10, max=10)

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
