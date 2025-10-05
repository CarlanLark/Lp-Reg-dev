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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple, Union

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty, compute_policy_loss_kl_cov, compute_policy_loss_clip_cov, compute_policy_loss_8020_split, compute_policy_loss_gspo, compute_policy_loss_lp_reg
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

        if self.use_fused_kernels:
            from verl.utils.experimental.torch_functional import FusedLinearForPPO

            self.fused_linear_for_ppo = FusedLinearForPPO()

            # FusedLinearForPPO has an error when compiled, disable for now
            # if self.config.get("use_torch_compile", True):
            #     self.fused_linear_for_ppo.compile(dynamic=True)
    def apply_min_p(self, logits, min_p=0.1, minp_soft=False, regular_type="minp"):
        """ 
        Filters logits using adaptive probability thresholding.
        """
        chunk_size = 4096
        logits_chunks = torch.split(logits.detach(), split_size_or_sections=chunk_size, dim=0)
        modified_logits = []
        for logit_chunk in logits_chunks:
            cloned_logit_chunk = logit_chunk.clone()
            # calculate probability distribution
            probability_values = torch.nn.functional.softmax(cloned_logit_chunk, dim=-1)
            if regular_type == "minp":
                print(f"dev_debug: regular_type = {regular_type}")
                # maximum probability of each token (for dynamic adjustment of threshold)
                max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
                adjusted_min_p = min_p * max_probabilities  # dynamic threshold
                # generate invalid token mask (tokens with probability below threshold)
                valid_token_mask = probability_values >= adjusted_min_p
                invalid_mask = ~valid_token_mask  # invalid token mask

                if not minp_soft:
                    cloned_logit_chunk[invalid_mask] = -1e4
                else:
                    min_logits = torch.amin(cloned_logit_chunk, dim=-1, keepdim=True).repeat_interleave(cloned_logit_chunk.size(-1), dim=-1)
                    cloned_logit_chunk[invalid_mask] = min_logits[invalid_mask]
            elif regular_type == "fixed":
                print(f"dev_debug: regular_type = {regular_type}")
                valid_token_mask = probability_values >= min_p
                invalid_mask = ~valid_token_mask  # invalid token mask
                cloned_logit_chunk[invalid_mask] = -1e4
            else:
                raise ValueError(f"regular_type {regular_type} not supported")
            modified_logits.append(cloned_logit_chunk)
        modified_logits = torch.cat(modified_logits, dim=0).detach()
        return modified_logits  # return modified cloned tensor

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, min_p=0, minp_soft=False, regular_type="minp", save_top_k=0) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            top_k_log_probs: # (bs, response_len, save_top_k) - only when save_top_k > 0
            top_k_input_ids: # (bs, response_len, save_top_k) - only when save_top_k > 0
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)
        
        # Initialize top_k variables
        top_k_log_probs = None
        top_k_input_ids = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    if min_p > 0:
                        raise NotImplementedError("FusedLinearForPPO does not support min_p")
                    if save_top_k > 0:
                        raise NotImplementedError("FusedLinearForPPO does not support save_top_k")
                    hidden_states = output.last_hidden_state
                    vocab_weights = self.actor_module.lm_head.weight

                    log_probs, entropy_rmpad = self.fused_linear_for_ppo(
                        hidden_states=hidden_states.squeeze(0),
                        vocab_weights=vocab_weights,
                        input_ids=input_ids_rmpad_rolled,
                        temperature=temperature,
                    )

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    # import madbg; madbg.set_trace(ip='0.0.0.0', port=1337+torch.distributed.get_rank()) 
                    if min_p > 0:
                        print(f"dev_debug: min_p = {min_p} > 0")
                        logits_rmpad = self.apply_min_p(logits_rmpad, min_p, minp_soft, regular_type)

                    # logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if save_top_k > 0:
                    # calculate top-k log_probs and corresponding token_ids
                    with torch.no_grad():
                        # get top-k logits and indices
                        topk_logits, topk_indices = torch.topk(logits_rmpad, k=save_top_k, dim=-1)  # (total_nnz, save_top_k)
                        topk_logits = topk_logits / temperature
                        topk_log_probs_rmpad = torch.log_softmax(topk_logits, dim=-1)  # (total_nnz, save_top_k)
                        
                        # gather top-k results if sp > 1
                        if self.use_ulysses_sp:
                            topk_log_probs_rmpad = gather_outpus_and_unpad(
                                topk_log_probs_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                            topk_indices = gather_outpus_and_unpad(
                                topk_indices,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                        
                        # pad back to (bsz, seqlen, save_top_k)
                        full_topk_log_probs = pad_input(
                            hidden_states=topk_log_probs_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        
                        # only return response part:
                        top_k_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]  # (bsz, response_length, save_top_k)
                        top_k_input_ids = full_topk_indices[:, -response_length - 1 : -1, :]  # (bsz, response_length, save_top_k)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    if min_p > 0:
                        raise NotImplementedError("FusedLinearForPPO does not support min_p")
                    if save_top_k > 0:
                        raise NotImplementedError("FusedLinearForPPO does not support save_top_k")
                    hidden_states = output.last_hidden_state
                    vocab_weights = self.actor_module.lm_head.weight

                    log_probs, entropy = self.fused_linear_for_ppo(
                        hidden_states=hidden_states[:, -response_length - 1 : -1, :],
                        vocab_weights=vocab_weights,
                        input_ids=micro_batch["responses"],
                        temperature=temperature,
                    )

                else:
                    logits = output.logits

                    if min_p > 0:
                        print(f"dev_debug: min_p = {min_p} > 0")
                        logits = self.apply_min_p(logits, min_p, minp_soft, regular_type)

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                    
                    if save_top_k > 0:
                        # calculate top-k log_probs and corresponding token_ids
                        with torch.no_grad():
                            # get response part of original logits for top-k calculation
                            response_logits = output.logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                            if min_p > 0:
                                response_logits = self.apply_min_p(response_logits, min_p, minp_soft, regular_type)
                            
                            # get top-k logits and indices
                            topk_logits, topk_indices = torch.topk(response_logits, k=save_top_k, dim=-1)  # (bsz, response_length, save_top_k)
                            topk_logits = topk_logits / temperature
                            top_k_log_probs = torch.log_softmax(topk_logits, dim=-1)  # (bsz, response_length, save_top_k)
                            top_k_input_ids = topk_indices  # (bsz, response_length, save_top_k)
            if save_top_k > 0:
                return entropy, log_probs, top_k_log_probs, top_k_input_ids
            else:
                return entropy, log_probs
    
    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, minp=0, minp_soft=False, regular_type="minp") -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            log_probs (torch.Tensor): the log_prob tensor of shape [batch_size, response_length]
            entropys (torch.Tensor): the entropy tensor of shape [batch_size, response_length] (if calculate_entropy=True)
            top_k_log_probs (torch.Tensor): top-k log probabilities of shape [batch_size, response_length, save_top_k] (if save_top_k > 0)
            top_k_input_ids (torch.Tensor): top-k token ids of shape [batch_size, response_length, save_top_k] (if save_top_k > 0)
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        save_top_k = data.meta_info.get("save_top_k", 0)  # default to 0 if not provided

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        top_k_log_probs_lst = []
        top_k_input_ids_lst = []
        
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                if save_top_k > 0:
                    entropy, log_probs, top_k_log_probs, top_k_input_ids = self._forward_micro_batch(
                        micro_batch, 
                        temperature=temperature, 
                        calculate_entropy=calculate_entropy, 
                        min_p=minp, 
                        minp_soft=minp_soft, 
                        regular_type=regular_type, 
                        save_top_k=save_top_k
                    )
                    top_k_log_probs_lst.append(top_k_log_probs)
                    top_k_input_ids_lst.append(top_k_input_ids)
                else:
                    entropy, log_probs = self._forward_micro_batch(
                        micro_batch, 
                        temperature=temperature, 
                        calculate_entropy=calculate_entropy, 
                        min_p=minp, 
                        minp_soft=minp_soft, 
                        regular_type=regular_type, 
                        save_top_k=save_top_k
                    )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        # Handle top_k results
        top_k_log_probs = None
        top_k_input_ids = None
        if save_top_k > 0:
            top_k_log_probs = torch.concat(top_k_log_probs_lst, dim=0)
            top_k_input_ids = torch.concat(top_k_input_ids_lst, dim=0)
        
        # Handle dynamic batch size reordering
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]
            if save_top_k > 0:
                top_k_log_probs = top_k_log_probs[revert_indices]
                top_k_input_ids = top_k_input_ids[revert_indices]

        # Return appropriate tuple based on save_top_k
        if save_top_k > 0:
            return log_probs, entropys, top_k_log_probs, top_k_input_ids
        else:
            return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        # data.batch["entropy"] is entropy of \pi_old
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.minp_old_log_prob:
            select_keys.append("minp_old_log_probs")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()


                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    loss_mode = self.config.loss_mode
                    

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0 or loss_mode == "lp_reg" or loss_mode == "8020_split" or loss_mode == "check_lp_reg":
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                    
                    if loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                        )
                    
                    elif loss_mode == "gspo":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_gspo(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            clip_ratio=clip_ratio,
                            clip_ratio_low=clip_ratio_low,
                            clip_ratio_high=clip_ratio_high,
                        )
                    
                    elif loss_mode == "clip_cov":
                        pg_loss, pg_clipfrac, ppo_kl= compute_policy_loss_clip_cov(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                            clip_ratio=self.config.clip_cov_ratio,
                            clip_cov_lb=self.config.clip_cov_lb,
                            clip_cov_ub=self.config.clip_cov_ub,
                        )

                    elif loss_mode == "kl_cov":
                        pg_loss, pg_clipfrac, ppo_kl= compute_policy_loss_kl_cov(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                            k_percent=self.config.k_percent,
                            ppo_kl_coef=self.config.ppo_kl_coef,
                        )

                    elif loss_mode == "8020_split":
                        pg_loss, pg_clipfrac, ppo_kl= compute_policy_loss_8020_split(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            entropy=entropy,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                        )

                    elif loss_mode == "lp_reg" or loss_mode == "check_lp_reg":
                        if self.config.minp_ref_dist:
                            pos_tgt_log_prob = ref_log_prob.detach()
                            neg_tgt_log_prob = ref_log_prob.detach()
                            del ref_log_prob
                        else:
                            if not self.config.minp_old_log_prob:
                                minp_pos_tgt_temperature = self.config.minp_pos_tgt_temperature
                                minp_neg_tgt_temperature = self.config.minp_neg_tgt_temperature
                                minp_soft = self.config.minp_soft
                                minp_p_threshold = self.config.minp_p_threshold
                                regular_type = self.config.regular_type
                                _, pos_tgt_log_prob = self._forward_micro_batch(micro_batch=data, temperature=minp_pos_tgt_temperature, calculate_entropy=False, min_p=minp_p_threshold, minp_soft=minp_soft, regular_type=regular_type)
                                pos_tgt_log_prob = pos_tgt_log_prob.detach()
                                _, neg_tgt_log_prob = self._forward_micro_batch(micro_batch=data, temperature=minp_neg_tgt_temperature, calculate_entropy=False, min_p=minp_p_threshold, minp_soft=minp_soft, regular_type=regular_type)
                                neg_tgt_log_prob = neg_tgt_log_prob.detach()
                            else:
                                print("dev_debug: use minp_old_log_probs")
                                minp_old_log_prob = data["minp_old_log_probs"]
                                pos_tgt_log_prob, neg_tgt_log_prob = minp_old_log_prob.detach(), minp_old_log_prob.detach()

                        pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss_lp_reg(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            pos_tgt_log_prob=pos_tgt_log_prob,
                            neg_tgt_log_prob=neg_tgt_log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                            logp_pos_k_percent=self.config.logp_pos_k_percent,
                            logp_neg_k_percent=self.config.logp_neg_k_percent,
                            ppo_kl_coef=self.config.ppo_kl_coef,
                            kl_type=self.config.kl_type,
                        )
                    
                    else:
                        raise ValueError(f"Unsupported loss mode: {self.config.loss_mode}")

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        # "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
    
    def compute_entropy_for_every_source(self, data: DataProto) -> torch.Tensor:
            
            if data.meta_info['train_mode'] ==True:
                self.actor_module.train()
                print("train mode")
            else:
                self.actor_module.eval()
                print("eval mode")
    
            micro_batch_size = data.meta_info['micro_batch_size']
            temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
            use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
            temperature = 1
            
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
            batch = data.select(batch_keys=select_keys).batch
    
            # Split to make minibatch iterator for updating the actor
            # See PPO paper for details. https://arxiv.org/abs/1707.06347
            if use_dynamic_bsz:
                # split using dynamic bsz
                max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            else:
                micro_batches = batch.split(micro_batch_size)
            
            
            # all_entropy_losses = {}
            entropy_lst = []
            entropy_loss_lst = []
            for micro_batch in micro_batches:
                responses = micro_batch['responses']
                response_length = responses.size(1)
                attention_mask = micro_batch['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                with torch.no_grad():
                    entropy, _ = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=True)
                    
                    entropy_loss = verl_F.masked_mean(entropy, response_mask, axis=-1)  # (bsz, response_length)
                entropy_loss_lst.append(entropy_loss)
                entropy_lst.append(entropy)
            entropy_loss_lst = torch.concat(entropy_loss_lst, dim=0)
            entropy_lst = torch.concat(entropy_lst, dim=0)
            
            # print("entropy_lst len,", len(entropy_lst))
            
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == entropy_lst.size(0), f"{len(indices)} vs. {entropy_lst.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                entropy_lst = entropy_lst[revert_indices]
                entropy_loss_lst = entropy_loss_lst[revert_indices]
            
            return entropy_lst, entropy_loss_lst
