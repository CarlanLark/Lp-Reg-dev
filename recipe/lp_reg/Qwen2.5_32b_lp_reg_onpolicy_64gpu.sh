#!/usr/bin/env bash
set -xeuo pipefail

export WANDB_API_KEY=your_wandb_api_key
# export VLLM_USE_V1=1

entity_name="your_wandb_entity"
project_name="your_wandb_project"
exp_name="Qwen2.5_32b_lp_reg_onpolicy_64gpu/$(date +%Y%m%d_%H%M%S)"

adv_estimator=grpo


# core params are minp_p_threshold and logp_neg_k_percent
loss_mode="lp_reg"
kl_type="low_var_kl"
minp_old_log_prob=True
use_clip=True
minp_p_threshold=0.02
logp_pos_k_percent=0
logp_neg_k_percent=0.005
dynamic_coef=1.0



use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=1.0
clip_ratio_high=9.0

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=-1
train_prompt_bsz=256
gen_prompt_bsz=256
train_prompt_mini_bsz=256
n_resp_per_prompt=8
max_token=$((1024 * 30))

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-8} # set your node number here
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"/model_path/Qwen2.5-32B"}
CKPTS_DIR=${CKPTS_DIR:-"/ckpt_dir/$project_name/$exp_name"}
TRAIN_FILE=${TRAIN_FILE:-"/data_path/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-["/data_path/test.parquet"]}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
ppo_kl_coef=1

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False

HYDRA_FULL_ERROR=1 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.minp_old_log_prob=${minp_old_log_prob} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_type=${kl_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_clip=${use_clip} \
    actor_rollout_ref.actor.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.logp_pos_k_percent=${logp_pos_k_percent} \
    actor_rollout_ref.actor.logp_neg_k_percent=${logp_neg_k_percent} \
    actor_rollout_ref.actor.dynamic_coef=${dynamic_coef} \
    actor_rollout_ref.actor.minp_p_threshold=${minp_p_threshold} \
    actor_rollout_ref.actor.ppo_kl_coef=${ppo_kl_coef} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.mode=sync \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_token} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    ++trainer.entity_name="${entity_name}" \
    ++trainer.project_name="${project_name}" \
    ++trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NNODES}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=8 \
    trainer.save_freq=64 \
    trainer.total_epochs=15 \
    trainer.save_train_samples_freq=32 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable