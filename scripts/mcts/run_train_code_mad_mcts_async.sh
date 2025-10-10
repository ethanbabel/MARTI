#!/bin/bash
set -x
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --include-dashboard=true
sleep 3

MODEL_DIR="/mnt/public/hf_models/Qwen/"
WANDB_KEY="51a11432da43d6d76b237f983cc99bcd0655b683"
ROOT_DIR=$(pwd)

# DATE=$(date +%m%d)
DATE=$(date +%m%d_%H%M%S)
ADVANTAGE="reinforce"
INFER_METHOD="ab-mcts"
# SHORT_NAME="Qwen2.5-3B"
SHORT_NAME="${1:-Qwen3-4B-Thinking-2507}"
CONFIG_NAME="${2:-ma1_ab_mcts}"
MCTS_NODES=${3:-8}
# TASK="CODE"
TASK=${4:-deepcoder}
MICRO_TRAIN_BS=${5:-2}
MICRO_ROLLOUT_BS=${6:-4}
TRAIN_BS=${7:-256}
ROLLOUT_BS=${8:-64}
NUM_WORKERS=${9:-16}
N_SAMPLE_PER_PROMPT=${10:-1}
WORK_FLOW_NAME="${SHORT_NAME}-async-bs${TRAIN_BS}-minibs${MICRO_TRAIN_BS}-rollbs${ROLLOUT_BS}-minirollbs${MICRO_ROLLOUT_BS}"
ALGO="ma1-ab_mcts"
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
# EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"
EXP="${DATE}-${TASK}-${CONFIG_NAME}-${ADVANTAGE}-${ALGO}-${WORK_FLOW_NAME}"

# SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${TASK}/${SHORT_NAME}/model"
# WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${TASK}/${SHORT_NAME}/workflow"
SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${TASK}/${CONFIG_NAME}/${WORK_FLOW_NAME}/model"
WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${TASK}/${CONFIG_NAME}/${WORK_FLOW_NAME}/workflow"

PROMPT_DATA="json@${ROOT_DIR}/local/${TASK}"
# TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"
# CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${TASK}/${SHORT_NAME}/ckpt"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${CONFIG_NAME}-${WORK_FLOW_NAME}"
CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${TASK}/${CONFIG_NAME}/${WORK_FLOW_NAME}/ckpt"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/logs/std"
mkdir -p "${ROOT_DIR}/logs/ab_mcts_async/${CONFIG_NAME}"
mkdir -p "${ROOT_DIR}/logs/tensorboard"
mkdir -p "${ROOT_DIR}/outputs"

# PROMPT_MAX_LEN=12288
# GENERATE_MAX_LEN=4096
PROMPT_MAX_LEN=2048
GENERATE_MAX_LEN=16384
LOG_PATH="${ROOT_DIR}/logs/ab_mcts_async/${CONFIG_NAME}/${DATE}-${EXP}.log"

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export RAY_PICKLE_VERBOSE_DEBUG=1
# export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1

ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["/data/", "/outputs/", "/eval_outputs/", ".git/", "local/", "/logs/", "/eval_logs/"],
  "pip": ["hydra-core", "antlr4-python3-runtime==4.9.3", "shortuuid", "class_registry", "json5", "mcp[cli]"]
}
EOF
)
# agent_func_path="marti/worlds/tools/code_step.py" \
ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train --config-name ${CONFIG_NAME} \
    async_workflow=True \
    parallel_loading=True \
    default_agent.is_reasoning_model=True \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=8 \
    default_agent.critic_num_nodes=1 \
    default_agent.critic_num_gpus_per_node=8 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=8 \
    default_agent.vllm_num_engines=8 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.colocate_all_models=True \
    default_agent.vllm_enable_sleep=True \
    default_agent.deepspeed_enable_sleep=True \
    default_agent.vllm_gpu_memory_utilization=0.9 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.micro_train_batch_size=${MICRO_TRAIN_BS} \
    default_agent.train_batch_size=${TRAIN_BS} \
    default_agent.num_episodes=1 \
    default_agent.save_steps=5 \
    default_agent.eval_steps=2 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=400000 \
    default_agent.micro_rollout_batch_size=${MICRO_ROLLOUT_BS} \
    default_agent.rollout_batch_size=${ROLLOUT_BS} \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=${N_SAMPLE_PER_PROMPT} \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.temperature=1.0 \
    default_agent.top_p=0.95 \
    default_agent.eval_temperature=1.0 \
    default_agent.lambd=1.0 \
    default_agent.gamma=1.0 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.00 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=100 \
    default_agent.normalize_reward=True \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    default_agent.load_checkpoint=True \
    default_agent.enable_thinking=True \
    default_agent.clip_eps_high=0.28 \
    workflow_args.max_num_nodes=${MCTS_NODES} \
    workflow_args.eval_max_num_nodes=1 \
    workflow_args.save_path="${WORKFLOW_SAVE_PATH}" \
    workflow_args.algo.class_name="AsyncABMCTSA" \
    workflow_func_path="marti/worlds/workflows/ab_mcts_workflow.py" \
    processor_func_path="marti/worlds/workflows/ab_mcts_processor.py" \
    tools_config.num_workers=${NUM_WORKERS} \
    filter_agents_data=False \
    verify_task='code' \
    verify_task_eval='code' \
    reward_alloc.name="margin" \
    reward_alloc.alpha=0.5 \
    reward_alloc.beta=0.5 \
    reward_alloc.use_ttrl=False \
    eval_before_training=True \
    eval_only=True \
    eval_workers=-1 \
    mask_truncated_completions=True \
    shared_agents=False \
    packing_samples=True \
    prompt_data="${PROMPT_DATA}" \
    input_key="prompt" \
    label_key="label" \
    metadata_key="extra_info" \
    add_prompt_suffix=null \
    wandb_project="MARTI" \
    wandb_run_name="${EXP}" \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee ${LOG_PATH}
    # use_wandb=${WANDB_KEY} \
    # add_prompt_suffix=null \
    # add_prompt_suffix="\n\nPlease reason step by step, and put your final answer within \\boxed{}." \
    # +add_system_prompt="You are an expert mathematician solving math problem. These are challenging mathematics problems that require creative thinking and advanced problem-solving techniques." \
    # use_wandb="${WANDB_KEY}" \

echo "Model Training Finished. Shutting down..."