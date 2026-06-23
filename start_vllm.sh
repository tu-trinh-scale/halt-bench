#!/usr/bin/env bash
# Start the vLLM inference server used by the ask_human oracle.
#
# Run this in a separate terminal (or tmux pane) before running run_halt_bench.py
# with the default --ask-human-backend vllm.
#
# Requirements:
#   - vLLM installed and available on PATH (or activate the appropriate venv first)
#   - At least one GPU; adjust --tensor-parallel-size and --gpu-memory-utilization
#     for your hardware
#
# The defaults below match the values in .env.example:
#   VLLM_BASE_URL=http://localhost:8808/v1
#   ASK_HUMAN_MODEL=casperhansen/llama-3.3-70b-instruct-awq

MODEL="${ASK_HUMAN_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
PORT="${VLLM_PORT:-8808}"
GPU_MEM="${VLLM_GPU_MEM:-0.7}"
TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-4}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

echo "Starting vLLM server: model=$MODEL  port=$PORT  tensor_parallel=$TENSOR_PARALLEL"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --pipeline-parallel-size 1 \
  --enforce-eager
