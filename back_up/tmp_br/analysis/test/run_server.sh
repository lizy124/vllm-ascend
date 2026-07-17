#!/bin/bash
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_INTRA_ROCE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export TASK_QUEUE_ENABLE=1
export HCCL_SO_PATH="/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libhccl.so"

MODEL_PATH="/mnt/weight/weight/qwen/Qwen3-30B-A3B"

vllm serve ${MODEL_PATH} \
  --served-model-name Qwen3-30B-A3B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --async-scheduling \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 2768 \
  --host 0.0.0.0 \
  --port 8000