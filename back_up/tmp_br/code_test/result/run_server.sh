export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export ASCEND_RT_VISIBLE_DEVICES=10,11

# export VLLM_LOGGING_LEVEL=DEBUG

export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

# export ASCEND_ENABLE_USE_FABRIC_MEM=1
# export MOONCAKE_MASTER="90.90.97.4:50061"
export MOONCAKE_CONFIG_PATH="/home/lizhongyang/FaultPatternLibrary/mooncake.json"

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export TASK_QUEUE_ENABLE=1

# export ASCEND_CONNECT_TIMEOUT=1
# export ASCEND_TRANSFER_TIMEOUT=1

MODEL_PATH="/mnt/weights/Qwen3-8B"

vllm serve ${MODEL_PATH} \
  --served-model-name Qwen3-8B \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --max-num-seqs 64 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "topic": "kv-events", "endpoint": "tcp://*:5555"}' \
  --kv-transfer-config \
    '{
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
        "backend": "mooncake",
        "lookup_rpc_port":"0",
        "use_layerwise": true
        }
    }'