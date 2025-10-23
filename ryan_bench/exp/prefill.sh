KV_BUFFER_SIZE=$((36000*(128+5)))
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NET_GDR_LEVEL=0 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=enp1s0f0 \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --port 8100 \
  --gpu-memory-utilization 0.7 \
  --dtype "half" \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":'${KV_BUFFER_SIZE}',"kv_ip":"10.121.187.102"}'

