CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NET_GDR_LEVEL=0 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=enp1s0f0 \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8200 \
  --gpu-memory-utilization 0.7 \
  --dtype "half" \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":1e9,"kv_ip":"10.121.187.102"}'
  

# sample request
# curl -sS -X POST http://127.0.0.1:8000/v1/completions   -H "Content-Type: application/json"   -d '{
#     "model": "Qwen/Qwen2.5-1.5B-Instruct",
#     "prompt": "Explain disaggregated prefill in 2 sentences.",
#     "max_tokens": 32,
#     "temperature": 0.0
#   }'
