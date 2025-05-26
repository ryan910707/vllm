"""
This file demonstrates the example usage of disaggregated prefilling across different machines.
We will launch 2 vllm instances:
- Machine 1: GPU 0 for prefill (producer)
- Machine 2: GPU 0 for decode (consumer)
The KV cache will be transferred between them over the network.
"""
import os
import sys
import time
import argparse
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import torch.distributed as dist


def run_prefill(args):
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Configure KV transfer for the prefill node (producer)
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_ip=args.ip,  # IP address of the prefill machine
        kv_port=args.port
    )

    # Initialize the LLM
    llm = LLM(
        model=args.model,
        kv_transfer_config=ktc,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="half" # for V100 only
    )

    print("Prefill task is starting...")
    llm.generate(prompts, sampling_params)
    print("Prefill task is finished.")

    # Keep the prefill node running
    time.sleep(3)
   
    dist.destroy_process_group()


def run_decode(args):
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    # Configure KV transfer for the decode node (consumer)
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_ip=args.ip,  # IP address of the prefill machine
        kv_port=args.port
    )

    # Initialize the LLM
    llm = LLM(
        model=args.model,
        kv_transfer_config=ktc,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="half"
    )

    print("Decode node is starting...")
    outputs = llm.generate(prompts, sampling_params)
    print("Decode task is completed...")

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # print("Waiting briefly for signal propagation...")
    time.sleep(3)

    from vllm.distributed import parallel_state
    if getattr(parallel_state, "_KV_TRANSFER", None) is not None:
        print("[test.py] Closing _KV_TRANSFER agent (decode).")
        parallel_state._KV_TRANSFER.close()
    dist.destroy_process_group()
    

def main():
    parser = argparse.ArgumentParser(description="Run distributed prefill/decode")
    parser.add_argument("--mode", choices=["prefill", "decode"], required=True,
                      help="Whether to run as prefill or decode node")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf",
                      help="Model to use")
    parser.add_argument("--gpu-id", default="0",
                      help="GPU ID to use")
    parser.add_argument("--ip", required=True,
                      help="IP address for KV transfer (use producer's IP)")
    parser.add_argument("--port", type=int, default=14579,
                      help="Port for KV transfer")
    parser.add_argument("--max-model-len", type=int, default=2048,
                      help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                      help="GPU memory utilization")

    args = parser.parse_args()

    if args.mode == "prefill":
        run_prefill(args)
    else:
        run_decode(args)

if __name__ == "__main__":
    main()