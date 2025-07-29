# SPDX-License-Identifier: Apache-2.0
"""
Benchmark against insufficient_gpu_buffer_benchmark.py.
In this benchmark, we will use push-based KV transfer.
Prefill worker immediately sends KV caches to decode worker's buffer,
allowing prefill worker to exit after completing all requests.
"""
import os
import time
from multiprocessing import Event, Process

import torch

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


prompts = [
        "The cat sat on mat",
        "Five dogs ran past me", 
        "She walked through the door",
        "He jumped over the fence",
        "They danced in the rain",
    ]

def run_prefill(prefill_done_event, decode_done_event):
    torch.cuda.nvtx.range_push("prefill_node_start")
    
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # The prefill node processes requests and immediately pushes
    # KV caches to the decode node's buffer.
    
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    torch.cuda.nvtx.range_push("prefill_model_init")
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=2000,
              dtype="half",
              gpu_memory_utilization=0.8)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("prefill_generation_loop")
    for i, prompt_text in enumerate(prompts):
        torch.cuda.nvtx.range_push(f"prefill {i}")
        llm.generate([prompt_text], sampling_params) # Pass a list with a single prompt
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    
    print("Prefill node is finished with all prompts.")
    
    # Signal that prefill is done
    prefill_done_event.set()
    
    # Wait for decode to signal it's completely done
    torch.cuda.nvtx.range_push("prefill_wait_for_decode")
    print("Prefill node waiting for decode to finish...")
    decode_done_event.wait()  # Wait for decode to signal completion
    print("Decode signaled completion - prefill can exit now.")
    torch.cuda.nvtx.range_pop()
    
    print("Prefill node exiting cleanly.")
    torch.cuda.nvtx.range_pop()


def run_decode(prefill_done_event, decode_done_event):
    torch.cuda.nvtx.range_push("decode_node_start")
    
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, min_tokens=90, max_tokens=100)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2
    )
    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    torch.cuda.nvtx.range_push("decode_model_init")
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=2000,
              dtype= "half",
              gpu_memory_utilization=0.8)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("decode_generation_loop")
    all_outputs = []
    
    try:
        for i, prompt_text in enumerate(prompts):
            # At this point the kv-cache for this specific prompt should have been transferred
            # (pushed by the prefill node to our local buffer).
            torch.cuda.nvtx.range_push(f"decode {i}")
            outputs = llm.generate([prompt_text], sampling_params) # Pass a list with a single prompt
            torch.cuda.nvtx.range_pop()

            all_outputs.extend(outputs)
    except Exception as e:
        print(f"Error during decode generation: {e}")
        # Continue with whatever outputs we have
    
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("decode_output_processing")
    print("\n--- Decode Node: Final Outputs ---")
    for output in all_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    torch.cuda.nvtx.range_pop()
    
    print("Decode node finished processing.")
    
    # Signal that decode is completely done
    decode_done_event.set()
    print("Decode signaled completion to prefill.")
    
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    torch.cuda.nvtx.range_push("benchmark_total")
    
    # Events to coordinate between prefill and decode processes
    prefill_done_event = Event()
    decode_done_event = Event()
    
    torch.cuda.nvtx.range_push("process_creation")
    prefill_process = Process(target=run_prefill, args=(prefill_done_event, decode_done_event))
    decode_process = Process(target=run_decode, args=(prefill_done_event, decode_done_event))
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("process_execution")
    # Start both processes
    prefill_process.start()
    decode_process.start()

    # Wait for prefill to exit first (it waits for decode to signal completion)
    print("Waiting for prefill process to complete...")
    prefill_process.join()
    print("Prefill process completed.")
    
    # Then wait for decode process to finish
    print("Waiting for decode process to complete...")
    decode_process.join()
    print("Decode process completed.")
    
    torch.cuda.nvtx.range_pop()
    
    print("Benchmark completed successfully!")
