# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and then transfer the KV cache between them.
"""
import os
import time
from multiprocessing import Event, Process

import torch

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


prompts = [
        "Hello, my name is",
        "Hi, your name is",
        # The decode node will actually "prefill" this request.
        "Tell me a very long story",
    ]

def run_prefill(prefill_done):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # The prefill node receives two requests, while the decode node receives
    # three requests. So the decode node will only receive the KV Cache for
    # requests 1 and 3. The decode node will use the KV Cache of requests 1
    # and 3 and do prefilling on request 2.
    
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=2000,
              dtype="half",
              gpu_memory_utilization=0.8)

    for i, prompt_text in enumerate(prompts):
        torch.cuda.nvtx.range_push(f"prefill {i}")
        llm.generate([prompt_text], sampling_params) # Pass a list with a single prompt
        torch.cuda.nvtx.range_pop()
    
    print("Prefill node is finished with all prompts.")

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(prefill_done):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=2000,
              dtype= "half",
              gpu_memory_utilization=0.8)

    all_outputs = []
    for i, prompt_text in enumerate(prompts):

        # At this point the kv-cache for this specific prompt should have been transferred
        # (as part of the overall KV cache transfer completed by the prefill node).
        torch.cuda.nvtx.range_push(f"decode {i}")
        outputs = llm.generate([prompt_text], sampling_params) # Pass a list with a single prompt
        torch.cuda.nvtx.range_pop()

        all_outputs.extend(outputs)


    print("\n--- Decode Node: Final Outputs ---")
    for output in all_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, ))
    decode_process = Process(target=run_decode, args=(prefill_done, ))

    # Start prefill node
    decode_process.start()
    prefill_process.start()

    # Start decode node


    # Terminate the prefill node when decode is finished
    decode_process.join()
    prefill_process.terminate()
