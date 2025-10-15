# SPDX-License-Identifier: Apache-2.0
"""
Benchmark against insufficient_gpu_buffer_benchmark.py.
In this benchmark, we will use push-based KV transfer.
Prefill worker immediately sends KV caches to decode worker's buffer,
allowing prefill worker to exit after completing all requests.
"""
import os
import logging
import random
from multiprocessing import Event, Process

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration - modify these values directly
NUM_PROMPTS = 10
PROMPT_LENGTH = 128  # target character length
BUFFER_SIZE = 8930*(128+5)

# Simple word list for generating prompts
WORDS = ["the", "cat", "dog", "tree", "house", "car", "sun", "moon", "water", 
         "fire", "mountain", "ocean", "flower", "stone", "cloud", "light", 
         "fast", "slow", "happy", "bright", "walk", "run", "jump", "dance", 
         "think", "create"]

def generate_prompts():
    """Generate prompts with exact target length"""
    random.seed(42)  # for reproducible results
    prompts = []
    for i in range(NUM_PROMPTS):
        prompt = f"Story {i+1}: "
        
        # Keep adding words until we're close to target length
        while len(prompt) < PROMPT_LENGTH - 1:
            word = random.choice(WORDS)
            if len(prompt) + len(word) + 1 <= PROMPT_LENGTH:
                prompt += word + " "
            else:
                break
        
        # Pad or trim to exact length
        prompt = prompt.strip()
        if len(prompt) < PROMPT_LENGTH:
            prompt += "." + "x" * (PROMPT_LENGTH - len(prompt) - 1)
        elif len(prompt) > PROMPT_LENGTH:
            prompt = prompt[:PROMPT_LENGTH-1] + "."
        else:
            prompt = prompt[:-1] + "."  # replace last char with period
            
        prompts.append(prompt)
    return prompts

prompts = generate_prompts()

def run_prefill(prefill_done_event, decode_done_event):
    logger.info("Prefill node starting")
    
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
        kv_parallel_size=2,
        kv_buffer_size=BUFFER_SIZE, 
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    logger.info("Initializing prefill model")
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              kv_transfer_config=ktc,
              dtype="half",
              gpu_memory_utilization=0.7)
    logger.info("Prefill model initialized")

    logger.info("Starting prefill generation loop")
    for i, prompt_text in enumerate(prompts):
        logger.info(f"Processing prefill prompt {i}")
        llm.generate([prompt_text], sampling_params) # Pass a list with a single prompt
        logger.info(f"Completed prefill prompt {i}")
    logger.info("Prefill generation loop completed")
    
    logger.info("Prefill node is finished with all prompts")
    
    # Signal that prefill is done
    prefill_done_event.set()
    
    # Wait for decode to signal it's completely done
    logger.info("Prefill node waiting for decode to finish...")
    decode_done_event.wait()  # Wait for decode to signal completion
    logger.info("Decode signaled completion - prefill can exit now")
    
    logger.info("Prefill node exiting cleanly")


def run_decode(prefill_done_event, decode_done_event):
    logger.info("Decode node starting")
    
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, min_tokens=128, max_tokens=129)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_buffer_size=BUFFER_SIZE, 
    )
    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    logger.info("Initializing decode model")
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              kv_transfer_config=ktc,
              dtype= "half",
              gpu_memory_utilization=0.7)
    logger.info("Decode model initialized")

    logger.info("Starting decode generation loop")
    all_outputs = []
    
    try:
        for i, prompt_text in enumerate(prompts):
            # At this point the kv-cache for this specific prompt should have been transferred
            # (pushed by the prefill node to our local buffer).
            logger.info(f"Processing decode prompt {i}")
            outputs = llm.generate([prompt_text], sampling_params) # Pass a list with a single prompt
            logger.info(f"Completed decode prompt {i}")

            all_outputs.extend(outputs)
    except Exception as e:
        logger.error(f"Error during decode generation: {e}")
        # Continue with whatever outputs we have
    
    logger.info("Decode generation loop completed")

    logger.info("--- Decode Node: Final Outputs ---")
    for output in all_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logger.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    logger.info("Decode node finished processing")
    
    # Signal that decode is completely done
    decode_done_event.set()
    logger.info("Decode signaled completion to prefill")


if __name__ == "__main__":
    logger.info("=== Push GPU Buffer Benchmark ===")
    logger.info(f"Config: {NUM_PROMPTS} prompts, length {PROMPT_LENGTH} chars")
    logger.info("Generated prompts:")
    for i, prompt in enumerate(prompts):
        logger.info(f"  {i+1}: {prompt[:50]}... (len: {len(prompt)})")
    
    logger.info("Starting benchmark")
    
    # Events to coordinate between prefill and decode processes
    prefill_done_event = Event()
    decode_done_event = Event()
    
    logger.info("Creating processes")
    prefill_process = Process(target=run_prefill, args=(prefill_done_event, decode_done_event))
    decode_process = Process(target=run_decode, args=(prefill_done_event, decode_done_event))

    logger.info("Starting processes")
    # Start both processes
    prefill_process.start()
    decode_process.start()

    # Wait for prefill to exit first (it waits for decode to signal completion)
    logger.info("Waiting for prefill process to complete...")
    prefill_process.join()
    logger.info("Prefill process completed")
    
    # Then wait for decode process to finish
    logger.info("Waiting for decode process to complete...")
    decode_process.join()
    logger.info("Decode process completed")
    
    logger.info("Benchmark completed successfully!")
