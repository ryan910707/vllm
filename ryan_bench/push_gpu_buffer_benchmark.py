# SPDX-License-Identifier: Apache-2.0
"""
Benchmark against insufficient_gpu_buffer_benchmark.py.
In this benchmark, we will use push-based KV transfer.
Prefill worker immediately sends KV caches to decode worker's buffer,
allowing prefill worker to exit after completing all requests.
"""
import argparse
import os
import logging
import random
import time
from multiprocessing import Event, Process

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure logging with timestamps including milliseconds
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Push GPU Buffer Benchmark")
    parser.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL"),
                        help="Model name or path (default: $VLLM_MODEL)")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts (default: 10)")
    parser.add_argument("--prompt-length", type=int, default=16,
                        help="Target prompt length in characters (default: 16)")
    parser.add_argument("--output-len", type=int, default=96,
                        help="Number of output tokens (default: 96)")
    parser.add_argument("--buffer-size", type=int, default=160000*(256+1),
                        help="KV transfer buffer size (default: 160000*257, not used in cpu_push_mode)")
    parser.add_argument("--qps", type=float, default=12.0,
                        help="Target queries per second, 0 = unlimited (default: 12.0)")
    return parser.parse_args()

args = parse_args()
MODEL = args.model
NUM_PROMPTS = args.num_prompts
PROMPT_LENGTH = args.prompt_length
OUTPUT_LEN = args.output_len
BUFFER_SIZE = args.buffer_size
QPS = args.qps

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
    llm = LLM(model=MODEL,
              kv_transfer_config=ktc,
              max_model_len=2048,
              dtype="half",
              gpu_memory_utilization=0.95)
    logger.info("Prefill model initialized")

    logger.info("Starting prefill generation loop")
    start_time = time.time()
    
    for i, prompt_text in enumerate(prompts):
        # Calculate when this request should be sent based on QPS
        if QPS > 0:
            target_time = start_time + (i / QPS)
            current_time = time.time()
            sleep_time = target_time - current_time
            
            if sleep_time > 0:
                logger.info(f"Rate limiting: sleeping {sleep_time:.3f}s to maintain {QPS} QPS")
                time.sleep(sleep_time)
        
        request_start = time.time()
        logger.info(f"Processing prefill prompt {i}")
        llm.generate([prompt_text], sampling_params, use_tqdm=False)
        request_duration = time.time() - request_start
        logger.info(f"Completed prefill prompt {i} in {request_duration:.3f}s")
    
    total_duration = time.time() - start_time
    actual_qps = NUM_PROMPTS / total_duration if total_duration > 0 else 0
    logger.info(f"Prefill generation loop completed in {total_duration:.2f}s")
    logger.info(f"Prefill actual QPS: {actual_qps:.2f} (target: {QPS if QPS > 0 else 'unlimited'})")
    
    logger.info("Prefill node is finished with all prompts")
    
    # Signal that prefill is done
    prefill_done_event.set()
    
    # Wait for decode to signal it's completely done
    logger.info("Prefill node waiting for decode to finish...")
    decode_done_event.wait()
    logger.info("Prefill node exiting cleanly")
    os._exit(0)


def run_decode(prefill_done_event, decode_done_event):
    logger.info("Decode node starting")
    
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, min_tokens=OUTPUT_LEN, max_tokens=OUTPUT_LEN+1)

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
    llm = LLM(model=MODEL,
              kv_transfer_config=ktc,
              max_model_len=2048,
              dtype= "half",
              gpu_memory_utilization=0.95)
    logger.info("Decode model initialized")

    logger.info("Starting decode generation loop")
    all_outputs = []
    start_time = time.time()
    latencies = []
    
    try:
        for i, prompt_text in enumerate(prompts):
            # Calculate when this request should be sent based on QPS
            if QPS > 0:
                target_time = start_time + (i / QPS)
                current_time = time.time()
                sleep_time = target_time - current_time
                
                if sleep_time > 0:
                    logger.info(f"Rate limiting: sleeping {sleep_time:.3f}s to maintain {QPS} QPS")
                    time.sleep(sleep_time)
            
            # At this point the kv-cache for this specific prompt should have been transferred
            # (pushed by the prefill node to our local buffer).
            request_start = time.time()
            logger.info(f"Processing decode prompt {i}")
            outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
            request_duration = time.time() - request_start
            latencies.append(request_duration)
            logger.info(f"Completed decode prompt {i} in {request_duration:.3f}s")

            all_outputs.extend(outputs)
    except Exception as e:
        logger.error(f"Error during decode generation: {e}")
        # Continue with whatever outputs we have
    
    total_duration = time.time() - start_time
    actual_qps = NUM_PROMPTS / total_duration if total_duration > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    logger.info("Decode generation loop completed")
    logger.info(f"Decode total time: {total_duration:.2f}s")
    logger.info(f"Decode actual QPS: {actual_qps:.2f} (target: {QPS if QPS > 0 else 'unlimited'})")
    logger.info(f"Decode latency - avg: {avg_latency:.3f}s, min: {min_latency:.3f}s, max: {max_latency:.3f}s")

    # logger.info("--- Decode Node: Final Outputs ---")
    # for output in all_outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     logger.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    logger.info("Decode node finished processing")
    
    # Signal that decode is completely done
    decode_done_event.set()
    logger.info("Decode signaled completion to prefill")
    os._exit(0)


if __name__ == "__main__":
    logger.info("=== Push GPU Buffer Benchmark ===")
    logger.info(f"Model (VLLM_MODEL): {MODEL}")
    logger.info(f"CONFIG num_prompts={NUM_PROMPTS} prompt_length={PROMPT_LENGTH} output_len={OUTPUT_LEN} qps={QPS} buffer_size={BUFFER_SIZE}")
    logger.info(f"Buffer size: {BUFFER_SIZE}")
    # logger.info("Generated prompts:")
    # for i, prompt in enumerate(prompts):
    #     logger.info(f"  {i+1}: {prompt[:50]}... (len: {len(prompt)})")
    
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
    # logger.info("Prefill process completed")
    
    # # Then wait for decode process to finish
    # logger.info("Waiting for decode process to complete...")
    decode_process.join()
    # logger.info("Decode process completed")
    
    # logger.info("Benchmark completed successfully!")
