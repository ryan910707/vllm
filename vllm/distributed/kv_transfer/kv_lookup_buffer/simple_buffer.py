# SPDX-License-Identifier: Apache-2.0
"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Push-based transfer: producer immediately sends KV to consumer's buffer
    - Non-blocking `insert`, non-blocking `drop_select`.
    - Use CPU signal pipe to coordinate transfer completion
    - Producer can exit after finishing all requests without waiting
"""
import threading
from collections import deque
import queue
from typing import Deque, List, Optional, Union
import time

from numpy import take_along_axis

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float, 
                 vram_limit_gb: float = 10,
                 role: Optional[str] = None):
        """
        signal_pipe: on CPU
        data_pipe: on device (e.g. GPU)
        buffer_size_thresh: threshold for total buffer size in bytes
        vram_limit_gb: max GB of VRAM to use for storing KV caches
        role: 'producer', 'consumer', or None (backward compatible)
            - 'producer': only start push worker at init (for sending KV)
            - 'consumer': only start receiver handler at init (for receiving KV)
            - None: lazy start on first use (backward compatible)
        
        In push mode:
        - Producer creates threads to push KV to consumer buffer
        - Consumer receives KV in background and stores locally
        - drop_select fetches from local buffer
        
        Dynamic storage strategy:
        - Store tensors on GPU when VRAM usage < vram_limit_gb
        - Store tensors on CPU when VRAM usage >= vram_limit_gb
        """

        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_cv = threading.Condition()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.receive_thread: Optional[threading.Thread] = None
        self.is_consumer = False
        self.is_receiving = False

        # Queue-based push mechanism to prevent race conditions
        self.push_queue: queue.Queue = queue.Queue()
        self.push_worker_thread: Optional[threading.Thread] = None
        self.push_worker_running = False

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None
        
        # GPU VRAM management - simple GB limit
        self.vram_limit_bytes = int(vram_limit_gb * 1024 * 1024 * 1024)
        self.vram_used_by_buffer = 0  # Track VRAM used by our stored tensors
        self._device_available = torch.cuda.is_available()
        
        # Start threads at initialization to avoid cold start based on role
        if role == 'consumer':
            # Consumer only needs receiver handler
            self.start_consumer_mode()
            logger.info("Started SimpleBuffer in consumer mode (receiver handler ready)")
        elif role == 'producer':
            # Producer only needs push worker
            self._start_push_worker()
            logger.info("Started SimpleBuffer in producer mode (push worker ready)")
        elif role is None:
            # Backward compatible: lazy start on first use
            logger.debug("SimpleBuffer initialized with lazy thread startup")
        else:
            raise ValueError(f"Invalid role: {role}. Must be 'producer', 'consumer', or None")

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # Handle device mismatches - move CPU tensor to GPU for comparison
        if tokens_sender.device.type == "cpu" and tokens_recver.device.type == "cuda":
            tokens_sender = tokens_sender.cuda()
        elif tokens_sender.device.type == "cuda" and tokens_recver.device.type == "cpu":
            tokens_recver = tokens_recver.cuda()

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))

        if torch.allclose(tokens_sender[:min_length],
                        tokens_recver[:min_length]):
            return min_length

        return 0

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _should_use_gpu_storage(self, data_size: int) -> bool:
        """Determine if we should store tensors on GPU based on VRAM limit"""
        if not self._device_available:
            return False
        
        # Simple check: will adding this data exceed our VRAM limit?
        projected_vram_usage = self.vram_used_by_buffer + data_size
        
        if projected_vram_usage > self.vram_limit_bytes:
            current_gb = self.vram_used_by_buffer/1024/1024/1024
            adding_gb = data_size/1024/1024/1024
            limit_gb = self.vram_limit_bytes/1024/1024/1024
            logger.debug(f"VRAM limit would be exceeded: "
                        f"current={current_gb:.2f}GB, "
                        f"adding={adding_gb:.2f}GB, "
                        f"limit={limit_gb:.2f}GB. Using CPU storage.")
            return False
        
        return True

    def _get_tensor_device_size(self, tensor: torch.Tensor) -> int:
        """Get the memory size of a tensor on its current device"""
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.numel()

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

        # Calculate total data size for storage planning (single pass)
        buffer_item_temp = [input_tokens, roi, key, value, hidden]
        data_size = sum([self._get_element_size(data) for data in buffer_item_temp])
        
        # Determine optimal storage device based on VRAM usage
        use_gpu_storage = self._should_use_gpu_storage(data_size)
        target_device = "cuda" if use_gpu_storage else "cpu"
        
        # Clone tensors to target device (optimized: single loop)
        buffer_item = []
        for tensor in buffer_item_temp:
            if isinstance(tensor, torch.Tensor):
                buffer_item.append(tensor.clone().to(target_device))
            else:
                buffer_item.append(tensor)
        
        # GPU memory used equals data_size if using GPU storage
        # (avoid redundant size calculation)
        gpu_memory_used = data_size if use_gpu_storage else 0

        with self.buffer_cv:
            self.buffer_size += data_size
            self.vram_used_by_buffer += gpu_memory_used
            self.buffer.append(buffer_item)
            self.buffer_cv.notify()
            
            vram_used_gb = self.vram_used_by_buffer/1024/1024/1024
            vram_limit_gb = self.vram_limit_bytes/1024/1024/1024
            logger.info(f"Stored KV cache on {target_device.upper()}: "
                        f"data_size={data_size/1024/1024:.1f}MB, "
                        f"vram_used={vram_used_gb:.2f}GB/"
                        f"{vram_limit_gb:.1f}GB")

    def _is_end_signal(self, signal):
        return signal is None

    def _start_push_worker(self):
        """Start the push worker thread to handle sequential KV sending"""
        if self.push_worker_thread is not None and self.push_worker_thread.is_alive():
            return  # Worker already running
            
        self.push_worker_running = True
        
        def push_worker():
            """Worker thread that processes the push queue sequentially"""
            logger.debug("Push worker thread started")
            
            while self.push_worker_running:
                try:
                    # Wait for KV data to send (with timeout to check shutdown)
                    kv_data = self.push_queue.get(timeout=1.0)
                    
                    if kv_data is None:  # Shutdown signal
                        break
                        
                    # Send KV data atomically
                    input_tokens, roi, key, value, hidden = kv_data
                    self._send_kv_data_atomic(input_tokens, roi, key, value, hidden)
                    
                    # Mark task as done
                    self.push_queue.task_done()
                    
                except queue.Empty:
                    # Timeout occurred, continue to check shutdown
                    continue
                except Exception as e:
                    logger.error(f"Error in push worker: {e}")
                    # Continue processing other items
                    continue
            
            logger.debug("Push worker thread finished")
        
        self.push_worker_thread = threading.Thread(target=push_worker, daemon=True)
        self.push_worker_thread.start()

    def _send_kv_data_atomic(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                            key: torch.Tensor, value: torch.Tensor, 
                            hidden: torch.Tensor):
        """Atomically send KV data to consumer (called by push worker)"""
        try:
            torch.cuda.nvtx.range_push("send_kv_data_atomic")
            
            # Send signal to indicate incoming KV data
            self.signal_pipe.send_tensor(self.normal_signal)
            
            # Send the KV data atomically
            self.data_pipe.send_tensor(input_tokens)
            self.data_pipe.send_tensor(roi.float() if roi is not None else roi)
            self.data_pipe.send_tensor(key)
            self.data_pipe.send_tensor(value)
            self.data_pipe.send_tensor(hidden)
            
            torch.cuda.nvtx.range_pop()
            logger.debug("Successfully sent KV cache to consumer")
            
        except Exception as e:
            logger.error(f"Error sending KV cache to consumer: {e}")
            raise  # Re-raise so push worker can handle it

    def receive_handler(self):
        """Consumer-side handler to receive pushed KV caches"""
        try:
            torch.cuda.nvtx.range_push("receive_handler")
            logger.debug("Starting receive handler for pushed KV caches")
            
            while self.is_receiving:
                try:
                    # Wait for signal indicating incoming data
                    signal = self.signal_pipe.recv_tensor()
                    if self._is_end_signal(signal):
                        logger.info("Received end signal in receive handler!")
                        break
                    
                    # Receive the KV data (initially to CPU, then _add_to_buffer 
                    # will choose optimal device)
                    input_tokens = self.data_pipe.recv_tensor()
                    roi = self.data_pipe.recv_tensor()
                    if roi is not None:
                        roi = (roi > 0.5)  # Convert back to bool
                    key = self.data_pipe.recv_tensor()
                    value = self.data_pipe.recv_tensor()
                    hidden = self.data_pipe.recv_tensor()
                    
                    # Add to local buffer using dynamic storage strategy
                    # _add_to_buffer will automatically choose GPU vs CPU based on VRAM usage
                    self._add_to_buffer(input_tokens, roi, key, value, hidden)
                    
                except (RuntimeError, torch.distributed.DistNetworkError) as e:
                    if any(msg in str(e) for msg in ['Connection closed by peer', 'Connection reset by peer']):
                        logger.debug("Connection closed by peer, stopping receive handler")
                        break
                    else:
                        logger.error(f"Error in receive handler: {e}")
                        break
                except Exception as e:
                    logger.error(f"Unexpected error in receive handler: {e}")
                    break
                        
            torch.cuda.nvtx.range_pop()
            
        except Exception as e:
            logger.error(f"Fatal error in receive handler: {e}")
        finally:
            logger.debug("Receive handler finished")

    def start_consumer_mode(self):
        """Start consumer mode to receive pushed KV caches"""
        if not self.is_consumer:
            self.is_consumer = True
            self.is_receiving = True
            self.receive_thread = threading.Thread(target=self.receive_handler, daemon=True)
            self.receive_thread.start()
            logger.debug("Started consumer mode with receive handler")

    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        
        torch.cuda.nvtx.range_push("drop_select")
        start_time = time.time()
        
        # Start consumer mode if not already started
        if not self.is_consumer:
            self.start_consumer_mode()

        # Query the local buffer for matching KV cache
        tokens_roi_recver = [
            input_tokens if input_tokens is not None else None,
            roi if roi is not None else None
        ]
        
        def is_buffer_available(tokens_roi_recver: List[torch.Tensor]) -> bool:
            # Early exit if buffer is empty
            if not self.buffer:
                return False
            
            # Fast path: check first element without rotation (common case)
            if self._matches(self.buffer[0], tokens_roi_recver) > 0:
                return True
            
            # Slow path: search remaining elements with rotation
            buffer_len = len(self.buffer)
            for _ in range(1, buffer_len):
                self.buffer.rotate(-1)
                if self._matches(self.buffer[0], tokens_roi_recver) > 0:
                    return True
            
            return False

        with self.buffer_cv:
            while not is_buffer_available(tokens_roi_recver):
                logger.debug("KV transfer buffer is not available. Waiting...")
                torch.cuda.nvtx.range_push("drop_select_wait")
                self.buffer_cv.wait()
                torch.cuda.nvtx.range_pop()
            
            # Get matching item from local buffer
            matched_item = self.buffer.popleft()
            target_device = "cpu"
            # Track VRAM usage before moving tensors
            gpu_memory_to_free = 0
            for tensor in matched_item:
                if tensor is not None and tensor.device.type == "cuda":
                    target_device = "cuda"
                    gpu_memory_to_free += self._get_tensor_device_size(tensor)
            
            # Move matched items to GPU if they're not already there
            # This ensures consumer always gets tensors on GPU for processing
            moved_item = []
            
            for item in matched_item:
                if item is not None:
                    if item.device.type == "cpu":
                        # Move from CPU to GPU for processing (non-blocking for better perf)
                        moved_item.append(item.cuda(non_blocking=True))
                    else:
                        # Already on GPU, just use as-is
                        moved_item.append(item)
                else:
                    moved_item.append(None)
            
            # Update buffer size and VRAM tracking
            data_size_freed = 0
            for tensor in matched_item:
                if tensor is not None:
                    data_size_freed += self._get_element_size(tensor)
            
            self.buffer_size -= data_size_freed
            self.vram_used_by_buffer -= gpu_memory_to_free
            self.buffer_cv.notify()
            
            vram_remaining_gb = self.vram_used_by_buffer/1024/1024/1024
            vram_limit_gb = self.vram_limit_bytes/1024/1024/1024
            logger.info(f"Retrieved KV cache from {target_device.upper()}: freed_size={data_size_freed/1024/1024:.1f}MB, "
                        f"vram_freed={gpu_memory_to_free/1024/1024:.1f}MB, "
                        f"vram_remaining={vram_remaining_gb:.2f}GB/"
                        f"{vram_limit_gb:.1f}GB")
        logger.info(f"Drop_select KV cache time: {time.time() - start_time}")
        torch.cuda.nvtx.range_pop()
        return moved_item

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        """
        Producer-side insert: queue KV for sequential sending to prevent race conditions
        """
        torch.cuda.nvtx.range_push("insert_queue_mode")
        
        # Start push worker if not already running
        if not self.push_worker_running:
            self._start_push_worker()
        
        # Clone tensors to ensure they remain valid when sent
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()
        
        # Queue the KV data for sequential sending by push worker
        kv_data = (input_tokens, roi, key, value, hidden)
        self.push_queue.put(kv_data)
        
        torch.cuda.nvtx.range_pop()
        logger.debug("KV cache queued for sending")

    def signal_end(self):
        """Signal that no more KV caches will be sent"""
        try:
            # Wait for all queued items to be processed
            if self.push_worker_running:
                logger.debug("Waiting for push queue to empty...")
                self.push_queue.join()  # Wait for all items to be processed
                
            self.signal_pipe.send_tensor(self.end_signal)
            logger.debug("Sent end signal to consumer")
        except Exception as e:
            logger.debug(f"Error sending end signal: {e}")

    def close(self):
        """Clean up resources"""
        logger.info("Closing SimpleBuffer")
        # Stop consumer receive thread
        if self.is_consumer and self.is_receiving:
            self.is_receiving = False
            
        if hasattr(self, "receive_thread") and self.receive_thread is not None:
            logger.info("Closing receive thread")
            self.receive_thread.join()
        
        # Stop producer push worker thread
        if self.push_worker_running:
            self.push_worker_running = False
            
            # Send shutdown signal to push worker
            # self.push_queue.put(None)
            
            if hasattr(self, "push_worker_thread") and self.push_worker_thread is not None:
                logger.info("Closing push worker thread")
                self.push_worker_thread.join(timeout=5.0)
                if self.push_worker_thread.is_alive():
                    logger.warning("Push worker thread did not shutdown cleanly")
        
        # Clear buffer and reset VRAM tracking
        with self.buffer_cv:
            self.buffer.clear()
            self.buffer_size = 0
            self.vram_used_by_buffer = 0
            logger.debug("Cleared buffer and reset VRAM tracking")
            
        # If this is a producer, signal end to consumer
        # if not self.is_consumer:
        #     self.signal_end()
