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
import time
import queue
from typing import Deque, List, Optional, Union, Dict, Tuple

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float):
        """
        signal_pipe: on CPU
        data_pipe: on device (e.g. GPU)
        
        In push mode:
        - Producer creates threads to push KV to consumer buffer
        - Consumer receives KV in background and stores locally
        - drop_select fetches from local buffer
        
        Enhanced for layer-wise transfer:
        - Buffer can store partial layer data and reconstruct complete entries
        - Proper synchronization ensures layer ordering
        """

        # Original buffer for complete entries (backward compatibility)
        self.buffer: Deque[List[torch.Tensor]] = deque()
        
        # Layer-wise buffer: Dict[request_id, Dict[layer_id, layer_data]]
        # request_id is derived from input_tokens hash
        # layer_data: [input_tokens, roi, key, value, hidden, layer_id, total_layers]
        self.layer_buffer: Dict[str, Dict[int, List[torch.Tensor]]] = {}
        self.layer_buffer_cv = threading.Condition()

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
        self.layer_signal = torch.tensor([1], device="cpu")  # New signal for layer-wise data
        self.end_signal = None

    def _get_request_id(self, input_tokens: torch.Tensor) -> str:
        """Generate a unique request ID from input tokens"""
        if input_tokens is None:
            return "none"
        # Use hash of tensor content as request ID
        return str(hash(input_tokens.cpu().detach().numpy().tobytes()))
    
    def _is_layer_complete(self, request_id: str) -> bool:
        """Check if all layers for a request have been received"""
        if request_id not in self.layer_buffer:
            return False
        
        layer_data = self.layer_buffer[request_id]
        if not layer_data:
            return False
            
        # Get total layers from any layer entry (they should all be the same)
        any_layer = next(iter(layer_data.values()))
        total_layers = any_layer[6].item()  # total_layers is at index 6
        
        return len(layer_data) == total_layers
    
    def _reconstruct_complete_entry(self, request_id: str) -> List[torch.Tensor]:
        """Reconstruct complete KV entry from layer-wise data"""
        if request_id not in self.layer_buffer:
            return None
            
        layer_data = self.layer_buffer[request_id]
        if not layer_data:
            return None
        
        # Sort layers by layer_id
        sorted_layers = sorted(layer_data.items(), key=lambda x: x[0])
        
        # Extract data from first layer for metadata
        first_layer = sorted_layers[0][1]
        input_tokens = first_layer[0]
        roi = first_layer[1]
        
        # Find hidden states from the final layer (should be non-empty)
        hidden = None
        for layer_id, layer_entry in sorted_layers:
            layer_hidden = layer_entry[4]
            if layer_hidden is not None and layer_hidden.numel() > 0:
                hidden = layer_hidden
                break
        # Fallback to last layer if none found
        if hidden is None:
            hidden = sorted_layers[-1][1][4]
        
        # Collect all keys and values
        keys = []
        values = []
        
        for layer_id, layer_entry in sorted_layers:
            keys.append(layer_entry[2])    # key
            values.append(layer_entry[3])  # value
        
        # Concatenate layers: [num_layers, ...] 
        keys = torch.cat(keys, dim=0)
        values = torch.cat(values, dim=0)
        
        return [input_tokens, roi, keys, values, hidden]

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

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

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

        buffer_item = [input_tokens, roi, key, value, hidden]
        data_size = sum([self._get_element_size(data) for data in buffer_item])

        with self.buffer_cv:
            if self.buffer_size + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                torch.cuda.nvtx.range_push("KV transfer buffer wait")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.buffer_cv.wait()
                torch.cuda.nvtx.range_pop()
                logger.debug(f"KV transfer buffer wait end")

            self.buffer_size += data_size
            self.buffer.append(buffer_item)
            self.buffer_cv.notify()
    
    def _add_layer_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                            key: torch.Tensor, value: torch.Tensor,
                            hidden: torch.Tensor, layer_id: int, total_layers: int):
        """Add a single layer to the layer-wise buffer"""
        
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

        layer_item = [input_tokens, roi, key, value, hidden, 
                     torch.tensor(layer_id), torch.tensor(total_layers)]
        data_size = sum([self._get_element_size(data) for data in layer_item])
        
        request_id = self._get_request_id(input_tokens)

        with self.layer_buffer_cv:
            # Wait if buffer is too full
            if self.buffer_size + data_size > self.buffer_size_threshold:
                logger.debug("Layer-wise KV transfer buffer is full. Handling...")
                torch.cuda.nvtx.range_push("layer_buffer_wait")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.layer_buffer_cv.wait()
                torch.cuda.nvtx.range_pop()
                logger.debug("Layer-wise KV transfer buffer wait end")

            # Initialize request entry if not exists
            if request_id not in self.layer_buffer:
                self.layer_buffer[request_id] = {}
            
            # Add layer data
            self.layer_buffer[request_id][layer_id] = layer_item
            self.buffer_size += data_size
            
            # Check if request is complete and move to main buffer
            if self._is_layer_complete(request_id):
                complete_entry = self._reconstruct_complete_entry(request_id)
                if complete_entry is not None:
                    self.buffer.append(complete_entry)
                    # Remove from layer buffer to free memory
                    del self.layer_buffer[request_id]
                    logger.debug(f"Completed layer-wise assembly for request {request_id[:8]}")
            
            self.layer_buffer_cv.notify_all()
            # Also notify main buffer waiters
            with self.buffer_cv:
                self.buffer_cv.notify_all()

    def _is_end_signal(self, signal):
        return signal is None
    
    def _is_layer_signal(self, signal):
        """Check if signal indicates layer-wise data"""
        if signal is None:
            return False
        return torch.allclose(signal, self.layer_signal)

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
                    if len(kv_data) == 5:
                        # Regular complete data
                        input_tokens, roi, key, value, hidden = kv_data
                        self._send_kv_data_atomic(input_tokens, roi, key, value, hidden)
                    elif len(kv_data) == 7:
                        # Layer-wise data
                        input_tokens, roi, key, value, hidden, layer_id, total_layers = kv_data
                        self._send_layer_data_atomic(input_tokens, roi, key, value, hidden, layer_id, total_layers)
                    
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

    def _send_layer_data_atomic(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                               key: torch.Tensor, value: torch.Tensor, 
                               hidden: torch.Tensor, layer_id: int, total_layers: int):
        """Atomically send layer-wise KV data to consumer (called by push worker)"""
        try:
            torch.cuda.nvtx.range_push("send_layer_data_atomic")
            
            # Send signal to indicate incoming layer-wise data
            self.signal_pipe.send_tensor(self.layer_signal)
            
            # Send the layer KV data atomically
            self.data_pipe.send_tensor(input_tokens)
            self.data_pipe.send_tensor(roi.float() if roi is not None else roi)
            self.data_pipe.send_tensor(key)
            self.data_pipe.send_tensor(value)
            self.data_pipe.send_tensor(hidden)
            self.data_pipe.send_tensor(torch.tensor(layer_id, device=key.device))
            self.data_pipe.send_tensor(torch.tensor(total_layers, device=key.device))
            
            torch.cuda.nvtx.range_pop()
            logger.debug(f"Successfully sent layer {layer_id} KV cache to consumer")
            
        except Exception as e:
            logger.error(f"Error sending layer {layer_id} KV cache to consumer: {e}")
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
                    
                    if self._is_layer_signal(signal):
                        # Receive layer-wise data
                        input_tokens = self.data_pipe.recv_tensor().cpu()
                        roi = self.data_pipe.recv_tensor().cpu()
                        if roi is not None:
                            roi = (roi > 0.5)  # Convert back to bool
                        key = self.data_pipe.recv_tensor().cpu()
                        value = self.data_pipe.recv_tensor().cpu()
                        hidden = self.data_pipe.recv_tensor().cpu()
                        layer_id = self.data_pipe.recv_tensor().cpu().item()
                        total_layers = self.data_pipe.recv_tensor().cpu().item()
                        
                        # Add to layer-wise buffer
                        self._add_layer_to_buffer(input_tokens, roi, key, value, hidden, layer_id, total_layers)
                        logger.debug(f"Received and buffered layer {layer_id} KV cache from producer")
                    else:
                        # Receive regular complete data
                        input_tokens = self.data_pipe.recv_tensor().cpu()
                        roi = self.data_pipe.recv_tensor().cpu()
                        if roi is not None:
                            roi = (roi > 0.5)  # Convert back to bool
                        key = self.data_pipe.recv_tensor().cpu()
                        value = self.data_pipe.recv_tensor().cpu()
                        hidden = self.data_pipe.recv_tensor().cpu()
                        
                        # Add to regular buffer
                        self._add_to_buffer(input_tokens, roi, key, value, hidden)
                        logger.info("Received and buffered complete KV cache from producer")
                    
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
        
        # Start consumer mode if not already started
        if not self.is_consumer:
            self.start_consumer_mode()

        # Query the local buffer for matching KV cache
        tokens_roi_recver = [
            input_tokens.cpu() if input_tokens is not None else None,
            roi.cpu() if roi is not None else None
        ]
        
        def is_buffer_available(tokens_roi_recver: List[torch.Tensor]) -> bool:
            # perform input tokens and roi matching
            for _ in range(len(self.buffer)):
                if self._matches(self.buffer[0], tokens_roi_recver) > 0:
                    return True
                # rotate the element we just accessed to the end
                self.buffer.rotate(-1)
            return False

        with self.buffer_cv:
            while not is_buffer_available(tokens_roi_recver):
                logger.debug("KV transfer buffer is not available. Waiting...")
                torch.cuda.nvtx.range_push("drop_select_wait")
                self.buffer_cv.wait()
                torch.cuda.nvtx.range_pop()
            
            # Get matching item from local buffer
            matched_item = self.buffer.popleft()
            # Move matched items to GPU
            matched_item = [item.cuda() if item is not None else None for item in matched_item]
            
            # Update buffer size
            for tensor in matched_item:
                if tensor is not None:
                    self.buffer_size -= self._get_element_size(tensor)
            self.buffer_cv.notify()

        torch.cuda.nvtx.range_pop()
        return matched_item

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

    def insert_layer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                    key: torch.Tensor, value: torch.Tensor,
                    hidden: torch.Tensor, layer_id: int, total_layers: int) -> None:
        """
        Producer-side layer-wise insert: queue single layer KV for sending
        
        Args:
            input_tokens: Input token sequence
            roi: Region of interest mask
            key: Key tensor for this layer
            value: Value tensor for this layer  
            hidden: Hidden state (typically only meaningful for final layer)
            layer_id: ID of the layer (0-indexed)
            total_layers: Total number of layers in the model
        """
        torch.cuda.nvtx.range_push("insert_layer")
        
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
        
        # Queue the layer KV data for sequential sending by push worker
        layer_data = (input_tokens, roi, key, value, hidden, layer_id, total_layers)
        self.push_queue.put(layer_data)
        
        torch.cuda.nvtx.range_pop()
        logger.debug(f"Layer {layer_id} KV cache queued for sending")

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
        
        # Clean up layer buffer
        with self.layer_buffer_cv:
            self.layer_buffer.clear()
        
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
            
        # If this is a producer, signal end to consumer
        # if not self.is_consumer:
        #     self.signal_end()
