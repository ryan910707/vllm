# SPDX-License-Identifier: Apache-2.0
"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Push-based transfer with buffer status checking: producer queries consumer's buffer status before sending
    - Blocking `insert` (process blocks until KV data is sent), non-blocking `drop_select`.
    - Use CPU signal pipe to coordinate transfer completion and buffer status queries
    - Producer waits for sufficient buffer space before sending to prevent deadlocks
"""
import threading
from collections import deque
import time
from typing import Deque, List, Optional, Union

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
        - Producer blocks on insert() until KV is sent to consumer buffer
        - Consumer receives KV in background and stores locally
        - drop_select fetches from local buffer
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

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None
        
        # Buffer status query signals
        self.buffer_status_query_signal = torch.tensor([1], device="cpu")  # Producer queries buffer status
        self.buffer_status_ok_signal = torch.tensor([2], device="cpu")     # Consumer responds: buffer has space
        self.buffer_status_full_signal = torch.tensor([3], device="cpu")   # Consumer responds: buffer is full

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
        # logger.info(f"Data size: {data_size}")

        with self.buffer_cv:
            if self.buffer_size + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                start_time = time.time()
                torch.cuda.nvtx.range_push("KV transfer buffer wait")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.buffer_cv.wait()
                torch.cuda.nvtx.range_pop()
                logger.debug(f"KV transfer buffer wait end")
                logger.info(f"Buffer blocked time: {time.time() - start_time}")

            self.buffer_size += data_size
            self.buffer.append(buffer_item)
            self.buffer_cv.notify()

    def _is_end_signal(self, signal):
        return signal is None
    
    def _is_buffer_status_query(self, signal):
        return signal is not None and torch.equal(signal, self.buffer_status_query_signal)
    
    def _is_buffer_status_response(self, signal):
        return (signal is not None and 
                (torch.equal(signal, self.buffer_status_ok_signal) or 
                 torch.equal(signal, self.buffer_status_full_signal)))
    
    def _query_buffer_status(self, data_size: int) -> bool:
        """
        Producer queries consumer buffer status before sending.
        Returns True if buffer has enough space, False otherwise.
        """
        logger.debug("Querying consumer buffer status...")
        
        # Send buffer status query signal
        self.signal_pipe.send_tensor(self.buffer_status_query_signal)
        
        # Send data size so consumer can check if it fits
        data_size_tensor = torch.tensor([data_size], device="cpu", dtype=torch.float32)
        self.signal_pipe.send_tensor(data_size_tensor)
        
        # Wait for consumer response
        response = self.signal_pipe.recv_tensor()
        
        if torch.equal(response, self.buffer_status_ok_signal):
            logger.debug("Consumer buffer has space - proceeding with send")
            return True
        elif torch.equal(response, self.buffer_status_full_signal):
            logger.debug("Consumer buffer is full - waiting")
            return False
        else:
            logger.warning(f"Unexpected buffer status response: {response}")
            return False

    def _send_kv_data_atomic(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                            key: torch.Tensor, value: torch.Tensor, 
                            hidden: torch.Tensor):
        """Atomically send KV data to consumer (called by push worker)"""
        try:
            torch.cuda.nvtx.range_push("send_kv_data_atomic")
            
            # Calculate data size first
            buffer_item = [input_tokens, roi, key, value, hidden]
            data_size = sum([self._get_element_size(data) for data in buffer_item])
            
            # Query buffer status and wait until there's enough space
            logger.debug(f"Checking buffer status for data size: {data_size}")
            start_wait_time = time.time()
            
            has_waited = False
            while not self._query_buffer_status(data_size):
                # logger.info("Buffer full, waiting before retry...")
                time.sleep(0.1)  # Wait before retrying
                has_waited = True

            wait_time = time.time() - start_wait_time
            if has_waited:  # Log if we waited more than 10ms
                logger.info(f" ----------- Waited {wait_time:.3f}s for buffer space")
            
            # Send signal to indicate incoming KV data
            self.signal_pipe.send_tensor(self.normal_signal)
            
            # Send the KV data atomically
            self.data_pipe.send_tensor(input_tokens)
            self.data_pipe.send_tensor(roi.float() if roi is not None else roi)
            self.data_pipe.send_tensor(key)
            self.data_pipe.send_tensor(value)
            self.data_pipe.send_tensor(hidden)
            
            torch.cuda.nvtx.range_pop()
            logger.info("---------------- Successfully sent KV cache to consumer")
            
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
                    # Wait for signal indicating incoming data or status query
                    signal = self.signal_pipe.recv_tensor()
                    if self._is_end_signal(signal):
                        logger.info("Received end signal in receive handler!")
                        break
                    
                    # Handle buffer status query
                    if self._is_buffer_status_query(signal):
                        logger.debug("Received buffer status query")
                        
                        # Receive the data size being queried
                        data_size_tensor = self.signal_pipe.recv_tensor()
                        data_size = int(data_size_tensor.item())
                        
                        # Check if buffer has enough space
                        with self.buffer_cv:
                            has_space = (self.buffer_size + data_size <= self.buffer_size_threshold)
                        
                        # Send response
                        if has_space:
                            logger.debug(f"Buffer has space for {data_size} bytes")
                            self.signal_pipe.send_tensor(self.buffer_status_ok_signal)
                        else:
                            logger.debug(f"Buffer full: {self.buffer_size} + {data_size} > {self.buffer_size_threshold}")
                            self.signal_pipe.send_tensor(self.buffer_status_full_signal)
                        
                        continue  # Go back to waiting for next signal
                    
                    # Handle normal KV data (existing logic)
                    if torch.equal(signal, self.normal_signal):
                        # Receive the KV data
                        input_tokens = self.data_pipe.recv_tensor()
                        roi = self.data_pipe.recv_tensor()
                        if roi is not None:
                            roi = (roi > 0.5)  # Convert back to bool
                        key = self.data_pipe.recv_tensor()
                        value = self.data_pipe.recv_tensor()
                        hidden = self.data_pipe.recv_tensor()
                        
                        # Add to local buffer
                        self._add_to_buffer(input_tokens, roi, key, value, hidden)
                        logger.debug("Received and buffered KV cache from producer")
                    
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
        tokens_roi_recver = [input_tokens, roi]
        
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
            # Update buffer size
            for tensor in matched_item:
                if tensor is not None:
                    self.buffer_size -= self._get_element_size(tensor)
            self.buffer_cv.notify()

        torch.cuda.nvtx.range_pop()
        logger.info("Drop-selected KV cache from buffer")
        return matched_item

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        """
        Producer-side insert: blocking send of KV data to consumer
        This will block the calling process until the KV data is sent
        """
        torch.cuda.nvtx.range_push("insert_blocking_mode")
        
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
        
        # Directly send KV data (blocking)
        self._send_kv_data_atomic(input_tokens, roi, key, value, hidden)
        
        torch.cuda.nvtx.range_pop()
        logger.debug("KV cache sent successfully")

    def signal_end(self):
        """Signal that no more KV caches will be sent"""
        try:
            # Since insert is now blocking, all KV data has already been sent
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
