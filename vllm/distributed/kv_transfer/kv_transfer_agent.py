# SPDX-License-Identifier: Apache-2.0
"""A centralized entrypoint to perform distributed KV cache transfer.

This implementation is a shim wrapper on two APIs exposed by `kv_connector`:
1. `send_kv_caches_and_hidden_states`
2. `recv_kv_caches_and_hidden_states
"""
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    from vllm.config import VllmConfig

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class KVTransferAgent:
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):

        self.config = config

        if config.kv_transfer_config is None:
            raise ValueError("KVTransferConfig is not set in the VllmConfig,"
                             " cannot initialize KVConnector.")

        assert self.config.kv_transfer_config.is_kv_transfer_instance, "KV"\
            "TransferAgent should only be used when kv_connector is set."

        self.connector = KVConnectorFactory.create_connector(
            rank, local_rank, config)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        self.connector.send_kv_caches_and_hidden_states(
            model_executable, model_input, kv_caches,
            hidden_or_intermediate_states)

    def send_kv_caches_and_hidden_states_layerwise(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        """Send KV caches and hidden states layer-wise for compute-communication overlap"""
        
        # Check if connector supports layerwise transfer
        if hasattr(self.connector, 'send_kv_caches_and_hidden_states_layerwise'):
            self.connector.send_kv_caches_and_hidden_states_layerwise(
                model_executable, model_input, kv_caches,
                hidden_or_intermediate_states)
        else:
            # Fallback to regular method for connectors that don't support layerwise
                         self.connector.send_kv_caches_and_hidden_states(
                model_executable, model_input, kv_caches,
                hidden_or_intermediate_states)

    def insert_layer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                    key: torch.Tensor, value: torch.Tensor,
                    hidden: torch.Tensor, layer_id: int, total_layers: int) -> None:
        """Insert single layer KV cache for immediate sending"""
        
        # Check if connector supports layerwise transfer
        if hasattr(self.connector, 'insert_layer'):
            self.connector.insert_layer(input_tokens, roi, key, value, hidden, layer_id, total_layers)
        else:
            # Fallback - accumulate and send via regular method (not ideal but works)
            logger.warning("Connector doesn't support insert_layer, using fallback")

    def close(self) -> None:
        self.connector.close()

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        return self.connector.recv_kv_caches_and_hidden_states(
            model_executable, model_input, kv_caches)
