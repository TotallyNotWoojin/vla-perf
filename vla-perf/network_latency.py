# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Network Latency Estimation Script

This script estimates network latency for transmitting data under various network conditions:
- WiFi 5/6/7 (802.11ac/ax/be)
- Data center networks (10 Gbps, 25 Gbps, 100 Gbps, 400 Gbps)

Calculates bidirectional latency for:
1. Robot → Server: Image transmission (various resolutions)
2. Server → Robot: Action commands OR KV-cache (for distributed inference)

For each configuration, it calculates:
- Data size in bytes
- Transfer time based on network bandwidth
- Total latency (transfer + fixed network latency)
- Maximum achievable frequency (Hz) for round-trip communication
"""


import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import sys
import logging
from datetime import datetime

from GenZ.Models.Model_sets.vla_models import vla_models
from perf_utils import setup_logging


# Global logger (will be initialized in main)
logger = None


@dataclass
class NetworkConfig:
    """Configuration for a network type."""
    name: str
    bandwidth_mbps_theoretical_upload: float  # Theoretical upload bandwidth in Mbps
    bandwidth_mbps_theoretical_download: float  # Theoretical download bandwidth in Mbps
    base_latency_ms: float  # Base latency (propagation + processing)
    description: str
    bandwidth_efficiency_upload: float = 1.0  # Upload efficiency factor (0 < efficiency ≤ 1.0)
    bandwidth_efficiency_download: float = 1.0  # Download efficiency factor (0 < efficiency ≤ 1.0)

    def bandwidth_mbps(self, direction: str) -> float:
        """
        Get effective bandwidth in Mbps for the specified direction.
        
        Args:
            direction: "upload" or "download"
            
        Returns:
            Effective bandwidth in Mbps
        """
        if direction == "upload":
            return self.bandwidth_mbps_theoretical_upload * self.bandwidth_efficiency_upload
        elif direction == "download":
            return self.bandwidth_mbps_theoretical_download * self.bandwidth_efficiency_download
        else:
            # If an invalid direction is provided, return the maximum effective bandwidth
            upload_bw = self.bandwidth_mbps_theoretical_upload * self.bandwidth_efficiency_upload
            download_bw = self.bandwidth_mbps_theoretical_download * self.bandwidth_efficiency_download
            return max(upload_bw, download_bw)


# 4G LTE (target effective ~80 Mbps downlink, ~20 Mbps uplink)
CELL_4G_LTE_CONFIG = NetworkConfig(
    name="4G LTE",
    bandwidth_mbps_theoretical_upload=75.0,      # single-user lab-ish peak
    bandwidth_mbps_theoretical_download=300.0,   # single-user lab-ish peak
    bandwidth_efficiency_upload=0.25,            # ≈20 Mbps effective
    bandwidth_efficiency_download=0.25,          # ≈81 Mbps effective
    base_latency_ms=25.0,                        # one-way; RTT ~50 ms
    description="4G LTE, urban/suburban good-case throughput (asymmetric)",
)

# 5G NR (SA, sub-6GHz) (target ~500 Mbps downlink, ~80 Mbps uplink)
CELL_5G_SA_CONFIG = NetworkConfig(
    name="5G NR SA (sub-6GHz)",
    bandwidth_mbps_theoretical_upload=320.0,     # sub-6GHz uplink
    bandwidth_mbps_theoretical_download=2000.0,  # sub-6GHz downlink
    bandwidth_efficiency_upload=0.25,            # ≈80 Mbps effective
    bandwidth_efficiency_download=0.25,          # ≈500 Mbps effective
    base_latency_ms=10.0,                        # one-way; RTT ~20 ms
    description="5G Standalone, sub-6GHz, low-latency core, good RF conditions (asymmetric)",
)

# WiFi Network Configurations (asymmetric: upload typically 70-80% of download)
WIFI_5_CONFIG = NetworkConfig(
    name="WiFi 5 (802.11ac)",
    bandwidth_mbps_theoretical_upload=350,      # ~70% of download for 2x2, 80 MHz
    bandwidth_mbps_theoretical_download=500,    # realistic good-case TCP for 2x2, 80 MHz
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=4.0,
    description="WiFi 5, 2x2 MIMO, 80MHz, optimistic good-case throughput (asymmetric)",
)

WIFI_6_CONFIG = NetworkConfig(
    name="WiFi 6 (802.11ax)",
    bandwidth_mbps_theoretical_upload=560,      # ~70% of download for 2x2, 80 MHz
    bandwidth_mbps_theoretical_download=800,    # realistic good-case TCP 2x2, 80 MHz
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=3.5,
    description="WiFi 6, 2x2 MIMO, 80MHz, improved efficiency (asymmetric)",
)

WIFI_6E_CONFIG = NetworkConfig(
    name="WiFi 6E (802.11ax 6GHz)",
    bandwidth_mbps_theoretical_upload=1050,     # ~70% of download for 2x2, 160 MHz
    bandwidth_mbps_theoretical_download=1500,   # good-case 2x2, 160 MHz
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=3.0,
    description="WiFi 6E, 2x2 MIMO, 160MHz, cleaner band (asymmetric)",
)

WIFI_7_CONFIG = NetworkConfig(
    name="WiFi 7 (802.11be)",
    bandwidth_mbps_theoretical_upload=2100,     # ~70% of download for 2x2, 320 MHz
    bandwidth_mbps_theoretical_download=3000,   # realistic high-end 2x2, 320 MHz
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=2.5,
    description="WiFi 7, 2x2 MIMO, 320MHz, MLO-enabled low-tail latency (asymmetric)",
)


# Ethernet Network Configurations (symmetric: upload == download)
ETHERNET_1G_CONFIG = NetworkConfig(
    name="Ethernet 1GbE",
    bandwidth_mbps_theoretical_upload=1000,
    bandwidth_mbps_theoretical_download=1000,
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=0.1,  # Low latency in data center
    description="Ethernet 1 Gigabit Ethernet (symmetric)",
)

ETHERNET_10G_CONFIG = NetworkConfig(
    name="Ethernet 10GbE",
    bandwidth_mbps_theoretical_upload=10000,
    bandwidth_mbps_theoretical_download=10000,
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=0.05,
    description="Ethernet 10 Gigabit Ethernet (symmetric)",
)

ETHERNET_25G_CONFIG = NetworkConfig(
    name="Ethernet 25GbE",
    bandwidth_mbps_theoretical_upload=25000,
    bandwidth_mbps_theoretical_download=25000,
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=0.05,
    description="Ethernet 25 Gigabit Ethernet (symmetric)",
)

ETHERNET_100G_CONFIG = NetworkConfig(
    name="Ethernet 100GbE",
    bandwidth_mbps_theoretical_upload=100000,
    bandwidth_mbps_theoretical_download=100000,
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=0.03,
    description="Ethernet 100 Gigabit Ethernet (symmetric)",
)

ETHERNET_400G_CONFIG = NetworkConfig(
    name="Ethernet 400GbE",
    bandwidth_mbps_theoretical_upload=400000,
    bandwidth_mbps_theoretical_download=400000,
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=0.02,
    description="Ethernet 400 Gigabit Ethernet (symmetric)",
)

# Cloud Network Configurations (includes local network + WAN latency)
CLOUD_FAST_CONFIG = NetworkConfig(
    name="Cloud (Fast: 10Gbps, 10ms)",
    bandwidth_mbps_theoretical_upload=10000,
    bandwidth_mbps_theoretical_download=10000,
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=10.0,  # WAN latency to cloud
    description="Cloud inference with fast network: 10 Gbps, 10ms latency",
)

CLOUD_SLOW_CONFIG = NetworkConfig(
    name="Cloud (Slow: 1Gbps, 100ms)",
    bandwidth_mbps_theoretical_upload=1000,
    bandwidth_mbps_theoretical_download=1000,
    bandwidth_efficiency_upload=1.0,
    bandwidth_efficiency_download=1.0,
    base_latency_ms=100.0,  # WAN latency to cloud
    description="Cloud inference with slow network: 1 Gbps, 100ms latency",
)

# All network configurations
ALL_WIFI_CONFIGS = [
    CELL_4G_LTE_CONFIG,
    CELL_5G_SA_CONFIG,
    WIFI_5_CONFIG,
    WIFI_6_CONFIG,
    WIFI_6E_CONFIG,
    WIFI_7_CONFIG,
]

ALL_DATACENTER_CONFIGS = [
    ETHERNET_1G_CONFIG,
    ETHERNET_10G_CONFIG,
    ETHERNET_25G_CONFIG,
    ETHERNET_100G_CONFIG,
    ETHERNET_400G_CONFIG,
]

ALL_CLOUD_CONFIGS = [
    CLOUD_FAST_CONFIG,
    CLOUD_SLOW_CONFIG,
]

ALL_NETWORK_CONFIGS = ALL_WIFI_CONFIGS + ALL_DATACENTER_CONFIGS + ALL_CLOUD_CONFIGS 

PI0_VLM_SEQUENCE_LENGTH = 256
OPENVLA_VLM_SEQUENCE_LENGTH = 256

@dataclass
class ImageConfig:
    """Configuration for an image."""
    resolution: int  # Image width/height (assuming square images)
    channels: int = 3  # RGB channels
    bytes_per_pixel: int = 1  # 1 byte for uint8, 2 for uint16, 4 for float32
    compression_ratio: float = 1.0  # 1.0 = no compression, 0.1 = 10x compression
    
    @property
    def name(self) -> str:
        """Human-readable name for this image config."""
        comp_str = "" if self.compression_ratio == 1.0 else f" (comp {self.compression_ratio:.1f}x)"
        depth_str = {1: "uint8", 2: "uint16", 4: "fp32"}.get(self.bytes_per_pixel, f"{self.bytes_per_pixel}B")
        return f"{self.resolution}x{self.resolution} {depth_str}{comp_str}"
    
    @property
    def size_bytes(self) -> int:
        """Calculate image size in bytes."""
        uncompressed_size = self.resolution * self.resolution * self.channels * self.bytes_per_pixel
        return int(uncompressed_size * self.compression_ratio)
    
    @property
    def size_mb(self) -> float:
        """Calculate image size in megabytes."""
        return self.size_bytes / (1024 * 1024)


@dataclass
class ActionConfig:
    """Configuration for robot action data."""
    num_dof: int  # Total degrees of freedom (e.g., 8 for 7-DoF arm + gripper, 40+ for humanoid)
    action_chunk_size: int = 1  # Number of future actions (action chunking)
    bytes_per_value: int = 4  # 4 bytes for float32
    
    @property
    def name(self) -> str:
        """Human-readable name for this action config."""
        horizon_str = f" x{self.action_chunk_size}" if self.action_chunk_size > 1 else ""
        return f"{self.num_dof}DoF{horizon_str}"
    
    @property
    def size_bytes(self) -> int:
        """Calculate action data size in bytes."""
        return self.num_dof * self.action_chunk_size * self.bytes_per_value
    
    @property
    def size_kb(self) -> float:
        """Calculate action size in kilobytes."""
        return self.size_bytes / 1024


@dataclass
class VLMKVCacheConfig:
    """Configuration for VLM KV-cache transfer (e.g., Helix distributed inference)."""
    model_name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_length: int  # Number of tokens in KV cache
    bytes_per_element: int = 2  # 2 bytes for fp16/bf16, 4 for fp32
    
    @property
    def name(self) -> str:
        """Human-readable name for this KV-cache config."""
        dtype_str = {2: "fp16", 4: "fp32"}.get(self.bytes_per_element, f"{self.bytes_per_element}B")
        return f"{self.model_name} (seq={self.seq_length}, {dtype_str})"
    
    @property
    def size_bytes(self) -> int:
        """
        Calculate KV-cache size in bytes.
        
        KV-cache stores both Key and Value tensors:
        - Shape: [num_layers, 2 (K+V), seq_length, num_kv_heads, head_dim]
        - Total elements: num_layers * 2 * seq_length * num_kv_heads * head_dim
        """
        total_elements = (
            self.num_layers * 
            2 *  # K and V
            self.seq_length * 
            self.num_kv_heads * 
            self.head_dim
        )
        return total_elements * self.bytes_per_element
    
    @property
    def size_mb(self) -> float:
        """Calculate KV-cache size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Calculate KV-cache size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)


# Common image resolutions
COMMON_IMAGE_CONFIGS = [
    ImageConfig(resolution=224),   # Small (e.g., ImageNet, MobileNet)
    ImageConfig(resolution=384),   # Medium (e.g., SigLIP)
    ImageConfig(resolution=512),   # Medium-large
    ImageConfig(resolution=1024),  # High-res (e.g., ViT-L)
]

# JPEG compressed variants (more realistic for robotics)
# Wikipedia: "JPEG typically achieves 10:1 compression with noticeable, 
# 	but widely agreed to be acceptable perceptible loss in image quality."
COMPRESSED_IMAGE_CONFIGS = [
    ImageConfig(resolution=224, compression_ratio=0.1),   # ~5KB
    ImageConfig(resolution=384, compression_ratio=0.1),   # ~15KB
    ImageConfig(resolution=512, compression_ratio=0.1),   # ~25KB
    ImageConfig(resolution=1024, compression_ratio=0.1),  # ~100KB
]


# Common robot action configurations
COMMON_ACTION_CONFIGS = [
    ActionConfig(num_dof=8, action_chunk_size=1),    # 7-DoF arm + gripper (e.g., OpenVLA)
    ActionConfig(num_dof=8, action_chunk_size=10),   # 7-DoF arm + gripper with action chunking
    ActionConfig(num_dof=40, action_chunk_size=1),   # Humanoid with hands
    ActionConfig(num_dof=40, action_chunk_size=10),   # Humanoid with dexterous hands with chunking
]

def get_kvcache_configs_from_model_config(model_name: str, seq_lengths: List[int], pretty_name: str = None) -> List[VLMKVCacheConfig]:
    """
    Utility to extract KV-cache configs from ModelConfig and build VLMKVCacheConfig objects
    for different sequence lengths.

    Args:
        model_name: The name key from vla_models dict (e.g. "openvla-7b-llm", "pi0-vlm", etc)
        seq_lengths: List of sequence lengths for the KV cache configs
        pretty_name: Human-readable name for reporting, otherwise use model_name

    Returns:
        List of VLMKVCacheConfig objects
    """
    if vla_models is None:
        raise ImportError("vla_models not available. Cannot load model configs dynamically.")
    
    mc = vla_models[model_name]
    
    # Get total number of layers (encoder + decoder)
    assert mc.num_encoder_layers == 0, "Encoder layers are not supported for KV cache configs"
    num_layers = mc.num_decoder_layers
    
    # Get KV heads (defaults to attention heads if not specified)
    num_kv_heads = mc.num_key_value_heads if mc.num_key_value_heads else mc.num_attention_heads
    
    # Calculate head dimension
    if mc.head_dim is not None:
        head_dim = mc.head_dim
    else:
        head_dim = mc.hidden_size // mc.num_attention_heads
    
    return [
        VLMKVCacheConfig(
            model_name=pretty_name or model_name,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_length=sl,
        )
        for sl in seq_lengths
    ]

# ---- Load configs by model ----

# OpenVLA: Llama 2 7B backbone
OPENVLA_KV_CONFIGS = get_kvcache_configs_from_model_config(
    model_name="openvla-7b-llm",
    seq_lengths=[OPENVLA_VLM_SEQUENCE_LENGTH],
    pretty_name="OpenVLA-7B",
)

# Pi0: Gemma 2B with MQA
PI0_KV_CONFIGS = get_kvcache_configs_from_model_config(
    model_name="pi0-vlm",
    seq_lengths=[PI0_VLM_SEQUENCE_LENGTH],
    pretty_name="Pi0 (Gemma 2B)",
)

# Pi0.6: Gemma 3 4B with GQA
PI0_6_KV_CONFIGS = get_kvcache_configs_from_model_config(
    model_name="pi0.6-vlm",
    seq_lengths=[PI0_VLM_SEQUENCE_LENGTH],
    pretty_name="Pi0.6 (Gemma 3 4B)",
        )
        
ALL_VLM_KV_CONFIGS = (
    OPENVLA_KV_CONFIGS + 
    PI0_KV_CONFIGS + 
    PI0_6_KV_CONFIGS
)


def calculate_transfer_time_ms(
    image_size_bytes: int,
    bandwidth_mbps: float,
) -> float:
    """
    Calculate transfer time in milliseconds.
    
    Args:
        image_size_bytes: Image size in bytes
        bandwidth_mbps: Network bandwidth in megabits per second
        
    Returns:
        Transfer time in milliseconds
    """
    # Convert bytes to bits
    image_size_bits = image_size_bytes * 8
    
    # Convert Mbps to bits per millisecond
    bandwidth_bpms = bandwidth_mbps * 1000  # 1 Mbps = 1000 bits/ms
    
    # Calculate transfer time
    transfer_time_ms = image_size_bits / bandwidth_bpms
    
    return transfer_time_ms


def estimate_image_latency(
    network_config: NetworkConfig,
    image_config: ImageConfig,
    protocol_overhead: float = 1.05,  # 5% overhead for TCP/IP headers
) -> dict:
    """
    Estimate network latency for transmitting an image.
    Images are always transmitted as upload (Robot → Server).

    Args:
        network_config: Network configuration
        image_config: Image configuration
        protocol_overhead: Multiplier for protocol overhead

    Returns:
        Dictionary with latency estimates for image transfer
    """
    # Calculate effective image size with protocol overhead
    effective_size_bytes = image_config.size_bytes * protocol_overhead

    # Use upload bandwidth for image transmission
    upload_bandwidth_mbps = network_config.bandwidth_mbps("upload")

    # Calculate transfer time
    transfer_time_ms = calculate_transfer_time_ms(
        effective_size_bytes,
        upload_bandwidth_mbps,
    )

    # Total latency = base latency + transfer time
    total_latency_ms = network_config.base_latency_ms + transfer_time_ms

    # Calculate max frequencies:
    # - Sync: Wait for full round-trip before next transmission (latency-limited)
    # - Async: Continuous streaming with pipelining (bandwidth-limited)
    max_frequency_sync_hz = 1000.0 / total_latency_ms
    max_frequency_async_hz = 1000.0 / transfer_time_ms

    return {
        "network": network_config.name,
        "network_bandwidth_mbps_upload": network_config.bandwidth_mbps("upload"),
        "network_bandwidth_mbps_download": network_config.bandwidth_mbps("download"),
        "network_base_latency_ms": network_config.base_latency_ms,
        "image_config": image_config.name,
        "image_resolution": image_config.resolution,
        "image_size_bytes": image_config.size_bytes,
        "image_size_mb": image_config.size_mb,
        "effective_size_bytes": effective_size_bytes,
        "transfer_time_ms": transfer_time_ms,
        "total_latency_ms": total_latency_ms,
        "max_frequency_sync_hz": max_frequency_sync_hz,
        "max_frequency_async_hz": max_frequency_async_hz,
    }


def evaluate_all_networks_image_transmission(
    network_configs: List[NetworkConfig] = None,
    image_configs: List[ImageConfig] = None,
    protocol_overhead: float = 1.05,
) -> pd.DataFrame:
    """
    Evaluate image transmission network latency for all combinations of networks and image configs.

    Args:
        network_configs: List of network configurations (default: all)
        image_configs: List of image configurations (default: common resolutions)
        protocol_overhead: Protocol overhead multiplier

    Returns:
        DataFrame with all results for image transmission
    """
    if network_configs is None:
        network_configs = ALL_NETWORK_CONFIGS

    if image_configs is None:
        image_configs = COMMON_IMAGE_CONFIGS

    results = []

    for net_config in network_configs:
        for img_config in image_configs:
            result = estimate_image_latency(
                net_config,
                img_config,
                protocol_overhead,
            )
            results.append(result)

    return pd.DataFrame(results)


def estimate_action_latency(
    network_config: NetworkConfig,
    action_config: ActionConfig,
    protocol_overhead: float = 1.05, 
) -> dict:
    """
    Estimate network latency for transmitting action commands (server → robot).
    Actions are always transmitted as download (Server → Robot).
    
    Args:
        network_config: Network configuration
        action_config: Action configuration
        protocol_overhead: Multiplier for protocol overhead
        
    Returns:
        Dictionary with latency estimates
    """
    effective_size_bytes = action_config.size_bytes * protocol_overhead
    
    # Use download bandwidth for action transmission
    download_bandwidth_mbps = network_config.bandwidth_mbps("download")
    
    transfer_time_ms = calculate_transfer_time_ms(
        effective_size_bytes,
        download_bandwidth_mbps,
    )
    
    total_latency_ms = network_config.base_latency_ms + transfer_time_ms
    
    # Calculate max frequencies (sync and async)
    max_frequency_sync_hz = 1000.0 / total_latency_ms
    max_frequency_async_hz = 1000.0 / transfer_time_ms
    
    return {
        "network": network_config.name,
        "network_bandwidth_mbps_upload": network_config.bandwidth_mbps("upload"),
        "network_bandwidth_mbps_download": network_config.bandwidth_mbps("download"),
        "network_base_latency_ms": network_config.base_latency_ms,
        "action_config": action_config.name,
        "action_dof": action_config.num_dof,
        "action_chunk_size": action_config.action_chunk_size,
        "action_size_bytes": action_config.size_bytes,
        "action_size_kb": action_config.size_kb,
        "effective_size_bytes": effective_size_bytes,
        "transfer_time_ms": transfer_time_ms,
        "total_latency_ms": total_latency_ms,
        "max_frequency_sync_hz": max_frequency_sync_hz,
        "max_frequency_async_hz": max_frequency_async_hz,
    }


def estimate_kvcache_latency(
    network_config: NetworkConfig,
    kvcache_config: VLMKVCacheConfig,
    protocol_overhead: float = 1.05,  
) -> dict:
    """
    Estimate network latency for transmitting KV-cache (server → robot or distributed inference).
    KV-cache is always transmitted as download (Server → Robot).
    
    Args:
        network_config: Network configuration
        kvcache_config: VLM KV-cache configuration
        protocol_overhead: Multiplier for protocol overhead
        
    Returns:
        Dictionary with latency estimates
    """
    effective_size_bytes = kvcache_config.size_bytes * protocol_overhead
    
    # Use download bandwidth for KV-cache transmission
    download_bandwidth_mbps = network_config.bandwidth_mbps("download")
    
    transfer_time_ms = calculate_transfer_time_ms(
        effective_size_bytes,
        download_bandwidth_mbps,
    )
    
    total_latency_ms = network_config.base_latency_ms + transfer_time_ms
    
    # Calculate max frequencies (sync and async)
    max_frequency_sync_hz = 1000.0 / total_latency_ms
    max_frequency_async_hz = 1000.0 / transfer_time_ms
    
    return {
        "network": network_config.name,
        "network_bandwidth_mbps_upload": network_config.bandwidth_mbps("upload"),
        "network_bandwidth_mbps_download": network_config.bandwidth_mbps("download"),
        "network_base_latency_ms": network_config.base_latency_ms,
        "kvcache_config": kvcache_config.name,
        "model_name": kvcache_config.model_name,
        "num_layers": kvcache_config.num_layers,
        "seq_length": kvcache_config.seq_length,
        "kvcache_size_bytes": kvcache_config.size_bytes,
        "kvcache_size_mb": kvcache_config.size_mb,
        "kvcache_size_gb": kvcache_config.size_gb,
        "effective_size_bytes": effective_size_bytes,
        "transfer_time_ms": transfer_time_ms,
        "total_latency_ms": total_latency_ms,
        "max_frequency_sync_hz": max_frequency_sync_hz,
        "max_frequency_async_hz": max_frequency_async_hz,
    }


def evaluate_all_actions(
    network_configs: List[NetworkConfig] = None,
    action_configs: List[ActionConfig] = None,
) -> pd.DataFrame:
    """Evaluate action transmission latency for all network/action combinations."""
    if network_configs is None:
        network_configs = ALL_NETWORK_CONFIGS
    
    if action_configs is None:
        action_configs = COMMON_ACTION_CONFIGS
    
    results = []
    for net_config in network_configs:
        for action_config in action_configs:
            result = estimate_action_latency(net_config, action_config)
            results.append(result)
    
    return pd.DataFrame(results)


def evaluate_all_kvcaches(
    network_configs: List[NetworkConfig] = None,
    kvcache_configs: List[VLMKVCacheConfig] = None,
) -> pd.DataFrame:
    """Evaluate KV-cache transmission latency for all network/cache combinations."""
    if network_configs is None:
        network_configs = ALL_NETWORK_CONFIGS
    
    if kvcache_configs is None:
        kvcache_configs = ALL_VLM_KV_CONFIGS
    
    results = []
    for net_config in network_configs:
        for kvcache_config in kvcache_configs:
            result = estimate_kvcache_latency(net_config, kvcache_config)
            results.append(result)
    
    return pd.DataFrame(results)


def compute_network_throughput_hz(
    network_config: NetworkConfig,
    image_config: Optional[ImageConfig] = None,
    action_config: Optional[ActionConfig] = None,
    kvcache_config: Optional[VLMKVCacheConfig] = None,
    protocol_overhead: float = 1.05,
) -> float:
    """
    Compute network throughput in Hz for continuous streaming (excluding base latency).

    For continuous streaming with pipelining, the network throughput is limited by
    transfer time only (base latency is one-time and can be hidden by pipelining).

    For bidirectional communication (image upload + action download), assumes
    time-multiplexed network: upload image, then download action, repeat.

    Network Hz = 1000 / (image_transfer_time + action_transfer_time)

    Args:
        network_config: Network configuration
        image_config: Image configuration 
        action_config: Action configuration (if using action mode)
        kvcache_config: KV-cache configuration (if using distributed inference mode)
        protocol_overhead: Protocol overhead multiplier

    Returns:
        Network throughput in Hz (excluding base latency)
    """
    # At least one of image_config, action_config, or kvcache_config must be provided
    assert (image_config is not None) or (action_config is not None) or (kvcache_config is not None), \
        "At least one of image_config, action_config, or kvcache_config must not be None."

    image_transfer_time_ms = 0.0
    if image_config is not None:
        effective_image_size = image_config.size_bytes * protocol_overhead
        upload_bandwidth_mbps = network_config.bandwidth_mbps("upload")
        image_transfer_time_ms = calculate_transfer_time_ms(
            effective_image_size,
            upload_bandwidth_mbps,
        )

    # Download transfer time (excluding base latency)
    action_transfer_time_ms = 0.0
    if action_config is not None:
        effective_action_size = action_config.size_bytes * protocol_overhead
        download_bandwidth_mbps = network_config.bandwidth_mbps("download")
        action_transfer_time_ms = calculate_transfer_time_ms(
            effective_action_size,
            download_bandwidth_mbps,
        )
    kvcache_transfer_time_ms = 0.0
    if kvcache_config is not None:
        effective_kvcache_size = kvcache_config.size_bytes * protocol_overhead
        download_bandwidth_mbps = network_config.bandwidth_mbps("download")
        kvcache_transfer_time_ms = calculate_transfer_time_ms(
            effective_kvcache_size,
            download_bandwidth_mbps,
        )

    # Total transfer time per cycle (time-multiplexed: upload then download)
    total_transfer_time_ms = image_transfer_time_ms + action_transfer_time_ms + kvcache_transfer_time_ms

    # Network throughput Hz = 1000 / transfer_time_ms
    network_hz = 1000.0 / total_transfer_time_ms if total_transfer_time_ms > 0 else 0.0

    return network_hz



def estimate_bidirectional_latency(
    network_config: NetworkConfig,
    image_config: ImageConfig,
    action_config: Optional[ActionConfig] = None,
    kvcache_config: Optional[VLMKVCacheConfig] = None,
) -> dict:
    """
    Estimate bidirectional (round-trip) latency for robot-server communication.
    
    Round trip consists of:
    1. Robot → Server: Image transmission
    2. Server → Robot: Action OR KV-cache transmission
    
    Args:
        network_config: Network configuration
        image_config: Image configuration
        action_config: Action configuration (if using action mode)
        kvcache_config: KV-cache configuration (if using distributed inference mode)
        
    Returns:
        Dictionary with bidirectional latency results
    """
    # Upload: Robot → Server (image)
    upload_result = estimate_image_latency(network_config, image_config)
    
    # Download: Server → Robot (action or KV-cache)
    if action_config is not None:
        download_result = estimate_action_latency(network_config, action_config)
        return_data_type = "action"
        return_data_config = action_config.name
        return_size_bytes = action_config.size_bytes
    elif kvcache_config is not None:
        download_result = estimate_kvcache_latency(network_config, kvcache_config)
        return_data_type = "kvcache"
        return_data_config = kvcache_config.name
        return_size_bytes = kvcache_config.size_bytes
    else:
        raise ValueError("Must provide either action_config or kvcache_config")
    
    # Total round-trip latency and transfer time
    round_trip_latency_ms = (
        upload_result["total_latency_ms"] + 
        download_result["total_latency_ms"]
    )
    round_trip_transfer_ms = (
        upload_result["transfer_time_ms"] + 
        download_result["transfer_time_ms"]
    )
    
    # Calculate max frequencies:
    # - Sync: Wait for full round-trip (request + response) before next iteration
    # - Async: Continuous streaming with pipelining (limited by transfer bandwidth)
    max_frequency_sync_hz = 1000.0 / round_trip_latency_ms
    max_frequency_async_hz = 1000.0 / round_trip_transfer_ms
    
    return {
        "network": network_config.name,
        "network_bandwidth_mbps_upload": network_config.bandwidth_mbps("upload"),
        "network_bandwidth_mbps_download": network_config.bandwidth_mbps("download"),
        "image_config": image_config.name,
        "image_size_bytes": image_config.size_bytes,
        "return_data_type": return_data_type,
        "return_data_config": return_data_config,
        "return_size_bytes": return_size_bytes,
        "upload_latency_ms": upload_result["total_latency_ms"],
        "download_latency_ms": download_result["total_latency_ms"],
        "round_trip_latency_ms": round_trip_latency_ms,
        "round_trip_transfer_ms": round_trip_transfer_ms,
        "max_frequency_sync_hz": max_frequency_sync_hz,
        "max_frequency_async_hz": max_frequency_async_hz,
    }
    

def print_kvcache_summary(df: pd.DataFrame) -> None:
    """Print KV-cache transmission latency summary."""
    logger.info("\n" + "=" * 130)
    logger.info("KV-Cache Transmission Latency (Server → Robot or Distributed Inference)")
    logger.info("=" * 130)
    
    for network_name in df["network"].unique():
        net_df = df[df["network"] == network_name].sort_values("kvcache_size_mb")
        
        logger.info(f"\n{network_name}:")
        logger.info(f"  Upload Bandwidth: {net_df.iloc[0]['network_bandwidth_mbps_upload']:.0f} Mbps")
        logger.info(f"  Download Bandwidth: {net_df.iloc[0]['network_bandwidth_mbps_download']:.0f} Mbps")
        logger.info(f"  Base Latency: {net_df.iloc[0]['network_base_latency_ms']:.3f} ms")
        logger.info("-" * 130)
        logger.info(f"  {'KV-Cache Config':<45} {'Size':<12} {'Transfer':<12} {'Total Lat':<12} {'Freq(sync)':<12} {'Freq(async)':<13}")
        logger.info(f"  {'':45} {'(MB)':12} {'(ms)':12} {'(ms)':12} {'(Hz)':12} {'(Hz)':13}")
        logger.info("-" * 130)
        
        for _, row in net_df.iterrows():
            logger.info(f"  {row['kvcache_config']:<45} "
                  f"{row['kvcache_size_mb']:>7.2f} MB  "
                  f"{row['transfer_time_ms']:<12.3f} "
                  f"{row['total_latency_ms']:<12.3f} "
                  f"{row['max_frequency_sync_hz']:<12.1f} "
                  f"{row['max_frequency_async_hz']:<13.1f}")


def print_image_summary(df: pd.DataFrame) -> None:
    """Print image transmission summary across all networks and resolutions."""
    logger.info("\n" + "=" * 130)
    logger.info("Image Transmission Latency (Robot → Server) - Comprehensive View")
    logger.info("=" * 130)
    
    # Select key resolutions to display
    key_resolutions = [224, 384, 512, 1024]
    
    for network_name in df["network"].unique():
        net_df = df[df["network"] == network_name]
        
        if net_df.empty:
            continue
            
        logger.info(f"\n{network_name}:")
        logger.info(f"  Upload Bandwidth: {net_df.iloc[0]['network_bandwidth_mbps_upload']:.0f} Mbps")
        logger.info(f"  Download Bandwidth: {net_df.iloc[0]['network_bandwidth_mbps_download']:.0f} Mbps")
        logger.info(f"  Base Latency: {net_df.iloc[0]['network_base_latency_ms']:.3f} ms")
        logger.info("-" * 130)
        logger.info(f"  {'Image Config':<35} {'Size':<12} {'Transfer':<12} {'Total Lat':<12} {'Freq(sync)':<12} {'Freq(async)':<13}")
        logger.info(f"  {'':35} {'':12} {'(ms)':12} {'(ms)':12} {'(Hz)':12} {'(Hz)':13}")
        logger.info("-" * 130)
        
        for res in key_resolutions:
            res_rows = net_df[net_df["image_resolution"] == res].sort_values("image_size_bytes")
            
            for _, row in res_rows.iterrows():
                # Determine if compressed or raw based on size
                # Compressed images have compression_ratio < 1.0 or are much smaller
                is_compressed = row['image_size_bytes'] < (res * res * 3 * 0.5)  # Less than half of raw size
                
                if is_compressed:
                    label = f"{res}x{res} compressed (JPEG 10:1)"
                else:
                    label = f"{res}x{res} raw (RGB uint8)"
                
                size_str = f"{row['image_size_mb']:.2f} MB" if row['image_size_mb'] >= 1 else f"{row['image_size_bytes']/1024:.1f} KB"
                
                logger.info(f"  {label:<35} "
                      f"{size_str:<12} "
                      f"{row['transfer_time_ms']:<12.3f} "
                      f"{row['total_latency_ms']:<12.3f} "
                      f"{row['max_frequency_sync_hz']:<12.1f} "
                      f"{row['max_frequency_async_hz']:<13.1f}")
        
        logger.info("")


def print_action_summary(df: pd.DataFrame) -> None:
    """Print action transmission summary across all networks."""
    logger.info("\n" + "=" * 120)
    logger.info("Action Transmission Latency (Server → Robot) - Comprehensive View")
    logger.info("=" * 120)
    
    for network_name in df["network"].unique():
        net_df = df[df["network"] == network_name].sort_values(["action_dof", "action_chunk_size"])
        
        logger.info(f"\n{network_name}:")
        logger.info(f"  Upload Bandwidth: {net_df.iloc[0]['network_bandwidth_mbps_upload']:.0f} Mbps")
        logger.info(f"  Download Bandwidth: {net_df.iloc[0]['network_bandwidth_mbps_download']:.0f} Mbps")
        logger.info(f"  Base Latency: {net_df.iloc[0]['network_base_latency_ms']:.3f} ms")
        logger.info("-" * 120)
        logger.info(f"  {'Action Config':<35} {'Size':<12} {'Transfer':<12} {'Total Lat':<12} {'Freq(sync)':<12} {'Freq(async)':<13}")
        logger.info(f"  {'':35} {'':12} {'(ms)':12} {'(ms)':12} {'(Hz)':12} {'(Hz)':13}")
        logger.info("-" * 120)
        
        for _, row in net_df.iterrows():
            size_str = f"{row['action_size_bytes']} B"
            logger.info(f"  {row['action_config']:<35} "
                  f"{size_str:<12} "
                  f"{row['transfer_time_ms']:<12.6f} "
                  f"{row['total_latency_ms']:<12.3f} "
                  f"{row['max_frequency_sync_hz']:<12.1f} "
                  f"{row['max_frequency_async_hz']:<13.1f}")
        
        logger.info("")


def print_bidirectional_summary(df: pd.DataFrame, min_frequency_hz: float = 10.0) -> None:
    """Print bidirectional (round-trip) latency summary."""
    logger.info("\n" + "=" * 140)
    logger.info("Bidirectional Latency (Robot ↔ Server Round Trip)")
    logger.info("=" * 140)
    
    for network_name in df["network"].unique():
        net_df = df[df["network"] == network_name]
        
        logger.info(f"\n{network_name}:")
        logger.info("-" * 140)
        logger.info(f"  {'Image':<25} {'Return Data':<25} {'Upload':<10} {'Download':<10} {'RoundTrip':<10} {'Freq(sync)':<11} {'Freq(async)':<12} {'Status'}")
        logger.info(f"  {'':25} {'':25} {'(ms)':10} {'(ms)':10} {'(ms)':10} {'(Hz)':11} {'(Hz)':12} {'':10}")
        logger.info("-" * 140)
        
        for _, row in net_df.iterrows():
            status = "✓" if row["max_frequency_sync_hz"] >= min_frequency_hz else "✗"
            
            # Truncate configs for display
            img_str = row["image_config"][:24]
            ret_str = row["return_data_config"][:24]
            
            logger.info(f"  {img_str:<25} "
                  f"{ret_str:<25} "
                  f"{row['upload_latency_ms']:<10.3f} "
                  f"{row['download_latency_ms']:<10.3f} "
                  f"{row['round_trip_latency_ms']:<10.3f} "
                  f"{row['max_frequency_sync_hz']:<11.1f} "
                  f"{row['max_frequency_async_hz']:<12.1f} "
                  f"{status}")
    
    logger.info("\n" + "=" * 140)


def run_full_analysis(output_dir: str = "perf_results") -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive network latency analysis.
    
    Analysis is organized into five main sections:
    1. Image Transmission (Robot → Server)
    2. Action Transmission (Server → Robot)
    3. KV-Cache Transmission (Server → Robot, for distributed inference)
    4. Bidirectional: Image + Action (typical robot control loop)
    5. Bidirectional: Image + KV-Cache (distributed VLM inference)
    
    Args:
        output_dir: Directory to save results and logs
        
    Returns:
        Dictionary of DataFrames with all results
    """
    # Setup logging
    global logger
    log_file = str(Path(output_dir) / "network_latency.log")
    logger = setup_logging(log_file)
    logger.info(f"Network Latency Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    results = {}
    
    # ========================================================================
    # SECTION 1: IMAGE TRANSMISSION (Robot → Server)
    # ========================================================================
    logger.info("\n" + "#" * 80)
    logger.info("# SECTION 1: IMAGE TRANSMISSION (Robot → Server)")
    logger.info("#" * 80)
    logger.info("Evaluating image transmission latency across all networks...")
    logger.info("Configurations: Compressed (JPEG 10:1) and Raw (RGB uint8)")
    logger.info("Resolutions: 224, 384, 512, 1024\n")
    
    df_images = evaluate_all_networks_image_transmission(
        network_configs=ALL_NETWORK_CONFIGS,
        image_configs=COMMON_IMAGE_CONFIGS + COMPRESSED_IMAGE_CONFIGS,
    )
    results["images"] = df_images
    print_image_summary(df_images)
    
    output_file = output_path / "network_latency_images.csv"
    df_images.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # ========================================================================
    # SECTION 2: ACTION TRANSMISSION (Server → Robot)
    # ========================================================================
    logger.info("\n\n" + "#" * 80)
    logger.info("# SECTION 2: ACTION TRANSMISSION (Server → Robot)")
    logger.info("#" * 80)
    logger.info("Evaluating action command transmission latency...")
    logger.info("Configurations: 8 DoF, 8 DoF x10, 40 DoF, 40 DoF x10\n")
    
    df_actions = evaluate_all_actions(
        network_configs=ALL_NETWORK_CONFIGS,
        action_configs=COMMON_ACTION_CONFIGS,
    )
    results["actions"] = df_actions
    print_action_summary(df_actions)
    
    output_file = output_path / "network_latency_actions.csv"
    df_actions.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # ========================================================================
    # SECTION 3: KV-CACHE TRANSMISSION (Server → Robot / Distributed Inference)
    # ========================================================================
    logger.info("\n\n" + "#" * 80)
    logger.info("# SECTION 3: KV-CACHE TRANSMISSION (For Distributed Inference)")
    logger.info("#" * 80)
    logger.info("Evaluating KV-cache transmission latency...")
    logger.info("Models: OpenVLA-7B, Pi0, Pi0.6")
    logger.info(f"Sequence lengths: {PI0_VLM_SEQUENCE_LENGTH} (Pi0/Pi0.6), {OPENVLA_VLM_SEQUENCE_LENGTH} (OpenVLA) tokens\n")
    
    df_kvcache = evaluate_all_kvcaches()
    results["kvcache"] = df_kvcache
    print_kvcache_summary(df_kvcache)
    
    output_file = output_path / "network_latency_kvcache.csv"
    df_kvcache.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # ========================================================================
    # SECTION 4: BIDIRECTIONAL - Image + Action (Typical Robot Control)
    # ========================================================================
    logger.info("\n\n" + "#" * 80)
    logger.info("# SECTION 4: BIDIRECTIONAL LATENCY - Image + Action")
    logger.info("#" * 80)
    logger.info("Evaluating round-trip latency for robot control loop...")
    logger.info("Round trip: Image upload → Server inference → Action download\n")
    
    # Representative configs for bidirectional analysis
    selected_images = [
        ImageConfig(resolution=224, compression_ratio=0.1),
        ImageConfig(resolution=512, compression_ratio=0.1),
    ]
    selected_actions = [
        ActionConfig(num_dof=8, action_chunk_size=1),
        ActionConfig(num_dof=40, action_chunk_size=10),
    ]
    
    bidirectional_results = []
    for net_config in ALL_NETWORK_CONFIGS:
        for img_config in selected_images:
            for action_config in selected_actions:
                result = estimate_bidirectional_latency(
                    net_config, img_config, action_config=action_config
                )
                bidirectional_results.append(result)
    
    df_bidirectional_action = pd.DataFrame(bidirectional_results)
    results["bidirectional_action"] = df_bidirectional_action
    print_bidirectional_summary(df_bidirectional_action, min_frequency_hz=10.0)
    
    output_file = output_path / "network_latency_bidirectional_action.csv"
    df_bidirectional_action.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # ========================================================================
    # SECTION 5: BIDIRECTIONAL - Image + KV-Cache (Distributed VLM Inference)
    # ========================================================================
    logger.info("\n\n" + "#" * 80)
    logger.info("# SECTION 5: BIDIRECTIONAL LATENCY - Image + KV-Cache")
    logger.info("#" * 80)
    logger.info("Evaluating round-trip latency for distributed VLM inference...")
    logger.info("Scenario: Image upload → Distributed inference → KV-cache download")
    logger.info("(e.g., Helix-style architecture with edge + cloud)\n")
    
    selected_kvcaches = [
        PI0_KV_CONFIGS[0],      
        PI0_6_KV_CONFIGS[0],   
        OPENVLA_KV_CONFIGS[0],  
    ]
    
    bidirectional_kv_results = []
    for net_config in ALL_NETWORK_CONFIGS:
        for img_config in selected_images:
            for kvcache_config in selected_kvcaches:
                result = estimate_bidirectional_latency(
                    net_config, img_config, kvcache_config=kvcache_config
                )
                bidirectional_kv_results.append(result)
    
    df_bidirectional_kv = pd.DataFrame(bidirectional_kv_results)
    results["bidirectional_kvcache"] = df_bidirectional_kv
    print_bidirectional_summary(df_bidirectional_kv, min_frequency_hz=10.0)
    
    output_file = output_path / "network_latency_bidirectional_kvcache.csv"
    df_bidirectional_kv.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # ========================================================================
    # ANALYSIS COMPLETE
    # ========================================================================
    logger.info("\n\n" + "=" * 80)
    logger.info("NETWORK LATENCY ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nAll results saved to: {output_path}/")
    logger.info("\nGenerated files:")
    logger.info("  - network_latency_images.csv")
    logger.info("  - network_latency_actions.csv")
    logger.info("  - network_latency_kvcache.csv")
    logger.info("  - network_latency_bidirectional_action.csv")
    logger.info("  - network_latency_bidirectional_kvcache.csv")
    logger.info("  - network_latency.log (printing messages to console and log file)")
    
    return results

if __name__ == "__main__":
    """Main entry point for network latency analysis."""
    results = run_full_analysis()
    