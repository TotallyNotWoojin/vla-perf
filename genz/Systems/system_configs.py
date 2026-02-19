# SPDX-FileCopyrightText: Copyright (c) 2024 Multifidelity Roofline Analysis
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT. Portions are Apache-2.0 while others are MIT.
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
#
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

from typing import Any, Dict, Union

# Flops can be either:
# - A single number (legacy format, assumed to be bf16 TFLOPS)
# - A dict mapping precision -> TFLOPS (e.g., {'bf16': 312, 'int8': 624, 'fp32': 156})
# Supported precisions: 'fp32', 'tf32', 'bf16', 'fp16', 'fp8', 'int8', 'int4', 'int2'
# Memory_size: in GB
# Memory_BW: in GB/s
# ICN: Interconnect bandwidth in GB/s

system_configs: Dict[str, Dict[str, Any]] = {
    # NVIDIA A100 40GB - https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
    'A100_40GB': {
        'Flops': {'fp32': 19.5, 'tf32': 156, 'bf16': 312, 'fp16': 312, 'int8': 624},
        'Memory_size': 40, 'Memory_BW': 1555, 'ICN': 150, 'real_values': True
    },
    # NVIDIA A100 80GB
    'A100_80GB': {
        'Flops': {'fp32': 19.5, 'tf32': 156, 'bf16': 312, 'fp16': 312, 'int8': 624},
        'Memory_size': 80, 'Memory_BW': 2039, 'ICN': 150, 'real_values': True
    },
    # NVIDIA H100 SXM - https://www.nvidia.com/en-us/data-center/h100/
    'H100': {
        'Flops': {'fp32': 67, 'tf32': 495, 'bf16': 989, 'fp16': 989, 'fp8': 1979, 'int8': 1979},
        'Memory_size': 80, 'Memory_BW': 3350, 'ICN': 450, 'real_values': True
    },
    # NVIDIA GH200 - Grace Hopper Superchip
    'GH200': {
        'Flops': {'fp32': 67, 'tf32': 495, 'bf16': 989, 'fp16': 989, 'fp8': 1979, 'int8': 1979},
        'Memory_size': 144, 'Memory_BW': 4900, 'ICN': 450, 'real_values': True
    },
    # NVIDIA B100 - https://resources.nvidia.com/en-us-blackwell-architecture
    'B100': {
        'Flops': {'fp32': 60, 'tf32': 875, 'bf16': 1750, 'fp16': 1750, 'fp8': 3500, 'fp4': 7000, 'int8': 3500},
        'Memory_size': 192, 'Memory_BW': 8000, 'ICN': 900, 'ICN_LL': 0.25, 'real_values': True
    },
    # NVIDIA GB200 - Blackwell
    'GB200': {
        'Flops': {'fp32': 75, 'tf32': 1125, 'bf16': 2250, 'fp16': 2250, 'fp8': 4500, 'fp4': 9000, 'int8': 4500},
        'Memory_size': 192, 'Memory_BW': 8000, 'ICN': 900, 'ICN_LL': 0.25, 'real_values': True
    },
    # Google TPUv6 (Trillium) - https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
    'TPUv6': {
        'Flops': {'bf16': 926, 'int8': 1852},
        'Memory_size': 32, 'Memory_BW': 1640, 'ICN': 100, 'real_values': True
    },
    # Google TPUv5e
    'TPUv5e': {
        'Flops': {'bf16': 197, 'int8': 394},
        'Memory_size': 16, 'Memory_BW': 820, 'ICN': 50, 'real_values': True
    },
    # Google TPUv5p
    'TPUv5p': {
        'Flops': {'bf16': 459, 'int8': 918},
        'Memory_size': 95, 'Memory_BW': 2765, 'ICN': 450, 'real_values': True
    },
    # Google TPUv4
    'TPUv4': {
        'Flops': {'bf16': 275, 'int8': 550},
        'Memory_size': 32, 'Memory_BW': 1228, 'ICN': 24, 'real_values': True
    },
    # AMD MI300X - https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
    'MI300X': {
        'Flops': {'fp32': 163, 'bf16': 1307, 'fp16': 1307, 'fp8': 2614, 'int8': 2614},
        'Memory_size': 192, 'Memory_BW': 5300, 'ICN': 400, 'real_values': True
    },
    # AMD MI325X
    'MI325X': {
        'Flops': {'fp32': 163, 'bf16': 1307, 'fp16': 1307, 'fp8': 2614, 'int8': 2614},
        'Memory_size': 256, 'Memory_BW': 6000, 'ICN': 400, 'real_values': True
    },
    # Intel Gaudi3 - https://habana.ai/products/gaudi3/
    'Gaudi3': {
        'Flops': {'bf16': 1835, 'fp16': 1835, 'fp8': 1835},
        'Memory_size': 128, 'Memory_BW': 3675, 'ICN': 300, 'real_values': True
    },

    # ==================== NVIDIA Jetson Family ====================
    # ICN=0 for all Jetson (always single device)
    # https://developer.nvidia.com/embedded/jetson-modules

    # Jetson Nano: 128-core Maxwell GPU
    # No tensor cores, no INT8 acceleration
    'Jetson_Nano': {
        'Flops': {'fp32': 0.25, 'fp16': 0.5},
        'Memory_size': 4, 'Memory_BW': 25.6, 'ICN': 0, 'real_values': True
    },
    # Jetson TX2: 256-core Pascal GPU
    # No tensor cores, no INT8 acceleration
    'Jetson_TX2': {
        'Flops': {'fp32': 0.67, 'fp16': 1.33},
        'Memory_size': 8, 'Memory_BW': 59.7, 'ICN': 0, 'real_values': True
    },
    # Jetson Xavier NX: 384-core Volta GPU + 48 Tensor Cores
    # Volta has INT8 tensor cores
    'Jetson_Xavier_NX': {
        'Flops': {'fp32': 1.3, 'fp16': 6.5, 'int8': 21},
        'Memory_size': 8, 'Memory_BW': 51.2, 'ICN': 0, 'real_values': True
    },
    # Jetson AGX Xavier: 512-core Volta GPU + 64 Tensor Cores
    'Jetson_AGX_Xavier': {
        'Flops': {'fp32': 2.8, 'fp16': 11, 'int8': 32},
        'Memory_size': 32, 'Memory_BW': 136.5, 'ICN': 0, 'real_values': True
    },
    # Jetson Orin Nano 4GB: 512-core Ampere GPU + 16 Tensor Cores
    # Ampere has INT8/INT4 tensor cores, sparse support
    'Jetson_Orin_Nano_4GB': {
        'Flops': {'fp32': 1.25, 'fp16': 5, 'int8': 10, 'int4': 20},
        'Memory_size': 4, 'Memory_BW': 34, 'ICN': 0, 'real_values': True
    },
    # Jetson Orin Nano 8GB: 1024-core Ampere GPU + 32 Tensor Cores
    'Jetson_Orin_Nano_8GB': {
        'Flops': {'fp32': 2.5, 'fp16': 10, 'int8': 20, 'int4': 40},
        'Memory_size': 8, 'Memory_BW': 68, 'ICN': 0, 'real_values': True
    },
    # Jetson Orin NX 8GB: 1024-core Ampere GPU + 32 Tensor Cores
    'Jetson_Orin_NX_8GB': {
        'Flops': {'fp32': 4.4, 'fp16': 17.5, 'int8': 35, 'int4': 70},
        'Memory_size': 8, 'Memory_BW': 68, 'ICN': 0, 'real_values': True
    },
    # Jetson Orin NX 16GB: 1024-core Ampere GPU + 32 Tensor Cores
    'Jetson_Orin_NX_16GB': {
        'Flops': {'fp32': 6.25, 'fp16': 25, 'int8': 50, 'int4': 100},
        'Memory_size': 16, 'Memory_BW': 102.4, 'ICN': 0, 'real_values': True
    },
    # Jetson AGX Orin 32GB: 1792-core Ampere GPU + 56 Tensor Cores
    'Jetson_AGX_Orin_32GB': {
        'Flops': {'fp32': 12.5, 'fp16': 50, 'int8': 100, 'int4': 200},
        'Memory_size': 32, 'Memory_BW': 204.8, 'ICN': 0, 'real_values': True
    },
    # Jetson AGX Orin 64GB: 2048-core Ampere GPU + 64 Tensor Cores
    'Jetson_AGX_Orin_64GB': {
        'Flops': {'fp32': 17, 'fp16': 67, 'int8': 138, 'int4': 275},
        'Memory_size': 64, 'Memory_BW': 204.8, 'ICN': 0, 'real_values': True
    },
    # Jetson AGX Thor: 2560-core Blackwell GPU + 96 Tensor Cores
    'Jetson_AGX_Thor': {
        'Flops': {'fp32': 100, 'fp16': 400, 'fp8': 800, 'int8': 800, 'fp4': 2000, 'int4': 2000},
        'Memory_size': 128, 'Memory_BW': 270, 'ICN': 0, 'real_values': True
    },

    # ==================== PC GPU Family ====================
    # NVIDIA RTX 4090 - Ada Lovelace architecture
    'RTX_4090': {
        'Flops': {'fp32': 82.6, 'bf16': 165, 'fp16': 165, 'int8': 330, 'int4': 660},
        'Memory_size': 24, 'Memory_BW': 1008, 'ICN': 0, 'real_values': True
    },
    # NVIDIA RTX 5090 - Blackwell architecture (GB202)
    # 21,760 CUDA cores, 680 5th-gen Tensor Cores, 32GB GDDR7
    # FP16/BF16 tensor TFLOPS are dense with FP32 accumulation (consistent with RTX 4090 convention)
    # https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
    'RTX_5090': {
        'Flops': {'fp32': 105, 'bf16': 209, 'fp16': 209, 'fp8': 419, 'int8': 419, 'fp4': 838, 'int4': 838},
        'Memory_size': 32, 'Memory_BW': 1792, 'ICN': 0, 'real_values': True
    },
}
