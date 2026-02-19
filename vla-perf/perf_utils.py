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
Shared utility functions for performance evaluation scripts.

This module provides common helper functions for:
- Parallelism calculations
- Pareto-optimal batch size filtering
- Performance data collection utilities
- Logging setup
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from GenZ import (
    prefill_moddeling,
    decode_moddeling,
    parallel_decode_modeling,
)
from GenZ.Models import get_configs
from GenZ.unit import Unit
from Systems.system_configs import system_configs


# Default result columns for consistency
RESULT_COLUMNS = [
    "model.name",
    "model.stage",
    "model.dec_steps",
    "model.seq_len_inference_prefill",
    "hardware.name",
    "hardware.num_chips",
    "batch_size",
    "boundness",
    "op_intensity",
    "time_ms",
    "weights_mb",
    "kv_cache_mb",
    "total_memory_mb",
]


def setup_logging(log_file: str = "perf_results/perf.log"):
    """
    Set up logging to both console and file.

    Args:
        log_file: Path to the log file

    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # Plain message format without timestamp/level prefix
        handlers=[
            logging.FileHandler(log_file, mode="w"),  # Overwrite log file each run
            logging.StreamHandler(),  # Console output
        ],
    )

    return logging.getLogger(__name__)


def evaluate_boundness(model_df: pd.DataFrame, op_intensity_mode='theoretical') -> tuple[str, float]:
    """
    Evaluate the overall boundness and operational intensity.

    Args:
        model_df: DataFrame containing layer-wise performance metrics with columns:
                  'Compute time (msec)', 'Memory time (msec)', 'Communication time (msec)',
                  'Op Intensity', 'Latency (msec)'

    Returns:
        Tuple of (overall_bound, weighted_avg_op_intensity):
            - overall_bound: "Comp", "Mem", or "Comm"
            - weighted_avg_op_intensity: Weighted average of op intensity across layers
    """
    # Compute total times
    total_compute_time = model_df["Compute time (msec)"].sum()
    total_memory_time = model_df["Memory time (msec)"].sum()
    total_communication_time = model_df["Communication time (msec)"].sum()

    # Compute overall bound: Compute : 0, Memory : 1, Communication : 2
    max_time = max(total_compute_time, total_memory_time, total_communication_time)
    if max_time == total_compute_time:
        overall_bound = "Comp"
    elif max_time == total_memory_time:
        overall_bound = "Mem"
    else:
        overall_bound = "Comm"

    # Compute weighted average op intensity (weighted by latency)
    assert op_intensity_mode in ['theoretical', 'actual']
    assert "Num ops (MFLOP)" in model_df.columns
    total_ops = model_df["Num ops (MFLOP)"].sum()

    if op_intensity_mode == 'actual':
        # Mode 'actual': Data actually loaded from memory (not "Total Data (MB)")
        assert "Op Intensity" in model_df.columns
        memory_bytes = model_df.apply(
            lambda row: (
                row["Num ops (MFLOP)"] / row["Op Intensity"]
                if row["Op Intensity"] != 0 and row["Num ops (MFLOP)"] != 0
                else 0
            ),
            axis=1,
        )
        total_memory_bytes = memory_bytes.sum()
    elif op_intensity_mode == 'theoretical':
        # Mode 'theoretical': Assume all input and output data loaded from memory
        assert "Total Data (MB)" in model_df.columns
        total_memory_bytes = model_df["Total Data (MB)"].sum()

    if total_ops > 0 and total_memory_bytes > 0:
        weighted_avg_op_intensity = total_ops / total_memory_bytes
    else:
        weighted_avg_op_intensity = 0.0

    return overall_bound, weighted_avg_op_intensity


def get_best_precision_for_system(system: str, preferred: str = "bf16") -> str:
    """
    Get the best available precision for a given system.

    Allows conversion between precisions of the same bit-width and type:
    - 32-bit float: fp32 ↔ tf32
    - 16-bit float: bf16 ↔ fp16
    - 8-bit float: fp8 (no alternatives)
    - 8-bit int: int8 (no alternatives)
    - 4-bit float: fp4 (no alternatives)
    - 4-bit int: int4 (no alternatives)

    Does NOT allow conversion between:
    - Different bit-widths (e.g., fp16 → fp32)
    - Integer and float types (e.g., int8 → fp8)

    Args:
        system: System name
        preferred: Preferred precision

    Returns:
        Best available precision string
    """
    if system not in system_configs:
        return preferred

    flops = system_configs[system].get("Flops", {})

    # If Flops is a dict, check available precisions
    if isinstance(flops, dict):
        available = list(flops.keys())

        # If preferred is available, use it
        if preferred in available:
            return preferred

        # Define precision groups: same bit-width and type can be converted
        # Priority within each group: first choice is the most preferred
        precision_groups = {
            # 32-bit float
            "fp32": ["fp32", "tf32"],
            "tf32": ["tf32", "fp32"],
            # 16-bit float
            "bf16": ["bf16", "fp16"],
            "fp16": ["fp16", "bf16"],
            # 8-bit float (no alternatives)
            "fp8": ["fp8"],
            # 8-bit int (no alternatives)
            "int8": ["int8"],
            # 4-bit float (no alternatives)
            "fp4": ["fp4"],
            # 4-bit int (no alternatives)
            "int4": ["int4"],
            # 2-bit int (no alternatives)
            "int2": ["int2"],
        }

        # Try to find an alternative from the same precision group
        if preferred in precision_groups:
            for alternative in precision_groups[preferred]:
                if alternative in available:
                    return alternative

    # If no compatible precision found, return preferred (will likely fail later)
    return preferred


def calculate_kv_cache_size_mb(
    model_name: str,
    seq_length: int,
    bits: str = "bf16",
) -> float:
    """
    Calculate KV cache size in MB for a given model and sequence length.

    Args:
        model_name: Model name (e.g., "pi0-vlm")
        seq_length: Sequence length (number of tokens)
        bits: Precision (bf16, fp16, fp32, etc.)

    Returns:
        KV cache size in MB
    """
    # Get model config
    model_config = get_configs(model_name)

    # Determine bytes per element based on precision
    bytes_per_element = {
        "bf16": 2,
        "fp16": 2,
        "fp32": 4,
        "int8": 1,
    }.get(
        bits.lower(), 2
    )  # Default to 2 bytes (bf16/fp16)

    # Calculate KV cache size
    # Formula: num_layers * 2 (K and V) * seq_length * num_kv_heads * head_dim * bytes_per_element
    num_layers = model_config.num_decoder_layers
    num_kv_heads = (
        model_config.num_key_value_heads
        if model_config.num_key_value_heads
        else model_config.num_attention_heads
    )
    head_dim = (
        model_config.head_dim
        if model_config.head_dim
        else (model_config.hidden_size // model_config.num_attention_heads)
    )

    kv_cache_bytes = (
        num_layers * 2 * seq_length * num_kv_heads * head_dim * bytes_per_element
    )
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)

    return kv_cache_mb


def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def get_powers_of_two_up_to(n: int) -> list[int]:
    """Return a list of powers of two up to the given number n."""
    powers_of_two = []
    power = 1
    while power <= n:
        powers_of_two.append(power)
        power *= 2
    return powers_of_two


def get_parallelism(num_devices: int, system: str = None) -> list[tuple[int, int]]:
    """
    Get all possible parallelism combinations (TP, PP) for a given number of devices.

    Args:
        num_devices: Total number of devices (must be a power of two)
        system: System name (e.g., "RTX_4090"). If provided and ICN=0, filters out TP>1 combinations.

    Returns:
        List of (tensor_parallel, pipeline_parallel) tuples
    """
    if not is_power_of_two(num_devices):
        raise ValueError(f"{num_devices} is not a power of two.")

    # Check system ICN if system is provided
    system_icn = None
    if system is not None:
        try:
            system_icn = system_configs.get(system, {}).get("ICN", None)
        except ImportError:
            pass

    parallelisms = []
    # If ICN=0, only allow tensor parallelism = 1 (single GPU)
    # Tensor parallelism requires interconnect between GPUs
    if system_icn is not None and system_icn == 0:
        # Skip this combination - tensor parallelism requires ICN
        parallelisms = [(1, 1)] if num_devices == 1 else []
    else:
        tp = 1
        while tp <= num_devices:
            pp = num_devices // tp
            if is_power_of_two(pp):
                parallelisms.append((tp, pp))
            tp *= 2

    return parallelisms


# Difference between get_pareto_df and get_optimal_df:
#
# - get_pareto_df: This function removes batch size configurations that are not Pareto-optimal
#   for throughput. Specifically, for each unique configuration (except batch_size),
#   it keeps only batch sizes where the latency for size 2n is significantly better than (or not
#   worse than) twice the latency for size n. If doubling the batch size does not bring at
#   least a ~2x efficiency or better, all larger batch sizes are dropped. The resulting DataFrame
#   has only batch sizes on the Pareto frontier (as far as batch scaling is concerned).
#
# - get_optimal_df: This function selects the single best (lowest time_ms) result for each
#   combination of configuration and batch size. Optionally, it can also invoke get_pareto_df
#   (apply_pareto=True, by default), so its output will be the overall best points per batch size,
#   then filtered to retain only the Pareto-optimal ones (if apply_pareto is set).
#   In effect, get_optimal_df is a strict superset: it applies per-batch-size argmin and then applies
#   the batch-scaling Pareto optimality filter.


def get_pareto_df(df: pd.DataFrame, rtol: float = 1e-02) -> pd.DataFrame:
    """
    Remove the rows with larger batch sizes but not better performance.

    Specifically, clean the DataFrame by removing rows where the latency for
    a batch size of 2n is >= twice the latency for a batch size of n (within 1% rtol).

    Args:
        df: DataFrame with performance results

    Returns:
        Filtered DataFrame with only Pareto-optimal batch sizes
    """
    rows_to_drop = []

    group_columns = [
        "model.name",
        "model.stage",
        "model.dec_steps",
        "model.seq_len_inference_prefill",
        "hardware.name",
        "hardware.num_chips",
    ]

    for group, group_df in df.groupby(group_columns):
        batch_sizes = sorted(group_df["batch_size"].unique())

        for batch_size in batch_sizes:
            if batch_size == batch_sizes[-1]:
                break

            n_row = group_df[group_df["batch_size"] == batch_size]
            two_n_row = group_df[group_df["batch_size"] == batch_size * 2]

            if len(two_n_row) == 0:
                continue

            latency_n = n_row["time_ms"].values[0]
            latency_2n = two_n_row["time_ms"].values[0]

            if latency_2n >= 2 * latency_n or np.allclose(
                latency_2n, 2 * latency_n, rtol=rtol
            ):
                rows_to_drop.extend(
                    group_df[group_df["batch_size"] >= batch_size * 2].index
                )
                break

    df.drop(rows_to_drop, inplace=True)
    return df


def get_optimal_df(df: pd.DataFrame, apply_pareto: bool = True) -> pd.DataFrame:
    """
    Filter DataFrame to keep only optimal performance results.

    Args:
        df: DataFrame with performance results
        apply_pareto: Whether to apply Pareto filtering for batch sizes

    Returns:
        Filtered DataFrame with optimal results
    """
    if df.empty:
        return df

    # Only keep the best performance for each combination
    df_optimal = df.loc[
        df.groupby(
            [
                "model.name",
                "model.stage",
                "model.dec_steps",
                "model.seq_len_inference_prefill",
                "hardware.name",
                "hardware.num_chips",
                "batch_size",
            ]
        ).time_ms.idxmin()
    ]

    if apply_pareto:
        df_optimal = get_pareto_df(df_optimal)

    return df_optimal


def collect_prefill_perf(
    model,
    system: str,
    num_devices: int,
    input_tokens: int,
    bits: str = "bf16",
    max_batch_size: int = 1024,
    batch_size_multiplier: int = 1,
) -> list[dict]:
    """
    Collect prefill performance for a model across batch sizes and parallelism configs.

    Args:
        model: Model name (str) or ModelConfig object
        system: System name (e.g., "H100")
        num_devices: Number of devices
        input_tokens: Number of input tokens
        bits: Bit precision (e.g., "bf16", "int8")
        max_batch_size: Maximum batch size to try
        batch_size_multiplier: Multiplier for effective batch size (e.g., for multi-frame inference)

    Returns:
        List of result dictionaries
    """
    # Automatically use the best available precision for the system
    bits = get_best_precision_for_system(system, bits)

    results = []
    parallelism_combinations = get_parallelism(num_devices, system=system)

    # Get model name for results
    model_name = model if isinstance(model, str) else model.model

    for tp, pp in parallelism_combinations:
        # When pipeline_parallel > 1, batch_size must be >= pp to avoid warnings
        # Start from pp instead of 1 to avoid micro-batch warnings
        batch_size = max(1, pp)
        while batch_size <= max_batch_size:
            try:
                # Apply batch_size_multiplier for models that process multiple frames/inputs
                effective_batch_size = batch_size * batch_size_multiplier
                
                prefill_output = prefill_moddeling(
                    model=model,
                    batch_size=effective_batch_size,
                    input_tokens=input_tokens,
                    system_name=system,
                    bits=bits,
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    debug=False,
                )

                # Evaluate boundness and op intensity
                boundness, op_intensity = evaluate_boundness(prefill_output["model_df"])

                # Extract memory information from summary_table
                unit = Unit()
                summary_table = prefill_output.get("summary_table")
                weights_mb = summary_table[f'Total Weights ({unit.unit_mem})'].values[0] if summary_table is not None else 0
                kv_cache_mb = summary_table[f'KV Cache ({unit.unit_mem})'].values[0] if summary_table is not None else 0
                total_memory_mb = weights_mb + kv_cache_mb

                # Store logical batch_size (divide by multiplier if used)
                # This ensures results always show the logical batch size
                logical_batch_size = batch_size
                
                results.append(
                    {
                        "model.name": model_name,
                        "model.stage": "prefill",
                        "model.dec_steps": 1,
                        "model.seq_len_inference_prefill": input_tokens,
                        "hardware.name": system,
                        "hardware.num_chips": num_devices,
                        "batch_size": logical_batch_size,
                        "boundness": boundness,
                        "op_intensity": op_intensity,
                        "time_ms": prefill_output["Latency"],
                        "weights_mb": weights_mb,
                        "kv_cache_mb": kv_cache_mb,
                        "total_memory_mb": total_memory_mb,
                    }
                )

                batch_size *= 2

            except Exception as e:
                if batch_size == 1:
                    print(
                        f"Warning: Failed for model={model_name}, system={system}, "
                        f"tp={tp}, pp={pp}, batch_size={batch_size}: {e}"
                    )
                # Break out of batch_size loop but continue to next (tp, pp) combination
                break

    return results


def collect_decode_perf(
    model,
    system: str,
    num_devices: int,
    input_tokens: int,
    output_tokens: int,
    beam: int = 1,
    bits: str = "bf16",
    max_batch_size: int = 1024,
) -> list[dict]:
    """
    Collect decode performance for a model across batch sizes and parallelism configs.

    Args:
        model: Model name (str) or ModelConfig object
        system: System name (e.g., "H100")
        num_devices: Number of devices
        input_tokens: Number of input tokens (context length)
        output_tokens: Number of output tokens to generate
        beam: Beam width for beam search
        bits: Bit precision (e.g., "bf16", "int8")
        max_batch_size: Maximum batch size to try

    Returns:
        List of result dictionaries
    """
    # Automatically use the best available precision for the system
    bits = get_best_precision_for_system(system, bits)

    results = []
    parallelism_combinations = get_parallelism(num_devices, system=system)

    # Get model name for results
    model_name = model if isinstance(model, str) else model.model

    for tp, pp in parallelism_combinations:
        # When pipeline_parallel > 1, batch_size must be >= pp to avoid warnings
        # Start from pp instead of 1 to avoid micro-batch warnings
        batch_size = max(1, pp)
        while batch_size <= max_batch_size:
            try:
                decode_output = decode_moddeling(
                    model=model,
                    batch_size=batch_size,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    Bb=beam,
                    system_name=system,
                    bits=bits,
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    debug=False,
                )

                # Evaluate boundness and op intensity
                boundness, op_intensity = evaluate_boundness(decode_output["model_df"])

                # Extract memory information from summary_table
                unit = Unit()
                summary_table = decode_output.get("summary_table")
                weights_mb = summary_table[f'Total Weights ({unit.unit_mem})'].values[0] if summary_table is not None else 0
                kv_cache_mb = summary_table[f'KV Cache ({unit.unit_mem})'].values[0] if summary_table is not None else 0
                total_memory_mb = weights_mb + kv_cache_mb

                results.append(
                    {
                        "model.name": model_name,
                        "model.stage": "decode",
                        "model.dec_steps": output_tokens,
                        "model.seq_len_inference_prefill": input_tokens,
                        "hardware.name": system,
                        "hardware.num_chips": num_devices,
                        "batch_size": batch_size,
                        "boundness": boundness,
                        "op_intensity": op_intensity,
                        "time_ms": decode_output["Latency"],
                        "weights_mb": weights_mb,
                        "kv_cache_mb": kv_cache_mb,
                        "total_memory_mb": total_memory_mb,
                    }
                )

                batch_size *= 2

            except Exception as e:
                if batch_size == 1:
                    print(
                        f"Warning: Failed for model={model_name}, system={system}, "
                        f"tp={tp}, pp={pp}, batch_size={batch_size}: {e}"
                    )
                break

    return results


def collect_parallel_decode_perf(
    model,
    system: str,
    num_devices: int,
    input_tokens: int,
    output_tokens_parallel: int,
    self_attention: bool = False,
    bits: str = "bf16",
    max_batch_size: int = 1024,
) -> list[dict]:
    """
    Collect parallel decode performance for a model across batch sizes and parallelism configs.

    Args:
        model: Model name (str) or ModelConfig object
        system: System name (e.g., "A100_80GB")
        num_devices: Number of devices
        input_tokens: Number of input tokens (context length)
        output_tokens_parallel: Number of tokens to decode in parallel
        self_attention: Whether parallel tokens attend to each other
        bits: Bit precision (e.g., "bf16", "int8")
        max_batch_size: Maximum batch size to try

    Returns:
        List of result dictionaries
    """
    # Automatically use the best available precision for the system
    bits = get_best_precision_for_system(system, bits)

    results = []
    parallelism_combinations = get_parallelism(num_devices, system=system)

    # Get model name for results
    model_name = model if isinstance(model, str) else model.model

    for tp, pp in parallelism_combinations:
        # When pipeline_parallel > 1, batch_size must be >= pp to avoid warnings
        # Start from pp instead of 1 to avoid micro-batch warnings
        batch_size = max(1, pp)
        while batch_size <= max_batch_size:
            try:
                parallel_decode_output = parallel_decode_modeling(
                    model=model,
                    batch_size=batch_size,
                    input_tokens=input_tokens,
                    output_tokens_parallel=output_tokens_parallel,
                    self_attention=self_attention,
                    system_name=system,
                    bits=bits,
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    debug=False,
                )

                # Evaluate boundness and op intensity
                boundness, op_intensity = evaluate_boundness(
                    parallel_decode_output["model_df"]
                )

                # Extract memory information from summary_table
                unit = Unit()
                summary_table = parallel_decode_output.get("summary_table")
                weights_mb = summary_table[f'Total Weights ({unit.unit_mem})'].values[0] if summary_table is not None else 0
                kv_cache_mb = summary_table[f'KV Cache ({unit.unit_mem})'].values[0] if summary_table is not None else 0
                total_memory_mb = weights_mb + kv_cache_mb

                results.append(
                    {
                        "model.name": model_name,
                        "model.stage": "parallel_decode",
                        "model.dec_steps": output_tokens_parallel,
                        "model.seq_len_inference_prefill": input_tokens,
                        "hardware.name": system,
                        "hardware.num_chips": num_devices,
                        "batch_size": batch_size,
                        "boundness": boundness,
                        "op_intensity": op_intensity,
                        "time_ms": parallel_decode_output["Latency"],
                        "weights_mb": weights_mb,
                        "kv_cache_mb": kv_cache_mb,
                        "total_memory_mb": total_memory_mb,
                    }
                )

                batch_size *= 2

            except Exception as e:
                if batch_size == 1:
                    print(
                        f"Warning: Failed for model={model_name}, system={system}, "
                        f"tp={tp}, pp={pp}, batch_size={batch_size}: {e}"
                    )
                break

    return results

def calculate_transformer_params(config):
    """
    Calculate the total number of parameters in a transformer model.
    
    For encoder-only models (num_decoder_layers=0):
    - Embedding (if vocab_size > 0): vocab_size * hidden_size
    - Per encoder layer:
      * Attention QKV: hidden_size * (num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim)
      * Attention output: hidden_size * hidden_size
      * FFN up: hidden_size * intermediate_size * num_ffi
      * FFN down: intermediate_size * hidden_size
    - Layer norms: ~2 * hidden_size per layer (negligible, but included)
    
    For decoder-only models (num_encoder_layers=0):
    - Embedding: vocab_size * hidden_size
    - Per decoder layer: same as encoder layer
    - Output projection: vocab_size * hidden_size
    
    For encoder-decoder models: sum of both
    """
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_encoder_layers = config.num_encoder_layers
    num_decoder_layers = config.num_decoder_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_ffi = config.num_ffi
    
    total_params = 0
    
    # Embedding layer (if vocab_size > 0)
    if vocab_size > 0:
        total_params += vocab_size * hidden_size
    
    # Per-layer parameters
    def layer_params(num_layers):
        """Calculate parameters for num_layers"""
        if num_layers == 0:
            return 0
        
        params = 0
        for _ in range(num_layers):
            # Attention QKV projection
            # Q: hidden_size * (num_attention_heads * head_dim)
            # K, V: hidden_size * (num_key_value_heads * head_dim) each
            qkv_params = hidden_size * (num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim)
            
            # Attention output projection
            out_proj_params = hidden_size * hidden_size
            
            # Feed-forward network
            # Up projection: can have num_ffi parallel projections
            ffn_up_params = hidden_size * intermediate_size * num_ffi
            # Down projection
            ffn_down_params = intermediate_size * hidden_size
            
            # Layer norms (2 per layer: attention norm and FFN norm)
            # Each norm has hidden_size params for scale and hidden_size for bias
            layer_norm_params = 2 * (hidden_size + hidden_size)  # scale + bias for each norm
            
            params += qkv_params + out_proj_params + ffn_up_params + ffn_down_params + layer_norm_params
        
        return params
    
    # Encoder layers
    total_params += layer_params(num_encoder_layers)
    
    # Decoder layers
    total_params += layer_params(num_decoder_layers)
    
    # Output projection (for decoder-only models with vocab_size > 0)
    if num_decoder_layers > 0 and vocab_size > 0:
        total_params += vocab_size * hidden_size
    
    return total_params


def format_param_count(count):
    """Format parameter count in a human-readable format."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    else:
        return str(int(count))