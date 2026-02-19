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
LLM Performance Evaluation Script

This script evaluates LLM performance across three phases:
1. Prefill - processing input context tokens
2. Decode - sequential autoregressive token generation
3. Parallel Decode - parallel token generation (useful for diffusion models)

The script tests a Llama model on A100 hardware with configurable parameters.
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Add parent directory to sys.path for local module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from perf_utils import (
    get_powers_of_two_up_to,
    get_optimal_df,
    collect_prefill_perf,
    collect_decode_perf,
    collect_parallel_decode_perf,
    RESULT_COLUMNS,
    setup_logging,
)

# ============================================================================
# CONFIGURATION - Modify these values to change test parameters
# ============================================================================

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-70B"  # Llama model to test
# Alternative models: "meta-llama/Llama-3.1-70B", "llama2_7b", etc.

# Hardware configuration
SYSTEM_NAME = "B100"  # Hardware system, e.g., B100, Jetson_AGX_Thor
BITS = "bf16"  # Precision: "bf16", "int8", etc.

# Input/Output configuration
INPUT_TOKENS = 102400 # Input context length (KV cache from VLM or other source)
OUTPUT_TOKENS = 1  # Number of tokens to generate (for decode and parallel decode)
PARALLEL_DECODE_TOKENS = 256  # Number of tokens to decode in parallel
PARALLEL_SELF_ATTENTION = True  # Whether parallel tokens attend to each other

# Device configuration
NUM_DEVICES_LIST = [1, 2, 4]  # List of device counts to test
MAX_BATCH_SIZE = 1024  # Maximum batch size to test

# Output configuration
OUTPUT_DIR = "perf_results"  # Directory to save results
LOG_FILE = "perf_results/test_llm_perf.log"  # Log file path

# ============================================================================


def get_llm_prefill_perf(
    system_list: list[str],
    num_device_list: list[int],
    model_name: str,
    input_tokens: int,
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate LLM prefill stage performance.
    
    Args:
        system_list: List of hardware systems to evaluate
        num_device_list: List of device counts to test
        model_name: Model name to test
        input_tokens: Number of input tokens
        bits: Precision
        max_batch_size: Maximum batch size to test
        logger: Logger instance
        
    Returns:
        DataFrame with prefill performance results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = []
    
    logger.info("Processing LLM prefill stage")
    for system in system_list:
        logger.info(f"  System: {system}")
        for num_devices in num_device_list:
            logger.info(f"    Devices: {num_devices}")
            model_results = collect_prefill_perf(
                model=model_name,
                system=system,
                num_devices=num_devices,
                input_tokens=input_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)
                logger.info(f"      Collected {len(model_results)} results")
            else:
                logger.warning(f"      No results collected for prefill on {system} with {num_devices} devices")
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    
    if df.empty:
        logger.warning("No prefill results collected")
        return df
    
    logger.info(f"Total prefill results: {len(df)}")
    df_optimal = get_optimal_df(df)
    logger.info(f"Optimal prefill results after filtering: {len(df_optimal)}")
    return df_optimal


def get_llm_decode_perf(
    system_list: list[str],
    num_device_list: list[int],
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate LLM decode stage performance (sequential autoregressive generation).
    
    Args:
        system_list: List of hardware systems to evaluate
        num_device_list: List of device counts to test
        model_name: Model name to test
        input_tokens: Number of input tokens (context length)
        output_tokens: Number of tokens to generate
        bits: Precision
        max_batch_size: Maximum batch size to test
        logger: Logger instance
        
    Returns:
        DataFrame with decode performance results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = []
    
    logger.info("Processing LLM decode stage (sequential)")
    for system in system_list:
        logger.info(f"  System: {system}")
        for num_devices in num_device_list:
            logger.info(f"    Devices: {num_devices}")
            model_results = collect_decode_perf(
                model=model_name,
                system=system,
                num_devices=num_devices,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)
                logger.info(f"      Collected {len(model_results)} results")
            else:
                logger.warning(f"      No results collected for decode on {system} with {num_devices} devices")
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    
    if df.empty:
        logger.warning("No decode results collected")
        return df
    
    logger.info(f"Total decode results: {len(df)}")
    df_optimal = get_optimal_df(df)
    logger.info(f"Optimal decode results after filtering: {len(df_optimal)}")
    return df_optimal


def get_llm_parallel_decode_perf(
    system_list: list[str],
    num_device_list: list[int],
    model_name: str,
    input_tokens: int,
    output_tokens_parallel: int,
    self_attention: bool = False,
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate LLM parallel decode stage performance (parallel token generation).
    
    Args:
        system_list: List of hardware systems to evaluate
        num_device_list: List of device counts to test
        model_name: Model name to test
        input_tokens: Number of input tokens (context length)
        output_tokens_parallel: Number of tokens to decode in parallel
        self_attention: Whether parallel tokens attend to each other
        bits: Precision
        max_batch_size: Maximum batch size to test
        logger: Logger instance
        
    Returns:
        DataFrame with parallel decode performance results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = []
    
    logger.info("Processing LLM parallel decode stage")
    for system in system_list:
        logger.info(f"  System: {system}")
        for num_devices in num_device_list:
            logger.info(f"    Devices: {num_devices}")
            model_results = collect_parallel_decode_perf(
                model=model_name,
                system=system,
                num_devices=num_devices,
                input_tokens=input_tokens,
                output_tokens_parallel=output_tokens_parallel,
                self_attention=self_attention,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)
                logger.info(f"      Collected {len(model_results)} results")
            else:
                logger.warning(f"      No results collected for parallel decode on {system} with {num_devices} devices")
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    
    if df.empty:
        logger.warning("No parallel decode results collected")
        return df
    
    logger.info(f"Total parallel decode results: {len(df)}")
    df_optimal = get_optimal_df(df)
    logger.info(f"Optimal parallel decode results after filtering: {len(df_optimal)}")
    return df_optimal


def get_llm_e2e_perf(
    system_list: list[str] = None,
    num_device_list: list[int] = None,
    model_name: str = None,
    input_tokens: int = None,
    output_tokens: int = None,
    output_tokens_parallel: int = None,
    self_attention: bool = None,
    bits: str = None,
    max_batch_size: int = None,
    output_dir: str = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate end-to-end LLM performance across all phases (prefill, decode, parallel decode).
    
    Args:
        system_list: List of hardware systems to evaluate (uses config if None)
        num_device_list: List of device counts to test (uses config if None)
        model_name: Model name to test (uses config if None)
        input_tokens: Input context length (uses config if None)
        output_tokens: Number of tokens for sequential decode (uses config if None)
        output_tokens_parallel: Number of tokens for parallel decode (uses config if None)
        self_attention: Whether parallel tokens attend to each other (uses config if None)
        bits: Precision (uses config if None)
        max_batch_size: Maximum batch size (uses config if None)
        output_dir: Directory to save results (uses config if None)
        logger: Logger instance
        
    Returns:
        Dictionary of DataFrames for each phase
    """
    # Use config values if not provided
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if system_list is None:
        system_list = [SYSTEM_NAME]
    if num_device_list is None:
        num_device_list = NUM_DEVICES_LIST
    if model_name is None:
        model_name = MODEL_NAME
    if input_tokens is None:
        input_tokens = INPUT_TOKENS
    if output_tokens is None:
        output_tokens = OUTPUT_TOKENS
    if output_tokens_parallel is None:
        output_tokens_parallel = PARALLEL_DECODE_TOKENS
    if self_attention is None:
        self_attention = PARALLEL_SELF_ATTENTION
    if bits is None:
        bits = BITS
    if max_batch_size is None:
        max_batch_size = MAX_BATCH_SIZE
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    
    # 1. Prefill performance
    logger.info("Evaluating LLM prefill...")
    df_prefill = get_llm_prefill_perf(
        system_list, num_device_list, model_name, input_tokens, bits, max_batch_size, logger=logger
    )
    results["prefill"] = df_prefill
    if not df_prefill.empty:
        df_prefill.to_csv(output_path / "llm_prefill_perf.csv", index=False)
        logger.info(f"  -> Saved to {output_path / 'llm_prefill_perf.csv'}")
    
    # 2. Sequential decode performance
    logger.info("Evaluating LLM decode (sequential)...")
    df_decode = get_llm_decode_perf(
        system_list, num_device_list, model_name, input_tokens, output_tokens, bits, max_batch_size, logger=logger
    )
    results["decode"] = df_decode
    if not df_decode.empty:
        df_decode.to_csv(output_path / "llm_decode_perf.csv", index=False)
        logger.info(f"  -> Saved to {output_path / 'llm_decode_perf.csv'}")
    
    # 3. Parallel decode performance
    logger.info("Evaluating LLM parallel decode...")
    df_parallel_decode = get_llm_parallel_decode_perf(
        system_list, num_device_list, model_name, input_tokens, output_tokens_parallel,
        self_attention, bits, max_batch_size, logger=logger
    )
    results["parallel_decode"] = df_parallel_decode
    if not df_parallel_decode.empty:
        df_parallel_decode.to_csv(output_path / "llm_parallel_decode_perf.csv", index=False)
        logger.info(f"  -> Saved to {output_path / 'llm_parallel_decode_perf.csv'}")
    
    logger.info("\nLLM performance evaluation complete!")
    return results


def print_summary(results: dict[str, pd.DataFrame], logger=None) -> None:
    """Print a summary of the LLM performance results as a table."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"\nLLM Performance Summary")
    logger.info(f"  - Model: {MODEL_NAME}")
    logger.info(f"  - Hardware: {SYSTEM_NAME}")
    logger.info(f"  - Input tokens: {INPUT_TOKENS}")
    logger.info(f"  - Output tokens (sequential): {OUTPUT_TOKENS}")
    logger.info(f"  - Output tokens (parallel): {PARALLEL_DECODE_TOKENS}")
    logger.info(f"  - Parallel self-attention: {PARALLEL_SELF_ATTENTION}")

    logger.info("\n" + "=" * 160)
    logger.info("Performance Comparison: Prefill vs Sequential Decode vs Parallel Decode")
    logger.info("=" * 160)
    
    # Create comparison table with memory columns
    header = f"{'Phase':<20} {'Hardware':<20} {'Chips':<8} {'Batch':<8} {'Latency (ms)':<15} {'Throughput (tok/s)':<20} {'Weights (MB)':<15} {'KV Cache (MB)':<15} {'Total Mem (MB)':<15}"
    separator = "-" * 160
    logger.info(header)
    logger.info(separator)
    
    # Collect best results for each phase
    for phase_name, phase_key in [("Prefill", "prefill"), ("Sequential Decode", "decode"), ("Parallel Decode", "parallel_decode")]:
        if phase_key in results and not results[phase_key].empty:
            df = results[phase_key]
            # Get best result (lowest latency) for each hardware config
            best = df.loc[df.groupby(["hardware.name", "hardware.num_chips"])["time_ms"].idxmin()]
            
            for _, row in best.iterrows():
                hw = row["hardware.name"]
                chips = int(row["hardware.num_chips"])
                batch = int(row["batch_size"])
                latency = row["time_ms"]
                weights_mb = row.get("weights_mb", 0)
                kv_cache_mb = row.get("kv_cache_mb", 0)
                total_memory_mb = row.get("total_memory_mb", 0)
                
                # Calculate throughput
                if phase_key == "prefill":
                    throughput = (batch * INPUT_TOKENS) / (latency / 1000) if latency > 0 else 0
                elif phase_key == "decode":
                    throughput = (batch * OUTPUT_TOKENS) / (latency / 1000) if latency > 0 else 0
                else:  # parallel_decode
                    throughput = (batch * PARALLEL_DECODE_TOKENS) / (latency / 1000) if latency > 0 else 0
                
                row_str = f"{phase_name:<20} {hw:<20} {chips:<8} {batch:<8} {latency:<15.2f} {throughput:<20.1f} {weights_mb:<15.1f} {kv_cache_mb:<15.1f} {total_memory_mb:<15.1f}"
                logger.info(row_str)
        else:
            logger.warning(f"No results available for {phase_name}")
    
    logger.info(separator)


if __name__ == "__main__":
    # Set up logging
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 80)
    logger.info("Starting LLM Performance Evaluation")
    logger.info("=" * 80)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  System: {SYSTEM_NAME}")
    logger.info(f"  Bits: {BITS}")
    logger.info(f"  Input tokens: {INPUT_TOKENS}")
    logger.info(f"  Output tokens (sequential): {OUTPUT_TOKENS}")
    logger.info(f"  Output tokens (parallel): {PARALLEL_DECODE_TOKENS}")
    logger.info(f"  Parallel self-attention: {PARALLEL_SELF_ATTENTION}")
    logger.info(f"  Number of devices: {NUM_DEVICES_LIST}")
    logger.info(f"  Max batch size: {MAX_BATCH_SIZE}")
    
    # Run end-to-end performance evaluation
    results = get_llm_e2e_perf(
        system_list=[SYSTEM_NAME],
        num_device_list=NUM_DEVICES_LIST,
        model_name=MODEL_NAME,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        output_tokens_parallel=PARALLEL_DECODE_TOKENS,
        self_attention=PARALLEL_SELF_ATTENTION,
        bits=BITS,
        max_batch_size=MAX_BATCH_SIZE,
        logger=logger,
    )
    
    # Print summary
    print_summary(results, logger=logger)
    
    logger.info("=" * 80)
    logger.info("Performance evaluation completed")
    logger.info("=" * 80)
