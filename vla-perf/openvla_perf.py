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
OpenVLA Performance Evaluation Script

OpenVLA (openvla/openvla-7b) is a Vision-Language-Action model for robotics:
- Vision: Dual-encoder (DINOv2 ViT-L/14 + SigLIP SoViT-400m/14), each producing 256 tokens
- Projector: 2-layer MLP fusing vision features into LLM space (2176 -> 4096)
- LLM: Llama 2 7B backbone (32 layers, 4096 hidden)
- Action: 7-DoF actions discretized into 256 bins (7 decode steps)

Performance modeling breakdown:
1. Vision encoding (DINOv2 + SigLIP) - encoder-only, parallel execution
2. LLM prefill - processes projected visual tokens (512 tokens from dual encoders)
3. LLM decode - generates 7 action tokens

Reference files in GenZ:
    genz/GenZ/Models/Model_sets/vla_models.py -> OpenVLA model configs
    genz_scripts/perf_utils.py -> shared performance utilities
"""

import pandas as pd
import logging
from pathlib import Path

from perf_utils import (
    get_powers_of_two_up_to,
    get_optimal_df,
    collect_prefill_perf,
    collect_decode_perf,
    RESULT_COLUMNS,
    setup_logging,
)


# OpenVLA Architecture Constants
# Reference: https://www.jetson-ai-lab.com/openvla.html
# "It has an input image resolution of 224x224 to stacked DINOv2/SigLIP vision 
# 	encoders that are projected to ~275 input tokens (plus the text prompt), and outputs 7 tokens"
OPENVLA_VISION_TOKENS_PER_ENCODER = 256  # Each vision encoder outputs 256 tokens
OPENVLA_TOTAL_VISION_TOKENS = OPENVLA_VISION_TOKENS_PER_ENCODER # Assuming language instruction much shorter than vision features
OPENVLA_ACTION_TOKENS = 7  # 7-DoF action output


def get_openvla_vision_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate OpenVLA vision encoders (DINOv2 + SigLIP).
    
    Both vision encoders run sequentially on the same hardware, so we sum the latencies.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Vision encoder models
    vision_models = [
        "dinov2-large-patch14-vision",      # DINOv2 ViT-L/14
        "siglip2-so400m-patch14-384-vision", # SigLIP SoViT-400m/14
    ]
    
    results = []
    
    for model in vision_models:
        logger.info(f"Processing vision encoder: {model}")
        for system in system_list:
            logger.info(f"  System: {system}")
            for num_devices in num_device_list:
                logger.info(f"    Devices: {num_devices}")
                model_results = collect_prefill_perf(
                    model=model,
                    system=system,
                    num_devices=num_devices,
                    input_tokens=OPENVLA_VISION_TOKENS_PER_ENCODER,
                    bits=bits,
                )
                if model_results:
                    results.extend(model_results)
                    logger.info(f"      Collected {len(model_results)} results")
                else:
                    logger.warning(f"      No results collected for {model} on {system} with {num_devices} devices")
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    
    if df.empty:
        logger.warning("No vision encoder results collected")
        return df
    
    logger.info(f"Total vision encoder results: {len(df)}")
    df_optimal = get_optimal_df(df, apply_pareto=True)
    logger.info(f"Optimal vision encoder results after filtering: {len(df_optimal)}")
    return df_optimal


def get_openvla_llm_prefill_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate OpenVLA LLM prefill stage.
    
    The LLM processes projected visual tokens from both encoders (512 tokens total).
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
                model="openvla-7b-llm",
                system=system,
                num_devices=num_devices,
                input_tokens=OPENVLA_TOTAL_VISION_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)
                logger.info(f"      Collected {len(model_results)} results")
            else:
                logger.warning(f"      No results collected for LLM prefill on {system} with {num_devices} devices")
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    
    if df.empty:
        logger.warning("No LLM prefill results collected")
        return df
    
    logger.info(f"Total LLM prefill results: {len(df)}")
    df_optimal = get_optimal_df(df, apply_pareto=True)
    logger.info(f"Optimal LLM prefill results after filtering: {len(df_optimal)}")
    return df_optimal


def get_openvla_llm_decode_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate OpenVLA LLM decode stage for action token generation.
    
    The LLM generates 7 action tokens (one per DoF) autoregressively.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = []
    
    logger.info("Processing LLM decode stage")
    for system in system_list:
        logger.info(f"  System: {system}")
        for num_devices in num_device_list:
            logger.info(f"    Devices: {num_devices}")
            model_results = collect_decode_perf(
                model="openvla-7b-llm",
                system=system,
                num_devices=num_devices,
                input_tokens=OPENVLA_TOTAL_VISION_TOKENS,
                output_tokens=OPENVLA_ACTION_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)
                logger.info(f"      Collected {len(model_results)} results")
            else:
                logger.warning(f"      No results collected for LLM decode on {system} with {num_devices} devices")
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    
    if df.empty:
        logger.warning("No LLM decode results collected")
        return df
    
    logger.info(f"Total LLM decode results: {len(df)}")
    df_optimal = get_optimal_df(df, apply_pareto=True)
    logger.info(f"Optimal LLM decode results after filtering: {len(df_optimal)}")
    return df_optimal


def get_openvla_e2e_perf(
    system_list: list[str] = ["A100_80GB", "H100", "B100"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate end-to-end OpenVLA performance across all components.
    
    Args:
        system_list: List of hardware systems to evaluate
        num_device_list: List of device counts to test (defaults to powers of 2 up to 4)
        bits: Precision (bf16, int8, etc.)
        output_dir: Directory to save results
        logger: Logger instance for logging messages
        
    Returns:
        Dictionary of DataFrames for each component
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    
    # 1. Vision encoder performance
    logger.info("Evaluating OpenVLA vision encoders...")
    df_vision = get_openvla_vision_perf(system_list, num_device_list, bits, logger=logger)
    results["vision"] = df_vision
    df_vision.to_csv(output_path / "openvla_vision_perf.csv", index=False)
    logger.info(f"  -> Saved to {output_path / 'openvla_vision_perf.csv'}")
    
    # 2. LLM prefill performance
    logger.info("Evaluating OpenVLA LLM prefill...")
    df_prefill = get_openvla_llm_prefill_perf(system_list, num_device_list, bits, logger=logger)
    results["llm_prefill"] = df_prefill
    df_prefill.to_csv(output_path / "openvla_llm_prefill_perf.csv", index=False)
    logger.info(f"  -> Saved to {output_path / 'openvla_llm_prefill_perf.csv'}")
    
    # 3. LLM decode performance
    logger.info("Evaluating OpenVLA LLM decode...")
    df_decode = get_openvla_llm_decode_perf(system_list, num_device_list, bits, logger=logger)
    results["llm_decode"] = df_decode
    df_decode.to_csv(output_path / "openvla_llm_decode_perf.csv", index=False)
    logger.info(f"  -> Saved to {output_path / 'openvla_llm_decode_perf.csv'}")
    
    # 4. Compute E2E latency by combining component latencies
    logger.info("Computing end-to-end latency estimates...")
    
    if df_vision.empty or df_prefill.empty or df_decode.empty:
        logger.warning("One or more component DataFrames are empty. Skipping E2E computation.")
        results["e2e"] = pd.DataFrame()
    else:
        group_cols = ["hardware.name", "hardware.num_chips", "batch_size"]
        
        # Sum vision latencies and boundness per config (sequential execution on same hardware)
        vision_sum = df_vision.groupby(group_cols)["time_ms"].sum().reset_index()
        vision_sum = vision_sum.rename(columns={"time_ms": "vision_time_ms"})
        
        # For vision boundness: join unique boundness values with "/" if multiple encoders
        vision_boundness = df_vision.groupby(group_cols)["boundness"].apply(
            lambda x: "/".join(x.unique())
        ).reset_index()
        vision_boundness = vision_boundness.rename(columns={"boundness": "vision_boundness"})
        
        # For vision op_intensity: weighted average across vision encoders
        vision_op_intensity = df_vision.groupby(group_cols).apply(
            lambda g: (g["op_intensity"] * g["time_ms"]).sum() / g["time_ms"].sum()
            if g["time_ms"].sum() > 0 else 0
        ).reset_index(name="vision_op_intensity")
        
        # Get prefill latency, boundness, and op_intensity
        prefill_times = df_prefill[group_cols + ["time_ms", "boundness", "op_intensity"]].copy()
        prefill_times = prefill_times.rename(columns={
            "time_ms": "prefill_time_ms",
            "boundness": "prefill_boundness",
            "op_intensity": "prefill_op_intensity"
        })
        
        # Get decode latency, boundness, and op_intensity (multiply by number of decode tokens)
        decode_times = df_decode[group_cols + ["time_ms", "boundness", "op_intensity"]].copy()
        decode_times = decode_times.rename(columns={
            "time_ms": "decode_time_ms",
            "boundness": "decode_boundness",
            "op_intensity": "decode_op_intensity"
        })
        decode_times["decode_time_ms"] *= OPENVLA_ACTION_TOKENS
        
        # Merge all components
        df_merged = vision_sum.merge(vision_boundness, on=group_cols, how="inner")
        df_merged = df_merged.merge(vision_op_intensity, on=group_cols, how="inner")
        df_merged = df_merged.merge(prefill_times, on=group_cols, how="inner")
        df_merged = df_merged.merge(decode_times, on=group_cols, how="inner")
        
        # Compute total E2E latency
        df_merged["e2e_time_ms"] = (
            df_merged["vision_time_ms"] +
            df_merged["prefill_time_ms"] +
            df_merged["decode_time_ms"]
        )
        
        # Add model metadata
        df_merged["model.name"] = "openvla-7b"
        df_merged["model.stage"] = "e2e"
        df_merged["model.dec_steps"] = OPENVLA_ACTION_TOKENS
        df_merged["model.seq_len_inference_prefill"] = OPENVLA_TOTAL_VISION_TOKENS
        
        # Reorder columns for consistency
        df_e2e = df_merged[[
            "model.name",
            "model.stage", 
            "model.dec_steps",
            "model.seq_len_inference_prefill",
            "hardware.name",
            "hardware.num_chips",
            "batch_size",
            "vision_time_ms",
            "prefill_time_ms",
            "decode_time_ms",
            "e2e_time_ms",
            "vision_boundness",
            "prefill_boundness",
            "decode_boundness",
            "vision_op_intensity",
            "prefill_op_intensity",
            "decode_op_intensity",
        ]]
        
        results["e2e"] = df_e2e
        df_e2e.to_csv(output_path / "openvla_e2e_perf.csv", index=False)
        logger.info(f"  -> Saved to {output_path / 'openvla_e2e_perf.csv'}")
    
    logger.info("\nOpenVLA performance evaluation complete!")
    return results


def print_summary(results: dict[str, pd.DataFrame], logger=None) -> None:
    """Print a summary of the OpenVLA performance results as a table."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"OpenVLA model characteristics:")
    logger.info(f"  - Model used: DINOv2 ViT-L/14 + SigLIP SoViT-400m/14, and Llama 2 7B")
    logger.info(f"  - Total vision tokens (prefill length): {OPENVLA_TOTAL_VISION_TOKENS}")
    logger.info(f"  - Action tokens (decode length): {OPENVLA_ACTION_TOKENS}")

    logger.info("\n" + "=" * 150)
    logger.info("OpenVLA Performance Summary")
    logger.info("=" * 150)
    
    if "e2e" in results and not results["e2e"].empty:
        df_e2e = results["e2e"]
        
        # Table header
        logger.info("-" * 150)
        logger.info(f"{'Hardware':<15} {'Chips':<6} {'Batch':<6} {'Vision':>28} {'Prefill':>28} {'Decode':>28} {'E2E (ms)':>12} {'Hz':>12}")
        logger.info(f"{'':15} {'':6} {'':6} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'':12} {'':12}")
        logger.info("-" * 150)
        
        # Table rows
        for _, row in df_e2e.iterrows():
            hw = row["hardware.name"]
            chips = int(row["hardware.num_chips"])
            batch = int(row["batch_size"])
            vision = row["vision_time_ms"]
            prefill = row["prefill_time_ms"]
            decode = row["decode_time_ms"]
            e2e = row["e2e_time_ms"]
            hz = 1000 / e2e if e2e > 0 else 0
            
            vision_bound = row.get("vision_boundness", "N/A")
            prefill_bound = row.get("prefill_boundness", "N/A")
            decode_bound = row.get("decode_boundness", "N/A")
            vision_oi = row.get("vision_op_intensity", 0)
            prefill_oi = row.get("prefill_op_intensity", 0)
            decode_oi = row.get("decode_op_intensity", 0)
            
            # Format with fixed width for alignment
            vision_str = f"{vision:6.2f}/{vision_bound:>4}/{vision_oi:6.1f}"
            prefill_str = f"{prefill:6.2f}/{prefill_bound:>4}/{prefill_oi:6.1f}"
            decode_str = f"{decode:6.2f}/{decode_bound:>4}/{decode_oi:6.1f}"
            
            logger.info(f"{hw:<15} {chips:<6} {batch:<6} {vision_str:>28} {prefill_str:>28} {decode_str:>28} {e2e:>12.2f} {hz:>12.1f}")
        
        logger.info("-" * 150)
    else:
        logger.warning("No E2E results available to display.")


if __name__ == "__main__":
    # Set up logging
    logger = setup_logging("perf_results/openvla_perf.log")
    logger.info("=" * 80)
    logger.info("Starting OpenVLA Performance Evaluation")
    logger.info("=" * 80)
    
    # Default configuration
    system_list = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"
    
    logger.info(f"Systems: {system_list}")
    logger.info(f"Number of devices: {num_device_list}")
    logger.info(f"Bits: {bits}")
    
    # Run end-to-end performance evaluation
    results = get_openvla_e2e_perf(
        system_list=system_list,
        num_device_list=num_device_list,
        bits=bits,
        logger=logger,
    )
    
    # Print summary
    print_summary(results, logger=logger)
    
    logger.info("=" * 80)
    logger.info("Performance evaluation completed")
    logger.info("=" * 80)

