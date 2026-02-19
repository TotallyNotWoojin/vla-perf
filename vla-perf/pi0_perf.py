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
Pi0 Family Performance Evaluation Script

Models from Physical Intelligence:
- π0 (pi0): PaliGemma-based (SigLIP + Gemma 2B) + 300M Action Expert

Key difference from OpenVLA:
- Uses Flow Matching (continuous diffusion) for action prediction
- Action Expert (DiT) runs N denoising iterations (typically 10-50 steps)
- Each denoising step is a forward pass through the DiT

Performance modeling breakdown:
1. Vision encoding (SigLIP SoViT-400m) - encoder prefill
2. VLM prefill (Gemma) - processes visual + text tokens
3. Action Expert (DiT) - N denoising iterations, each is a prefill-style pass

Reference files:
    genz/GenZ/Models/Model_sets/vla_models.py -> Pi0 model configs
    genz_scripts/perf_utils.py -> shared performance utilities
"""

import pandas as pd
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import copy

from GenZ.Models.default_models import ModelConfig, MODEL_DICT
from Systems.system_configs import system_configs

from perf_utils import (
    get_powers_of_two_up_to,
    get_optimal_df,
    collect_prefill_perf,
    collect_parallel_decode_perf,
    collect_decode_perf,
    RESULT_COLUMNS,
    setup_logging,
    calculate_kv_cache_size_mb,
    calculate_transformer_params, 
    format_param_count,
    is_power_of_two,
)

# Network latency imports
from network_latency import (
    NetworkConfig,
    ImageConfig,
    ActionConfig,
    VLMKVCacheConfig,
    estimate_image_latency,
    estimate_action_latency,
    estimate_kvcache_latency,
    compute_network_throughput_hz,
    get_kvcache_configs_from_model_config,
    ALL_WIFI_CONFIGS,
    ALL_DATACENTER_CONFIGS,
    ALL_CLOUD_CONFIGS,
    ALL_NETWORK_CONFIGS,
    WIFI_6_CONFIG,
    WIFI_7_CONFIG,
    ETHERNET_1G_CONFIG,
    ETHERNET_10G_CONFIG,
    CELL_4G_LTE_CONFIG,
    CELL_5G_SA_CONFIG,
    CLOUD_FAST_CONFIG,
    CLOUD_SLOW_CONFIG,
)


# ==============================================================================
# Pi0 Architecture Constants
# Pi0 inference pipeline: Vision (SigLIP) -> VLM (Gemma) -> Action Expert (DiT)
# ==============================================================================
PI0_VISION_TOKENS = 256  # SigLIP SoViT-400m outputs 256 tokens per image
PI0_VISION_FRAMES = 3 # Number of cameras (multi-view input)
PI0_LANGUAGE_TOKENS = 32 # Number of language instruction tokens
PI0_VLM_SEQUENCE_LENGTH = PI0_VISION_TOKENS * PI0_VISION_FRAMES + PI0_LANGUAGE_TOKENS # Total prefill length
PI0_ACTION_CHUNK_SIZE = 50  # Number of future actions predicted at once
PI0_DEFAULT_DENOISING_STEPS = 10  # Default number of flow matching denoising iterations
PI0_ACTION_DOF = 14  # Degrees of freedom (two 7-DoF arms)



@dataclass
class Pi0Config:
    """Configuration for a Pi0-family model."""
    name: str
    vision_model: str  # Vision encoder model name
    vlm_model: str  # VLM backbone model name
    action_expert_model: str  # Action expert (DiT) model name
    vision_tokens: int = PI0_VISION_TOKENS
    vision_frames: int = PI0_VISION_FRAMES
    language_tokens: int = PI0_LANGUAGE_TOKENS
    vlm_sequence_length: int = PI0_VLM_SEQUENCE_LENGTH  # VLM context/KV cache size for action expert
    action_chunk_size: int = PI0_ACTION_CHUNK_SIZE
    denoising_steps: int = PI0_DEFAULT_DENOISING_STEPS
    action_dof: int = PI0_ACTION_DOF  # Degrees of freedom for actions


# ==============================================================================
# Pi0 Family Model Configurations
# Each config maps to model definitions in genz/GenZ/Models/Model_sets/vla_models.py
# ==============================================================================
PI0_CONFIG = Pi0Config(
    name="pi0",
    vision_model="pi0-vision",
    vlm_model="pi0-vlm",
    action_expert_model="pi0-action-expert",
)

PI0_6_CONFIG = Pi0Config(
    name="pi0.6",
    vision_model="pi0.6-vision",
    vlm_model="pi0.6-vlm",
    action_expert_model="pi0.6-action-expert",  # Larger action expert
)

# Do not use Pi0.6 here because we don't know its exact action expert parameters
ALL_PI0_CONFIGS = [PI0_CONFIG] #, PI0_6_CONFIG]


# ==============================================================================
# Component-level performance functions
# Each evaluates one stage of the Pi0 pipeline independently.
# ==============================================================================

def get_pi0_vision_perf(
    config: Pi0Config,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
) -> pd.DataFrame:
    """
    Evaluate Pi0 vision encoder (SigLIP SoViT-400m).
    """
    results = []
    
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model=config.vision_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.vision_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
                batch_size_multiplier=config.vision_frames,  # Process multiple camera views per inference
            )
            # Tag results with pi0 model name
            # Note: batch_size in results is already the logical batch size (handled by collect_prefill_perf)
            for r in model_results:
                r["model.name"] = f"{config.name}/vision"
            results.extend(model_results)
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    return get_optimal_df(df, apply_pareto=True) if not df.empty else df


def get_pi0_vlm_perf(
    config: Pi0Config,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
) -> pd.DataFrame:
    """
    Evaluate Pi0 VLM backbone (Gemma variants).
    
    The VLM processes visual tokens from the vision encoder.
    """
    results = []
    
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model=config.vlm_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.vlm_sequence_length,  # Full sequence: vision tokens + language tokens
                bits=bits,
                max_batch_size=max_batch_size,
            )
            # Tag results with pi0 model name
            for r in model_results:
                r["model.name"] = f"{config.name}/vlm"
            results.extend(model_results)
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    return get_optimal_df(df, apply_pareto=True) if not df.empty else df


def get_pi0_action_expert_perf(
    config: Pi0Config,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    denoising_steps: Optional[int] = None,
    vlm_sequence_length: Optional[int] = None,
    action_chunk_size: Optional[int] = None,
    max_batch_size: int = 1024,
) -> pd.DataFrame:
    """
    Evaluate Pi0 Action Expert (DiT) for flow matching using parallel decode.
    
    The action expert:
    - Attends to VLM KV cache (vlm_sequence_length tokens)
    - Generates action tokens in parallel (action_chunk_size tokens)
    - Runs N denoising iterations (each iteration is a parallel decode pass)
    - Total latency = single_pass_latency * denoising_steps
    
    Args:
        config: Pi0Config for the model
        system_list: List of hardware systems to evaluate
        num_device_list: List of device counts to test
        bits: Precision
        denoising_steps: Number of flow matching denoising steps
        vlm_sequence_length: VLM context/KV cache size (defaults to vision_tokens)
        action_chunk_size: Action chunk size (defaults to config.action_chunk_size)
        max_batch_size: Maximum batch size to test
        
    Returns:
        DataFrame with action expert performance results
    """
    if denoising_steps is None:
        denoising_steps = config.denoising_steps
    
    if vlm_sequence_length is None:
        # Default to vision tokens (can be extended with text tokens)
        vlm_sequence_length = config.vision_tokens
    
    if action_chunk_size is None:
        action_chunk_size = config.action_chunk_size
    
    results = []
    
    for system in system_list:
        for num_devices in num_device_list:
            # Get single-step latency using parallel decode
            model_results = collect_parallel_decode_perf(
                model=config.action_expert_model,
                system=system,
                num_devices=num_devices,
                input_tokens=vlm_sequence_length,  # VLM KV cache size
                output_tokens_parallel=action_chunk_size,  # Action tokens in parallel
                self_attention=True,  # Diffusion models use self-attention
                bits=bits,
                max_batch_size=max_batch_size,
            )
            
            # Multiply by denoising steps for total action expert time
            for r in model_results:
                r["model.name"] = f"{config.name}/action-expert"
                r["time_ms"] = r["time_ms"] * denoising_steps
            
            results.extend(model_results)
    
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    return get_optimal_df(df, apply_pareto=True) if not df.empty else df


# ==============================================================================
# End-to-end performance: combines vision + VLM + action expert latencies
# ==============================================================================

def check_model_fits_system(
    vision_model: str,
    vlm_model: str, 
    action_model: str,
    system: str,
    num_devices: int,
    bits: str,
    vlm_sequence_length: int,
    logger=None
) -> tuple[bool, float, float]:
    """
    Check if the total model size (vision + vlm + action) fits in system memory.
    
    Args:
        vision_model: Vision model name
        vlm_model: VLM model name
        action_model: Action expert model name
        system: System name (e.g., "Jetson_AGX_Thor")
        num_devices: Number of devices (for tensor/pipeline parallelism)
        bits: Precision (e.g., "bf16", "int8")
        vlm_sequence_length: Sequence length for VLM (affects KV cache size)
        logger: Logger instance (optional)
        
    Returns:
        Tuple of (fits: bool, total_memory_gb: float, system_capacity_gb: float)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get system memory capacity
    if system not in system_configs:
        logger.warning(f"System {system} not found in system_configs")
        return False, 0.0, 0.0
    
    system_capacity_gb = system_configs[system].get('Memory_size', 0)
    
    # Calculate memory requirement for each component
    # Bytes per parameter based on precision
    bytes_per_param = {
        'fp32': 4, 'f32': 4, 'tf32': 4,
        'bf16': 2, 'fp16': 2,
        'fp8': 1, 'int8': 1,
        'fp6': 0.75,
        'fp4': 0.5, 'int4': 0.5,
        'int2': 0.25
    }
    
    if bits not in bytes_per_param:
        logger.warning(f"Unknown precision {bits}, assuming bf16")
        bits = 'bf16'
    
    bytes_per_element = bytes_per_param[bits]
    
    # Calculate parameter counts and memory
    vision_config = MODEL_DICT.get_model(vision_model)
    vlm_config = MODEL_DICT.get_model(vlm_model)
    action_config = MODEL_DICT.get_model(action_model)
    
    vision_params = calculate_transformer_params(vision_config)
    vlm_params = calculate_transformer_params(vlm_config)
    action_params = calculate_transformer_params(action_config)
    
    # Model weights memory (in GB)
    vision_memory_gb = (vision_params * bytes_per_element) / (1024**3)
    vlm_memory_gb = (vlm_params * bytes_per_element) / (1024**3)
    action_memory_gb = (action_params * bytes_per_element) / (1024**3)
    
    # KV cache memory (in GB) - for VLM with batch_size=1
    vlm_kv_cache_gb = calculate_kv_cache_size_mb(
        model_name=vlm_model,
        seq_length=vlm_sequence_length,
        bits=bits,
    ) / 1024.0
    
    # Total memory requirement
    total_memory_gb = vision_memory_gb + vlm_memory_gb + action_memory_gb + vlm_kv_cache_gb
    
    # For tensor/pipeline parallelism, memory is split across devices
    # For single device (num_devices=1), full memory is needed per chip
    memory_per_chip_gb = total_memory_gb / num_devices if num_devices > 1 else total_memory_gb
    
    fits = memory_per_chip_gb <= system_capacity_gb
    
    if not fits:
        logger.info(
            f"  Model does not fit on {system}: "
            f"requires {memory_per_chip_gb:.2f} GB per chip, "
            f"system has {system_capacity_gb:.2f} GB"
        )
    
    return fits, memory_per_chip_gb, system_capacity_gb


def get_pi0_e2e_perf(
    config: Pi0Config,
    system_list: list[str] = ["A100_80GB", "H100", "B100"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    denoising_steps: Optional[int] = None,
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate end-to-end performance for a Pi0-family model.
    
    Args:
        config: Pi0Config for the model variant
        system_list: Hardware systems to evaluate
        num_device_list: Device counts to test
        bits: Precision (bf16, int8, etc.)
        denoising_steps: Number of flow matching denoising steps
        output_dir: Directory to save results
        logger: Logger instance (optional)
        
    Returns:
        Dictionary of DataFrames for each component (vision, vlm, action_expert, e2e)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)
    
    if denoising_steps is None:
        denoising_steps = config.denoising_steps
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    model_name = config.name
    
    # Filter system_list to only include systems where the model fits
    # Check for single chip case (most restrictive)
    filtered_system_list = []
    for system in system_list:
        fits, memory_required, system_capacity = check_model_fits_system(
            vision_model=config.vision_model,
            vlm_model=config.vlm_model,
            action_model=config.action_expert_model,
            system=system,
            num_devices=1,  # Check single chip case (most restrictive)
            bits=bits,
            vlm_sequence_length=config.vlm_sequence_length,
            logger=logger
        )
        if fits:
            filtered_system_list.append(system)
        else:
            logger.info(
                f"Skipping {system} for {model_name}: "
                f"model requires {memory_required:.2f} GB, "
                f"system has {system_capacity:.2f} GB"
            )
    
    if not filtered_system_list:
        logger.warning(f"No systems can fit {model_name}. Returning empty results.")
        return {
            "vision": pd.DataFrame(),
            "vlm": pd.DataFrame(), 
            "action_expert": pd.DataFrame(),
            "e2e": pd.DataFrame()
        }
    
    # Use filtered system list for evaluation
    system_list = filtered_system_list
    
    # 1. Vision encoder performance
    logger.info(f"Evaluating {model_name} vision encoder...")
    df_vision = get_pi0_vision_perf(config, system_list, num_device_list, bits)
    results["vision"] = df_vision
    
    # 2. VLM backbone performance
    logger.info(f"Evaluating {model_name} VLM backbone...")
    df_vlm = get_pi0_vlm_perf(config, system_list, num_device_list, bits)
    results["vlm"] = df_vlm
    
    # 3. Action expert performance
    logger.info(f"Evaluating {model_name} action expert ({denoising_steps} denoising steps)...")
    df_action = get_pi0_action_expert_perf(
        config, system_list, num_device_list, bits, denoising_steps,
        vlm_sequence_length=config.vlm_sequence_length
    )
    results["action_expert"] = df_action
    
    # 4. Compute E2E latency by combining component latencies
    logger.info(f"Computing {model_name} end-to-end latency...")
    
    if df_vision.empty or df_vlm.empty or df_action.empty:
        logger.warning("One or more component DataFrames are empty. Skipping E2E computation.")
        results["e2e"] = pd.DataFrame()
    else:
        hw_group_cols = ["hardware.name", "hardware.num_chips"]
        group_cols = hw_group_cols + ["batch_size"]
        
        # Extract latencies, boundness, and op_intensity
        vision_times = df_vision[group_cols + ["time_ms", "boundness", "op_intensity"]].copy()
        vision_times = vision_times.rename(columns={
            "time_ms": "vision_time_ms", 
            "boundness": "vision_boundness",
            "op_intensity": "vision_op_intensity"})
        
        vlm_times = df_vlm[group_cols + ["time_ms", "boundness", "op_intensity"]].copy()
        vlm_times = vlm_times.rename(columns={
            "time_ms": "vlm_time_ms", 
            "boundness": "vlm_boundness",
            "op_intensity": "vlm_op_intensity"})
        
        action_times = df_action[group_cols + ["time_ms", "boundness", "op_intensity"]].copy()
        action_times = action_times.rename(columns={
            "time_ms": "action_time_ms",
            "boundness": "action_boundness",
            "op_intensity": "action_op_intensity"
        })
        
        # Pad dataframes to have all batch sizes for each hardware config
        # If a component can't handle larger batch sizes, scale latency linearly
        def pad_batch_sizes(df, time_col, boundness_col, op_intensity_col):
            """
            Pad dataframe with missing batch sizes by scaling from available batch sizes.
            Uses the batch size that gives minimal scaled latency.
            """
            padded_rows = []
            
            for (hw_name, num_chips), hw_df in df.groupby(hw_group_cols):
                existing_batch_sizes = set(hw_df["batch_size"])
                
                # Sanity check: all batch sizes must be powers of two
                for bs in existing_batch_sizes:
                    if not is_power_of_two(bs):
                        logger.warning(f"Batch size {bs} is not a power of two for "
                                     f"{hw_name} with {num_chips} chips")
                
                # Find all batch sizes across all components for this hardware
                all_batch_sizes = set()
                for comp_df in [vision_times, vlm_times, action_times]:
                    hw_comp = comp_df[(comp_df["hardware.name"] == hw_name) & 
                                     (comp_df["hardware.num_chips"] == num_chips)]
                    all_batch_sizes.update(hw_comp["batch_size"])
                
                # Sanity check: all target batch sizes must be powers of two
                for bs in all_batch_sizes:
                    if not is_power_of_two(bs):
                        logger.warning(f"Target batch size {bs} is not a power of two")
                
                # For each missing batch size, find the optimal base for scaling
                if existing_batch_sizes:
                    for target_batch_size in all_batch_sizes:
                        if target_batch_size not in existing_batch_sizes:
                            # Try all available batch sizes and pick the one with minimal scaled latency
                            min_scaled_time = float('inf')
                            best_boundness = None
                            best_op_intensity = None
                            
                            for base_batch_size in existing_batch_sizes:
                                if base_batch_size <= target_batch_size:
                                    base_row = hw_df[hw_df["batch_size"] == base_batch_size].iloc[0]
                                    base_time = base_row[time_col]
                                    base_boundness = base_row[boundness_col]
                                    base_op_intensity = base_row[op_intensity_col]
                                    
                                    # Linear scaling: larger batch = multiple sequential runs
                                    scale_factor = target_batch_size / base_batch_size
                                    scaled_time = base_time * scale_factor
                                    
                                    if scaled_time < min_scaled_time:
                                        min_scaled_time = scaled_time
                                        best_boundness = base_boundness
                                        best_op_intensity = base_op_intensity
                            
                            if min_scaled_time < float('inf'):
                                padded_rows.append({
                                    "hardware.name": hw_name,
                                    "hardware.num_chips": num_chips,
                                    "batch_size": target_batch_size,
                                    time_col: min_scaled_time,
                                    boundness_col: best_boundness,
                                    op_intensity_col: best_op_intensity,
                                })
            
            if padded_rows:
                df_padded = pd.concat([df, pd.DataFrame(padded_rows)], ignore_index=True)
                return df_padded
            return df
        
        vision_times = pad_batch_sizes(vision_times, "vision_time_ms", "vision_boundness", "vision_op_intensity")
        vlm_times = pad_batch_sizes(vlm_times, "vlm_time_ms", "vlm_boundness", "vlm_op_intensity")
        action_times = pad_batch_sizes(action_times, "action_time_ms", "action_boundness", "action_op_intensity")
        
        # Merge all components (now all should have matching batch sizes)
        df_merged = vision_times.merge(vlm_times, on=group_cols, how="inner")
        df_merged = df_merged.merge(action_times, on=group_cols, how="inner")
        
        # Compute total E2E latency
        df_merged["e2e_time_ms"] = (
            df_merged["vision_time_ms"] +
            df_merged["vlm_time_ms"] +
            df_merged["action_time_ms"]
        )
        
        # Add model metadata
        df_merged["model.name"] = model_name
        df_merged["model.stage"] = "e2e"
        df_merged["model.seq_len_inference_prefill"] = PI0_VISION_TOKENS
        
        # Reorder columns
        df_e2e = df_merged[[
            "model.name",
            "model.stage",
            "model.seq_len_inference_prefill",
            "hardware.name",
            "hardware.num_chips",
            "batch_size",
            "vision_time_ms",
            "vlm_time_ms",
            "action_time_ms",
            "e2e_time_ms",
            "vision_boundness",
            "vlm_boundness",
            "action_boundness",
            "vision_op_intensity",
            "vlm_op_intensity",
            "action_op_intensity",
        ]]
        
        results["e2e"] = df_e2e
    
    return results


def get_all_pi0_perf(
    system_list: list[str] = ["Jetson_AGX_Thor", "RTX_4090", "A100_80GB", "H100", "B100"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    denoising_steps: int = PI0_DEFAULT_DENOISING_STEPS,
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Evaluate all Pi0 family models (pi0, pi0.5, pi0.6).
    
    Args:
        experiment_num: Experiment number for logging (optional)
    
    Returns:
        Nested dict: {model_name: {component: DataFrame}}
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    all_e2e_results = []
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}PI0 FAMILY PERFORMANCE EVALUATION (MULTI-GPU, ALLOW BATCHING)")
    logger.info("=" * 150)
    
    for config in ALL_PI0_CONFIGS:
        logger.info(f"\n{'-'*60}")
        logger.info(f"Evaluating {config.name}")
        logger.info(f"{'-'*60}")
        
        results = get_pi0_e2e_perf(
            config=config,
            system_list=system_list,
            num_device_list=num_device_list,
            bits=bits,
            denoising_steps=denoising_steps,
            output_dir=output_dir,
            logger=logger,
        )
        
        all_results[config.name] = results
        
        if not results["e2e"].empty:
            all_e2e_results.append(results["e2e"])
    
    # Save combined E2E results
    if all_e2e_results:
        df_combined = pd.concat(all_e2e_results, ignore_index=True)
        output_file = output_path / "pi0_family_e2e_perf.csv"
        df_combined.to_csv(output_file, index=False)
        logger.info(f"\nCombined E2E results saved to {output_file}")
    
    return all_results


def print_all_pi0_perf_summary(all_results: dict[str, dict[str, pd.DataFrame]], logger=None) -> None:
    """Print a summary of Pi0 family performance results in table format."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "-" * 100)
    logger.info("Pi0 Family Performance Summary")
    logger.info("-" * 100)
    
    for model_name, results in all_results.items():
        if "e2e" not in results or results["e2e"].empty:
            continue
            
        logger.info(f"\n{model_name}:")
        df_e2e = results["e2e"]
        
        # Collect batch_size=1, num_chips=1 rows for hardware comparison
        hw_comparison_rows = []
        
        for hw in df_e2e["hardware.name"].unique():
            hw_df = df_e2e[df_e2e["hardware.name"] == hw].sort_values(["hardware.num_chips", "batch_size"])
            
            logger.info(f"\n  {hw}:")
            logger.info("-" * 150)
            logger.info(f"{'Chips':<8} {'Batch':<8} {'Vision':>28} {'VLM':>28} {'Action':>28} {'E2E (ms)':>12} {'Freq (Hz)':>12}")
            logger.info(f"{'':8} {'':8} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'(ms/bound/OI,10x)':>28} {'':12} {'':12}")
            logger.info("-" * 150)
            
            for _, row in hw_df.iterrows():
                chips = int(row["hardware.num_chips"])
                batch = int(row["batch_size"])
                e2e = row["e2e_time_ms"]
                vision = row["vision_time_ms"]
                vlm = row["vlm_time_ms"]
                action = row["action_time_ms"]
                vision_bound = row.get("vision_boundness", "N/A")
                vlm_bound = row.get("vlm_boundness", "N/A")
                action_bound = row.get("action_boundness", "N/A")
                vision_oi = row.get("vision_op_intensity", 0)
                vlm_oi = row.get("vlm_op_intensity", 0)
                action_oi = row.get("action_op_intensity", 0)
                
                # Format with fixed width for alignment
                vision_str = f"{vision:6.2f}/{vision_bound:>4}/{vision_oi:6.1f}"
                vlm_str = f"{vlm:6.2f}/{vlm_bound:>4}/{vlm_oi:6.1f}"
                action_str = f"{action:6.2f}/{action_bound:>4}/{action_oi:6.1f}"
                
                logger.info(f"{chips:<8} {batch:<8} {vision_str:>28} {vlm_str:>28} {action_str:>28} {e2e:>12.2f} {1000/e2e:>12.1f}")
                
                # Collect row for batch_size=1, num_chips=1 comparison
                if chips == 1 and batch == 1:
                    hw_comparison_rows.append({
                        "hardware": hw,
                        "vision": vision,
                        "vlm": vlm,
                        "action": action,
                        "e2e": e2e,
                        "vision_bound": vision_bound,
                        "vlm_bound": vlm_bound,
                        "action_bound": action_bound,
                        "vision_oi": vision_oi,
                        "vlm_oi": vlm_oi,
                        "action_oi": action_oi,
                    })
        
        # Print hardware comparison for batch_size=1, num_chips=1
        if hw_comparison_rows:
            logger.info(f"\n  Hardware Comparison (Batch=1, Chips=1):")
            logger.info("-" * 150)
            logger.info(f"{'Hardware':<20} {'Vision':>28} {'VLM':>28} {'Action':>28} {'E2E (ms)':>12} {'Freq (Hz)':>12}")
            logger.info(f"{'':20} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'(ms/bound/OI,10x)':>28} {'':12} {'':12}")
            logger.info("-" * 150)
            
            for row in hw_comparison_rows:
                hw_name = row["hardware"]
                vision = row["vision"]
                vlm = row["vlm"]
                action = row["action"]
                e2e = row["e2e"]
                vision_bound = row["vision_bound"]
                vlm_bound = row["vlm_bound"]
                action_bound = row["action_bound"]
                vision_oi = row["vision_oi"]
                vlm_oi = row["vlm_oi"]
                action_oi = row["action_oi"]
                
                # Format with fixed width for alignment
                vision_str = f"{vision:6.2f}/{vision_bound:>4}/{vision_oi:6.1f}"
                vlm_str = f"{vlm:6.2f}/{vlm_bound:>4}/{vlm_oi:6.1f}"
                action_str = f"{action:6.2f}/{action_bound:>4}/{action_oi:6.1f}"
                
                logger.info(f"{hw_name:<20} {vision_str:>28} {vlm_str:>28} {action_str:>28} {e2e:>12.2f} {1000/e2e:>12.1f}")


# ==============================================================================
# Experiment 2: Model Size Scaling
# Creates hypothetical Pi0 variants (pi-0, pi-0-L, pi-0-XL, pi-0-XXL) using
# progressively larger VLM backbones to study how model size affects latency.
# ==============================================================================

def create_action_expert_config_from_vlm(vlm_config: ModelConfig, name: str) -> ModelConfig:
    """
    Create an action expert config from a VLM config.
    Action expert is 4-8x smaller than VLM (hidden_size /= 2, intermediate_size /= 4).
    
    Args:
        vlm_config: VLM model config to base the action expert on
        name: Name for the action expert model
        
    Returns:
        ModelConfig for the action expert
    """
    action_config = copy.deepcopy(vlm_config)
    action_config.model = name
    action_config.hidden_size = vlm_config.hidden_size // 2
    action_config.intermediate_size = vlm_config.intermediate_size // 4
    action_config.vocab_size = 0  # Action expert doesn't need vocab
    
    return action_config


def get_model_size_scaling_perf(
    system_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    denoising_steps: int = PI0_DEFAULT_DENOISING_STEPS,
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Pi0Config]]:
    """
    Evaluate Pi0 performance scaling with different model sizes.
    
    Tests four model variants:
    - pi-0: siglip2_so400m + gemma_2b + action expert (4-8x smaller)
    - pi-0-L: siglip2_g (1B) + llama2_7b + action expert (4-8x smaller)
    - pi-0-XL: siglip2_g (1B) + llama2_13b + action expert (4-8x smaller)
    - pi-0-XXL: siglip2_g (1B) + llama2_70b + action expert (4-8x smaller)
    
    Args:
        system_list: Hardware systems to evaluate
        num_device_list: Device counts to test (default: [1, 2, 4])
        bits: Precision (bf16, int8, etc.)
        denoising_steps: Number of flow matching denoising steps
        output_dir: Directory to save results
        experiment_num: Experiment number for logging (optional)
        logger: Logger instance (optional)
        
    Returns:
        Tuple of (results dict, configs dict):
        - results: {model_name: {component: DataFrame}}
        - configs: {model_name: Pi0Config}
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if num_device_list is None:
        num_device_list = [1]  # Test with single chip for fair comparison
    
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create model variants
    model_variants = []
    
    # pi-0 (default): siglip2_so400m + gemma_2b
    vlm_pi0 = copy.deepcopy(MODEL_DICT.get_model("gemma-2b"))
    vlm_pi0.vocab_size = 0  # Avoid counting final token prediction latency
    vlm_pi0.model = "gemma-2b-no-vocab"
    action_pi0 = create_action_expert_config_from_vlm(vlm_pi0, "vla/pi0-size-scaling-action-2b")
    # Register the model
    MODEL_DICT.add_model(action_pi0)
    MODEL_DICT.add_model(vlm_pi0)
    
    pi0_config = Pi0Config(
        name="pi-0",
        vision_model="siglip2-so400m-patch14-384-vision",
        vlm_model=vlm_pi0.model,
        action_expert_model=action_pi0.model,
    )
    model_variants.append(pi0_config)
    
    # pi-0-L: siglip2_g + llama2_7b
    vlm_l = copy.deepcopy(MODEL_DICT.get_model("llama2_7b"))
    vlm_l.vocab_size = 0  # Avoid counting final token prediction latency
    vlm_l.model = "llama2_7b-no-vocab"
    action_l = create_action_expert_config_from_vlm(vlm_l, "vla/pi0-size-scaling-action-7b")
    MODEL_DICT.add_model(action_l)
    MODEL_DICT.add_model(vlm_l)
    
    pi0_l_config = Pi0Config(
        name="pi-0-L",
        vision_model="siglip2-giant-opt-patch16-384-vision",
        vlm_model=vlm_l.model,
        action_expert_model=action_l.model,
    )
    model_variants.append(pi0_l_config)
    
    # pi-0-XL: siglip2_g + llama2_13b
    vlm_xl = copy.deepcopy(MODEL_DICT.get_model("llama2_13b"))
    vlm_xl.vocab_size = 0  # Avoid counting final token prediction latency
    vlm_xl.model = "llama2_13b-no-vocab"
    action_xl = create_action_expert_config_from_vlm(vlm_xl, "vla/pi0-size-scaling-action-13b")
    MODEL_DICT.add_model(action_xl)
    MODEL_DICT.add_model(vlm_xl)

    pi0_xl_config = Pi0Config(
        name="pi-0-XL",
        vision_model="siglip2-giant-opt-patch16-384-vision",
        vlm_model=vlm_xl.model,
        action_expert_model=action_xl.model,
    )
    model_variants.append(pi0_xl_config)
    
    # pi-0-XXL: siglip2_g + llama2_70b
    vlm_xxl = copy.deepcopy(MODEL_DICT.get_model("llama2_70b"))
    vlm_xxl.vocab_size = 0  # Avoid counting final token prediction latency
    vlm_xxl.model = "llama2_70b-no-vocab"
    action_xxl = create_action_expert_config_from_vlm(vlm_xxl, "vla/pi0-size-scaling-action-70b")
    MODEL_DICT.add_model(action_xxl)
    MODEL_DICT.add_model(vlm_xxl)

    pi0_xxl_config = Pi0Config(
        name="pi-0-XXL",
        vision_model="siglip2-giant-opt-patch16-384-vision",
        vlm_model=vlm_xxl.model,
        action_expert_model=action_xxl.model,
    )
    model_variants.append(pi0_xxl_config)
    
    # Calculate and save parameter counts
    logger.info("\n" + "-" * 150)
    logger.info("CALCULATING MODEL PARAMETER COUNTS")
    logger.info("-" * 150)
    
    # Component parameter data
    component_params_data = []
    model_total_params_data = []
    
    for config in model_variants:
        # Get model configs
        vision_config = MODEL_DICT.get_model(config.vision_model)
        vlm_config = MODEL_DICT.get_model(config.vlm_model)
        action_config = MODEL_DICT.get_model(config.action_expert_model)
        
        # Calculate parameters
        vision_params = calculate_transformer_params(vision_config)
        
        # For VLM, get size without vocab table
        vlm_params = calculate_transformer_params(vlm_config)
        
        action_params = calculate_transformer_params(action_config)
        
        total_params = vision_params + vlm_params + action_params
        
        # Store component params
        component_params_data.append({
            "model": config.name,
            "vision_model": config.vision_model,
            "vision_params_M": vision_params / 1e6,
            "vlm_model": config.vlm_model,
            "vlm_params_M": vlm_params / 1e6,
            "action_model": config.action_expert_model,
            "action_params_M": action_params / 1e6,
        })
        
        # Store total model params
        model_total_params_data.append({
            "model": config.name,
            "total_params_M": total_params / 1e6,
            "vision_params_M": vision_params / 1e6,
            "vlm_params_M": vlm_params / 1e6,
            "action_params_M": action_params / 1e6,
        })
        
        # Print sizes
        logger.info(f"\n{config.name}:")
        logger.info(f"  Vision ({config.vision_model}): {format_param_count(vision_params)} ({vision_params / 1e6:.2f}M)")
        logger.info(f"  VLM ({config.vlm_model}): {format_param_count(vlm_params)} ({vlm_params / 1e6:.2f}M)")
        logger.info(f"  Action Expert ({config.action_expert_model}): {format_param_count(action_params)} ({action_params / 1e6:.2f}M)")
        logger.info(f"  Total: {format_param_count(total_params)} ({total_params / 1e6:.2f}M)")
    
    # Save component parameters to CSV
    df_component_params = pd.DataFrame(component_params_data)
    component_params_file = output_path / "pi0_model_component_params.csv"
    df_component_params.to_csv(component_params_file, index=False)
    logger.info(f"\nComponent parameter counts saved to {component_params_file}")
    
    # Save total model parameters to CSV
    df_model_params = pd.DataFrame(model_total_params_data)
    model_params_file = output_path / "pi0_model_total_params.csv"
    df_model_params.to_csv(model_params_file, index=False)
    logger.info(f"Total model parameter counts saved to {model_params_file}")
    
    # Run evaluation
    all_results = {}
    all_configs = {}
    all_e2e_results = []
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}PI0 MODEL SIZE SCALING EVALUATION")
    logger.info("-" * 150)
    logger.info("Testing performance scaling with different model sizes:")
    logger.info("  - pi-0:     SigLIP-So400m + Gemma2-2B   + Action Expert (4-8x smaller)")
    logger.info("  - pi-0-L:   SigLIP-Giant  + Llama2-7B   + Action Expert (4-8x smaller)")
    logger.info("  - pi-0-XL:  SigLIP-Giant  + Llama2-13B  + Action Expert (4-8x smaller)")
    logger.info("  - pi-0-XXL: SigLIP-Giant + Llama2-70B  + Action Expert (4-8x smaller)")
    logger.info("=" * 150)
    
    for config in model_variants:
        logger.info(f"\n{'-'*60}")
        logger.info(f"Evaluating {config.name}")
        logger.info(f"  Vision: {config.vision_model}")
        logger.info(f"  VLM: {config.vlm_model}")
        logger.info(f"  Action Expert: {config.action_expert_model}")
        logger.info(f"{'-'*60}")
        
        results = get_pi0_e2e_perf(
            config=config,
            system_list=system_list,
            num_device_list=num_device_list,
            bits=bits,
            denoising_steps=denoising_steps,
            output_dir=output_dir,
            logger=logger,
        )
        
        all_results[config.name] = results
        all_configs[config.name] = config
        
        if not results["e2e"].empty:
            all_e2e_results.append(results["e2e"])
    
    # Save combined E2E results
    if all_e2e_results:
        df_combined = pd.concat(all_e2e_results, ignore_index=True)
        output_file = output_path / "pi0_model_size_scaling.csv"
        df_combined.to_csv(output_file, index=False)
        logger.info(f"\nCombined E2E results saved to {output_file}")
    
    return all_results, all_configs


def print_model_size_scaling_summary(
    all_results: dict[str, dict[str, pd.DataFrame]], 
    model_configs: dict[str, Pi0Config] = None,
    logger=None
) -> None:
    """Print a summary of Pi0 model size scaling results.
    
    Args:
        all_results: Dictionary mapping model names to their component results
        model_configs: Dictionary mapping model names to their Pi0Config objects
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "-" * 150)
    logger.info("Pi0 Model Size Scaling Summary")
    logger.info("-" * 150)
    
    # Collect results for comparison (batch_size=1, num_chips=1)
    comparison_rows = []
    # Component data: {component_model_name: {hw: latency}}
    component_data = {"vision": {}, "vlm": {}, "action_expert": {}}
    
    for model_name, results in all_results.items():
        if "e2e" not in results or results["e2e"].empty:
            continue
        
        df_e2e = results["e2e"]
        
        # Get component model names from config if available
        vision_model_name = model_name
        vlm_model_name = model_name
        action_model_name = model_name
        
        if model_configs and model_name in model_configs:
            config = model_configs[model_name]
            vision_model_name = config.vision_model
            vlm_model_name = config.vlm_model
            action_model_name = config.action_expert_model
        
        # Get single-chip, single-batch results for each hardware
        for hw in df_e2e["hardware.name"].unique():
            hw_rows = df_e2e[
                (df_e2e["hardware.name"] == hw) & 
                (df_e2e["hardware.num_chips"] == 1) & 
                (df_e2e["batch_size"] == 1)
            ]
            
            if not hw_rows.empty:
                row = hw_rows.iloc[0]
                comparison_rows.append({
                    "model": model_name,
                    "hardware": hw,
                    "vision_ms": row["vision_time_ms"],
                    "vlm_ms": row["vlm_time_ms"],
                    "action_ms": row["action_time_ms"],
                    "e2e_ms": row["e2e_time_ms"],
                    "freq_hz": 1000.0 / row["e2e_time_ms"],
                })
        
        # Collect component-level data with actual component model names
        for component_name, component_key, component_model in [
            ("vision", "vision", vision_model_name), 
            ("vlm", "vlm", vlm_model_name), 
            ("action_expert", "action_expert", action_model_name)
        ]:
            if component_key in results and not results[component_key].empty:
                df_comp = results[component_key]
                
                # Initialize dict for this component model if not exists
                if component_model not in component_data[component_name]:
                    component_data[component_name][component_model] = {}
                
                for hw in df_comp["hardware.name"].unique():
                    hw_rows = df_comp[
                        (df_comp["hardware.name"] == hw) & 
                        (df_comp["hardware.num_chips"] == 1) & 
                        (df_comp["batch_size"] == 1)
                    ]
                    
                    if not hw_rows.empty:
                        row = hw_rows.iloc[0]
                        # Store latency for this component model and hardware
                        component_data[component_name][component_model][hw] = row["time_ms"]
    
    if not comparison_rows:
        logger.info("No results to display")
        return
    
    df_comparison = pd.DataFrame(comparison_rows)
    
    # ========== Print Component-Level Performance Tables ==========
    
    # Helper function to print component table
    def print_component_table(comp_data, title):
        if not comp_data:
            return
        
        logger.info("\n" + "-" * 120)
        logger.info(f"{title} - Single Chip, Batch Size = 1")
        logger.info("-" * 120)
        
        # Get all unique hardware platforms across all component models
        all_hardware = set()
        for model_data in comp_data.values():
            all_hardware.update(model_data.keys())
        hardware_list = sorted(all_hardware)
        
        # Get component model names
        component_models = sorted(comp_data.keys())
        
        # Print header
        header = f"{'Model':<40}"
        for hw in hardware_list:
            header += f"{hw:>15}"
        logger.info(header)
        logger.info("-" * 120)
        
        # Print rows for each component model
        for model in component_models:
            row_str = f"{model:<40}"
            for hw in hardware_list:
                if hw in comp_data[model]:
                    latency = comp_data[model][hw]
                    row_str += f"{latency:>15.2f}"
                else:
                    row_str += f"{'N/A':>15}"
            logger.info(row_str)
        logger.info("-" * 120)
    
    # 1. Vision Encoder Table
    print_component_table(component_data["vision"], "VISION ENCODER LATENCY (ms)")
    
    # 2. VLM Backbone Table
    print_component_table(component_data["vlm"], "VLM BACKBONE LATENCY (ms)")
    
    # 3. Action Expert Table
    print_component_table(component_data["action_expert"], "ACTION EXPERT LATENCY (ms)")
    
    # ========== Original E2E Performance Table ==========
    logger.info("\n" + "-" * 150)
    logger.info("END-TO-END PERFORMANCE - Single Chip, Batch Size = 1")
    logger.info("-" * 150)
    
    # Print by hardware
    for hw in df_comparison["hardware"].unique():
        hw_df = df_comparison[df_comparison["hardware"] == hw].sort_values("model")
        
        logger.info(f"\n{hw}:")
        logger.info("-" * 150)
        logger.info(f"{'Model':<12} {'Vision (ms / %)':>20} {'VLM (ms / %)':>20} {'Action (ms / %)':>20} {'E2E (ms)':>15} {'Freq (Hz)':>15}")
        logger.info("-" * 150)
        
        for _, row in hw_df.iterrows():
            # Calculate percentages
            vision_pct = (row['vision_ms'] / row['e2e_ms']) * 100 if row['e2e_ms'] > 0 else 0
            vlm_pct = (row['vlm_ms'] / row['e2e_ms']) * 100 if row['e2e_ms'] > 0 else 0
            action_pct = (row['action_ms'] / row['e2e_ms']) * 100 if row['e2e_ms'] > 0 else 0
            
            logger.info(f"{row['model']:<12} "
                       f"{row['vision_ms']:>8.2f} / {vision_pct:>5.1f}%  "
                       f"{row['vlm_ms']:>8.2f} / {vlm_pct:>5.1f}%  "
                       f"{row['action_ms']:>8.2f} / {action_pct:>5.1f}%  "
                       f"{row['e2e_ms']:>15.2f} {row['freq_hz']:>15.2f}")
        
        logger.info("-" * 150)
    
    # Print scaling comparison (relative to pi-0)
    logger.info("\n" + "-" * 150)
    logger.info("Relative Performance (normalized to pi-0)")
    logger.info("-" * 150)
    
    for hw in df_comparison["hardware"].unique():
        hw_df = df_comparison[df_comparison["hardware"] == hw].sort_values("model")
        
        # Get pi-0 baseline
        pi0_row = hw_df[hw_df["model"] == "pi-0"]
        if pi0_row.empty:
            continue
        
        pi0_e2e = pi0_row.iloc[0]["e2e_ms"]
        
        logger.info(f"\n{hw}:")
        logger.info("-" * 100)
        logger.info(f"{'Model':<12} {'E2E (ms)':>12} {'Relative Latency':>18} {'Relative Throughput':>22}")
        logger.info("-" * 100)
        
        for _, row in hw_df.iterrows():
            rel_latency = row['e2e_ms'] / pi0_e2e
            rel_throughput = pi0_e2e / row['e2e_ms']
            logger.info(f"{row['model']:<12} {row['e2e_ms']:>12.2f} {rel_latency:>17.2f}x "
                       f"{rel_throughput:>20.2f}x")
        
        logger.info("-" * 100)


# ==============================================================================
# Experiment 3: Long Context
# Tests how latency grows as the robot accumulates more observation history
# (more timesteps = longer KV cache for the action expert).
# ==============================================================================

def run_long_context_experiment(
    config: Pi0Config = PI0_CONFIG,
    systems: list[str] = ["Jetson_AGX_Thor", "RTX_4090", "A100_80GB", "H100", "B100"],
    timestep_counts: list[int] = [1, 10, 100, 1000, 10000],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate Pi0 performance with longer context (multiple timesteps).
    
    Each timestep = 3 frames (as defined by PI0_VISION_FRAMES).
    
    Models incremental updates where:
    - timesteps=1: Regular inference (vision + vlm prefill + action)
    - timesteps=N (N>1): Previous N-1 timesteps already in KV cache, so:
      1. Vision: encode 1 new timestep (3 frames, batch_size=1)
      2. VLM: incremental prefill with 768 new tokens (3 frames * 256 tokens/frame)
      3. Action: parallel decode with N*768 token context length
    
    Args:
        config: Pi0Config for the model
        systems: List of systems to test
        timestep_counts: List of timestep counts to test
        bits: Precision
        experiment_num: Experiment number for logging (optional)
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}LONG CONTEXT EXPERIMENT: {config.name}")
    logger.info("=" * 150)
    logger.info(f"Timesteps: {timestep_counts} (each timestep = {config.vision_frames} frames)")
    logger.info(f"Systems: {systems}")
    
    results = []
    
    # Tokens per timestep = vision_tokens * vision_frames
    tokens_per_timestep = config.vision_tokens * config.vision_frames
    language_tokens = config.language_tokens
    
    for system in systems:
        # Use single device for simplicity unless scaling is needed
        num_devices = 1 
        
        logger.info(f"\n  System: {system}")
        
        for timesteps in timestep_counts:
            # Vision: Always encode 1 new timestep (3 frames)
            df_vision = get_pi0_vision_perf(config, [system], [num_devices], bits)
            
            if df_vision.empty:
                logger.warning(f"    Timesteps={timesteps}: Failed to get vision latency")
                continue
            
            vision_row = df_vision[df_vision["batch_size"] == 1].iloc[0]
            vision_lat = vision_row["time_ms"]
            
            # VLM: For timesteps=1, full prefill. For timesteps>1, parallel decode (KV cache already has previous timesteps)
            # For timesteps>1: KV cache has (timesteps-1)*768 tokens, we add 768 new tokens in parallel
            df_vlm = get_pi0_vlm_perf(config, [system], [num_devices], bits)
            
            if df_vlm.empty:
                logger.warning(f"    Timesteps={timesteps}: Failed to get VLM latency")
                continue
            
            vlm_row = df_vlm[df_vlm["batch_size"] == 1].iloc[0]
            
            if timesteps == 1:
                # Full prefill with 768 tokens (3 frames * 256 tokens/frame)
                vlm_lat = vlm_row["time_ms"]
            else:
                # Parallel decode: existing KV cache + 768 new tokens
                existing_kv_tokens = (timesteps - 1) * tokens_per_timestep + language_tokens
                vlm_results = collect_parallel_decode_perf(
                    model=config.vlm_model,
                    system=system,
                    num_devices=num_devices,
                    input_tokens=existing_kv_tokens,  # Existing KV cache length
                    output_tokens_parallel=tokens_per_timestep,  # 768 new tokens to add
                    self_attention=True, 
                    bits=bits,
                    max_batch_size=1,
                )
                
                if not vlm_results:
                    logger.warning(f"    Timesteps={timesteps}: Failed to get VLM parallel decode latency")
                    continue
                
                # Get batch_size=1 result
                vlm_lat = None
                for r in vlm_results:
                    if r["batch_size"] == 1:
                        vlm_lat = r["time_ms"]
                        break
                
                if vlm_lat is None:
                    logger.warning(f"    Timesteps={timesteps}: No batch_size=1 result for VLM parallel decode")
                    continue
            
            # Calculate KV cache size in MB
            total_context_tokens = timesteps * tokens_per_timestep + language_tokens
            vlm_kv_cache_mb = calculate_kv_cache_size_mb(
                model_name=config.vlm_model,
                seq_length=total_context_tokens,
                bits=bits,
            )
            
            # Action: Use get_pi0_action_expert_perf with the full context length
            df_action = get_pi0_action_expert_perf(
                config=config,
                system_list=[system],
                num_device_list=[num_devices],
                bits=bits,
                denoising_steps=config.denoising_steps,
                vlm_sequence_length=total_context_tokens,  # Full context length
                action_chunk_size=config.action_chunk_size,
                max_batch_size=1,
            )
            
            if df_action.empty:
                logger.warning(f"    Timesteps={timesteps}: Failed to get action expert latency")
                continue
            
            # Get batch_size=1 result
            action_row = df_action[df_action["batch_size"] == 1]
            if action_row.empty:
                logger.warning(f"    Timesteps={timesteps}: No batch_size=1 result for action expert")
                continue
            
            action_lat_single = action_row.iloc[0]["time_ms"]
            
            e2e = vision_lat + vlm_lat + action_lat_single
            fps = 1000.0 / e2e
            
            # Get memory information from the dataframes (already calculated)
            # Vision and VLM weights are already in the result rows
            vision_weights_mb = vision_row.get("weights_mb", 0)
            vlm_weights_mb = vlm_row.get("weights_mb", 0)
            action_weights_mb = action_row.iloc[0].get("weights_mb", 0)
            total_weights_mb = vision_weights_mb + vlm_weights_mb + action_weights_mb
            
            # Total memory = weights + KV cache
            total_memory_mb = total_weights_mb + vlm_kv_cache_mb
            
            results.append({
                "system": system,
                "timesteps": timesteps,
                "total_tokens": total_context_tokens,
                "weights_mb": total_weights_mb,
                "vlm_kv_cache_mb": vlm_kv_cache_mb,
                "total_memory_mb": total_memory_mb,
                "vision_ms": vision_lat,
                "vlm_ms": vlm_lat,
                "action_ms": action_lat_single,
                "e2e_ms": e2e,
                "fps": fps
            })
            
            # Format memory display (use MB or GB as appropriate)
            if total_memory_mb >= 1024:
                total_mem_str = f"{total_memory_mb/1024:.2f} GB"
            else:
                total_mem_str = f"{total_memory_mb:.2f} MB"
            
            if vlm_kv_cache_mb >= 1024:
                kv_cache_str = f"{vlm_kv_cache_mb/1024:.2f} GB"
            else:
                kv_cache_str = f"{vlm_kv_cache_mb:.2f} MB"
            
            logger.info(f"    Timesteps={timesteps:<5} (Total Tokens={total_context_tokens:<7}, "
                       f"Total Mem={total_mem_str:<10}, KV Cache={kv_cache_str:<10}): "
                       f"Vis={vision_lat:.1f}ms, VLM={vlm_lat:.1f}ms, "
                       f"Act={action_lat_single:.1f}ms -> E2E={e2e:.1f}ms ({fps:.1f} Hz)")
                
    df = pd.DataFrame(results)
    
    # Save results
    if not df.empty:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        df.to_csv(output_path / "pi0_long_context.csv", index=False)
        logger.info(f"\nResults saved to {output_path / 'pi0_long_context.csv'}")
        
    return df


# ==============================================================================
# Experiment 4: Denoising Steps x Action Chunk Size
# Sweeps over diffusion denoising steps and action chunk sizes to understand
# their joint impact on E2E latency and operational intensity.
# ==============================================================================

def compare_denoising_steps_action_lengths(
    configs: list[Pi0Config] = [PI0_CONFIG],
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_devices: int = 1,
    bits: str = "bf16",
    step_range: list[int] = [1, 5, 10, 20, 50],
    action_chunk_size_range: list[int] = [5, 25, 50, 100, 250],
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Compare performance across different numbers of denoising steps and action chunk sizes.
    
    This helps understand the trade-off between action quality and latency for both dimensions.
    
    Args:
        configs: List of Pi0Config objects to compare (defaults to all configs)
        systems: List of system names to evaluate (defaults to ["H100"])
        num_devices: Number of devices
        bits: Precision
        step_range: List of denoising step counts to test
        action_chunk_size_range: List of action chunk sizes to test (defaults to [10, 25, 50, 100, 200])
        default_denoising_steps: Default denoising steps for fixed-step comparison
        default_action_chunk_size: Default action chunk size for fixed-chunk comparison
        experiment_num: Experiment number for logging (optional)
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}DENOISING STEPS AND ACTION CHUNK SIZE COMPARISON")
    logger.info("=" * 150)
    logger.info(f"Systems: {systems}, Batch Size: 1")
    logger.info(f"Models: {[c.name for c in configs]}")
    logger.info(f"Denoising Steps Range: {step_range}")
    logger.info(f"Action Chunk Size Range: {action_chunk_size_range}")
    
    results = []
    
    # Iterate over system x model combinations
    for system in systems:
        for config in configs:
            logger.info(f"\nEvaluating {config.name} on {system}...")
            
            # Get vision and VLM performance (constant for all steps/chunk sizes)
            # Precision is automatically selected based on system capabilities
            df_vision = get_pi0_vision_perf(config, [system], [num_devices], bits)
            df_vlm = get_pi0_vlm_perf(config, [system], [num_devices], bits)
            
            if df_vision.empty or df_vlm.empty:
                logger.warning(f"  Skipped {config.name} on {system} (memory constraints)")
                continue
            
            vision_row = df_vision[df_vision["batch_size"] == 1].iloc[0]
            vision_time = vision_row["time_ms"]
            vision_bound = vision_row.get("boundness", "N/A")
            vision_oi = vision_row.get("op_intensity", 0)
            
            vlm_row = df_vlm[df_vlm["batch_size"] == 1].iloc[0]
            vlm_time = vlm_row["time_ms"]
            vlm_bound = vlm_row.get("boundness", "N/A")
            vlm_oi = vlm_row.get("op_intensity", 0)
            
            # Evaluate all combinations of steps and chunk sizes
            for steps in step_range:
                for chunk_size in action_chunk_size_range:
                    df_action = get_pi0_action_expert_perf(
                        config, [system], [num_devices], bits, steps,
                        vlm_sequence_length=config.vlm_sequence_length,
                        action_chunk_size=chunk_size
                    )
                    
                    if not df_action.empty:
                        action_row = df_action[df_action["batch_size"] == 1].iloc[0]
                        action_time = action_row["time_ms"]
                        action_bound = action_row.get("boundness", "N/A")
                        action_oi = action_row.get("op_intensity", 0)
                        e2e_time = vision_time + vlm_time + action_time
                        
                        results.append({
                            "system": system,
                            "model": config.name,
                            "denoising_steps": steps,
                            "action_chunk_size": chunk_size,
                            "vision_ms": vision_time,
                            "vlm_ms": vlm_time,
                            "action_ms": action_time,
                            "e2e_ms": e2e_time,
                            "frequency_hz": 1000 / e2e_time,
                            "precision": bits,
                            "vision_boundness": vision_bound,
                            "vlm_boundness": vlm_bound,
                            "action_boundness": action_bound,
                            "vision_op_intensity": vision_oi,
                            "vlm_op_intensity": vlm_oi,
                            "action_op_intensity": action_oi,
                        })
    
    df = pd.DataFrame(results)
    
    # Save results to CSV
    if not df.empty:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / "pi0_denoising_steps_action_lengths.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")
    
    return df


def print_denoising_steps_action_lengths_summarys(
    df: pd.DataFrame,
    default_denoising_steps: int = PI0_DEFAULT_DENOISING_STEPS,
    default_action_chunk_size: int = PI0_ACTION_CHUNK_SIZE,
    logger=None
) -> None:
    """
    Print formatted summaries showing 4 different views for each system x model combination:
    (a) Diffusion step latency table with fixed default action chunk size
    (b) Action chunk size latency table with default diffusion step
    (c) 2D grid showing latency of various combinations
    (d) Relative speedup grid using default combination as baseline
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.info("No results to display.")
        return
    
    # Get unique values
    systems = sorted(df["system"].unique()) if "system" in df.columns else [None]
    models = sorted(df["model"].unique())
    step_range = sorted(df["denoising_steps"].unique())
    chunk_range = sorted(df["action_chunk_size"].unique())
    
    # Iterate over system x model combinations
    for system in systems:
        for model in models:
            # Filter data for this system x model combination
            if system is not None:
                combo_df = df[(df["system"] == system) & (df["model"] == model)].copy()
                combo_name = f"{model} on {system}"
            else:
                combo_df = df[df["model"] == model].copy()
                combo_name = model
            
            if combo_df.empty:
                continue
            
            logger.info("\n" + "-" * 100)
            logger.info(f"Summary: {combo_name}")
            logger.info("-" * 100)
    
            # (a) Diffusion step latency table with fixed default action chunk size
            logger.info("\n" + "-" * 150)
            logger.info(f"(a) Denoising Steps Comparison (Action Chunk Size = {default_action_chunk_size})")
            logger.info("-" * 150)
            logger.info(f"{'Steps':<8} {'Vision':>28} {'VLM':>28} {'Action':>28} {'E2E (ms)':>12} {'Freq (Hz)':>12}")
            logger.info(f"{'':8} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'':12} {'':12}")
            logger.info("-" * 150)
            
            df_fixed_chunk = combo_df[combo_df["action_chunk_size"] == default_action_chunk_size].copy()
            df_fixed_chunk_sorted = df_fixed_chunk.sort_values(["denoising_steps"])
            
            for _, row in df_fixed_chunk_sorted.iterrows():
                vision_oi = row.get('vision_op_intensity', 0)
                vlm_oi = row.get('vlm_op_intensity', 0)
                action_oi = row.get('action_op_intensity', 0)
                vision_str = f"{row['vision_ms']:6.2f}/{row.get('vision_boundness', 'N/A'):>4}/{vision_oi:6.1f}"
                vlm_str = f"{row['vlm_ms']:6.2f}/{row.get('vlm_boundness', 'N/A'):>4}/{vlm_oi:6.1f}"
                action_str = f"{row['action_ms']:6.2f}/{row.get('action_boundness', 'N/A'):>4}/{action_oi:6.1f}"
                
                logger.info(f"{int(row['denoising_steps']):<8} "
                      f"{vision_str:>28} "
                      f"{vlm_str:>28} "
                      f"{action_str:>28} "
                      f"{row['e2e_ms']:>12.2f} "
                      f"{row['frequency_hz']:>12.2f}")
            
            # (b) Action chunk size latency table with default diffusion step
            logger.info("\n" + "-" * 150)
            logger.info(f"(b) Action Chunk Size Comparison (Denoising Steps = {default_denoising_steps})")
            logger.info("-" * 150)
            logger.info(f"{'Chunk':<8} {'Vision':>28} {'VLM':>28} {'Action':>28} {'E2E (ms)':>12} {'Freq (Hz)':>12}")
            logger.info(f"{'':8} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'(ms/bound/OI)':>28} {'':12} {'':12}")
            logger.info("-" * 150)
            
            df_fixed_steps = combo_df[combo_df["denoising_steps"] == default_denoising_steps].copy()
            df_fixed_steps_sorted = df_fixed_steps.sort_values(["action_chunk_size"])
            
            for _, row in df_fixed_steps_sorted.iterrows():
                vision_oi = row.get('vision_op_intensity', 0)
                vlm_oi = row.get('vlm_op_intensity', 0)
                action_oi = row.get('action_op_intensity', 0)
                vision_str = f"{row['vision_ms']:6.2f}/{row.get('vision_boundness', 'N/A'):>4}/{vision_oi:6.1f}"
                vlm_str = f"{row['vlm_ms']:6.2f}/{row.get('vlm_boundness', 'N/A'):>4}/{vlm_oi:6.1f}"
                action_str = f"{row['action_ms']:6.2f}/{row.get('action_boundness', 'N/A'):>4}/{action_oi:6.1f}"
                
                logger.info(f"{int(row['action_chunk_size']):<8} "
                      f"{vision_str:>28} "
                      f"{vlm_str:>28} "
                      f"{action_str:>28} "
                      f"{row['e2e_ms']:>12.2f} "
                      f"{row['frequency_hz']:>12.2f}")
            
            # (c) 2D grid showing latency of various combinations
            logger.info("\n" + "-" * 100)
            logger.info(f"(c) 2D Latency Grid: {combo_name} (E2E Latency in ms)")
            logger.info("-" * 100)
            logger.info(f"  Y-axis: Denoising Steps ↓  |  X-axis: Action Chunk Size →")
            logger.info("-" * 100)
            
            model_df = combo_df.copy()
        
            # Create pivot table: steps as rows, chunk sizes as columns
            pivot = model_df.pivot_table(
                values="e2e_ms",
                index="denoising_steps",
                columns="action_chunk_size",
                aggfunc="first"
            )
            
            # Print header with clear column labels (chunk sizes)
            header = f"{'Steps ↓':<12}"
            for chunk in chunk_range:
                header += f"{chunk:>12}"
            logger.info(header)
            logger.info("-" * (12 + 12 * len(chunk_range)))
            
            # Print rows with clear step labels (denoising steps)
            for steps in step_range:
                if steps in pivot.index:
                    row_str = f"{steps:<12}"
                    for chunk in chunk_range:
                        if chunk in pivot.columns:
                            val = pivot.loc[steps, chunk]
                            row_str += f"{val:>12.2f}"
                        else:
                            row_str += f"{'N/A':>12}"
                    logger.info(row_str)
                else:
                    row_str = f"{steps:<12}"
                    for chunk in chunk_range:
                        row_str += f"{'N/A':>12}"
                    logger.info(row_str)
            
            # (d) Relative speedup grid using default combination as baseline
            logger.info("\n" + "-" * 100)
            logger.info(f"(d) Relative Speedup Grid: {combo_name} (Baseline: {default_denoising_steps} steps, {default_action_chunk_size} chunk)")
            logger.info("-" * 100)
            logger.info(f"  Y-axis: Denoising Steps ↓  |  X-axis: Action Chunk Size →")
            logger.info("-" * 100)
            
            # Get baseline latency
            baseline_df = combo_df[
                (combo_df["denoising_steps"] == default_denoising_steps) &
                (combo_df["action_chunk_size"] == default_action_chunk_size)
            ]
            
            if baseline_df.empty:
                logger.warning(f"  Baseline combination ({default_denoising_steps} steps, {default_action_chunk_size} chunk) not found for {combo_name}")
                continue
            
            baseline_latency = baseline_df["e2e_ms"].values[0]
            
            # Create pivot table for speedup
            pivot = model_df.pivot_table(
                values="e2e_ms",
                index="denoising_steps",
                columns="action_chunk_size",
                aggfunc="first"
            )
            
            # Compute speedup (baseline / current)
            speedup_pivot = baseline_latency / pivot
            
            # Print header with clear column labels (chunk sizes)
            header = f"{'Steps ↓':<12}"
            for chunk in chunk_range:
                header += f"{chunk:>12}"
            logger.info(header)
            logger.info("-" * (12 + 12 * len(chunk_range)))
            
            # Print rows with clear step labels
            for steps in step_range:
                if steps in speedup_pivot.index:
                    row_str = f"{steps:<12}"
                    for chunk in chunk_range:
                        if chunk in speedup_pivot.columns:
                            val = speedup_pivot.loc[steps, chunk]
                            row_str += f"{val:>12.2f}x"
                        else:
                            row_str += f"{'N/A':>12}"
                    logger.info(row_str)
                else:
                    row_str = f"{steps:<12}"
                    for chunk in chunk_range:
                        row_str += f"{'N/A':>12}"
                    logger.info(row_str)


# ==============================================================================
# Experiment 5: Autoregressive vs Diffusion
# Compares four action generation strategies:
#   A) Autoregressive (sequential decode, like OpenVLA)
#   B) Small Diffusion (Pi0 action expert, ~300M)
#   C) Large Diffusion (VLM-sized DiT, ~2B)
#   D) Autoregressive with Parallel Decoding
# ==============================================================================

def compare_autoregressive_vs_diffusion(
    config: Pi0Config = PI0_CONFIG,
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_devices: int = 1,
    bits: str = "bf16",
    denoising_steps_range: list[int] = [10],
    action_chunk_sizes: list[int] = [1, 5, 10, 50, 100],
    dof_values: list[int] = [7, 14, 21, 28, 35, 42],
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Compare Autoregressive vs Diffusion action generators for Pi0.
    
    Four setups (all using pi0-vlm-action-predictor for Setups A, C, D):
    - Setup A: Autoregressive VLA (VLM backbone as action predictor, sequential decoding)
      * Also tests different DoF values to see impact on latency
    - Setup B: Small Diffusion model (Pi-0 action expert, ~300M params)
    - Setup C: Large Diffusion model (VLM-sized action predictor as DiT, ~2B params)
    - Setup D: Autoregressive with Parallel Decoding (VLM backbone as action predictor, parallel generation)
      * Also tests different DoF values with chunk_size=1
    
    For each setup, we vary the action chunk size (1, 5, 10, 50, 100) and measure latency.
    
    Note: pi0-vlm-action-predictor is the same architecture as pi0-vlm but with vocab_size=0,
    eliminating the vocabulary head overhead for fair comparison.
    
    Args:
        config: Pi0Config for the model
        systems: List of system names to evaluate
        num_devices: Number of devices
        bits: Precision
        denoising_steps_range: List of denoising step counts for diffusion models
        action_chunk_sizes: List of action chunk sizes to test
        dof_values: List of DoF values to test (for autoregressive VLA heatmap)
        experiment_num: Experiment number for logging
        logger: Logger instance
    
    Returns:
        DataFrame with latency results for each setup x denoising_steps x action_chunk_size x dof
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}AUTOREGRESSIVE VS DIFFUSION ACTION GENERATORS COMPARISON")
    logger.info("=" * 150)
    logger.info(f"Systems: {systems}, Batch Size: 1")
    logger.info(f"Model: {config.name}")
    logger.info(f"Denoising Steps Range: {denoising_steps_range}")
    logger.info(f"Action Chunk Sizes: {action_chunk_sizes}")
    logger.info(f"DoF Values (for Autoregressive): {dof_values}")
    
    results = []
    
    
    # Iterate over systems
    for system in systems:
        logger.info(f"\nEvaluating on {system}...")
        
        # Get vision and VLM performance (constant for all setups)
        df_vision = get_pi0_vision_perf(config, [system], [num_devices], bits)
        df_vlm = get_pi0_vlm_perf(config, [system], [num_devices], bits)
        
        if df_vision.empty or df_vlm.empty:
            logger.warning(f"  Skipped {system} (memory constraints)")
            continue
        
        vision_row = df_vision[df_vision["batch_size"] == 1].iloc[0]
        vision_time = vision_row["time_ms"]
        
        vlm_row = df_vlm[df_vlm["batch_size"] == 1].iloc[0]
        vlm_time = vlm_row["time_ms"]
        
        # ==== Setup A: Autoregressive VLA ====
        # Use VLM backbone as action predictor to generate actions autoregressively
        # Test with different DoF values to see impact on latency
        logger.info(f"  Setup A: Autoregressive VLA (using VLM backbone as action predictor)")
        
        action_predictor_model = "pi0-vlm-action-predictor"
        pi0_vlm_action_predictor_config = copy.deepcopy(MODEL_DICT.get_model(config.vlm_model))
        pi0_vlm_action_predictor_config.model = action_predictor_model
        pi0_vlm_action_predictor_config.vocab_size = 0  # No vocabulary head for action prediction
        MODEL_DICT.add_model(pi0_vlm_action_predictor_config)

                # For default DoF, test all chunk sizes (for main comparison table)
        for chunk_size in action_chunk_sizes:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.vlm_sequence_length,
                output_tokens=config.action_dof * chunk_size,  # Total action tokens to generate
                bits=bits,
                max_batch_size=1,
            )
            
            if decode_results:
                decode_row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_time = decode_row["time_ms"] * config.action_dof * chunk_size
                action_oi = decode_row.get("op_intensity", 0)
                e2e_time = vision_time + vlm_time + action_time
                
                results.append({
                    "system": system,
                    "setup": "A: Autoregressive",
                    "denoising_steps": "N/A", 
                    "action_chunk_size": chunk_size,
                    "dof": config.action_dof,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "action_oi": action_oi,
                    "e2e_ms": e2e_time,
                    "frequency_hz": 1000 / e2e_time,
                })
        
        # For DoF variations with chunk_size=1 (for DoF comparison plot)
        logger.info(f"    Testing DoF variations with chunk_size=1...")
        for dof in dof_values:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.vlm_sequence_length,
                output_tokens=dof * 1,  # chunk_size=1, vary DoF
                bits=bits,
                max_batch_size=1,
            )
            
            if decode_results:
                decode_row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_time = decode_row["time_ms"] * dof * 1
                action_oi = decode_row.get("op_intensity", 0)
                e2e_time = vision_time + vlm_time + action_time
                
                results.append({
                    "system": system,
                    "setup": "A: Autoregressive (DoF Comparison)",
                    "denoising_steps": "N/A", 
                    "action_chunk_size": 1,
                    "dof": dof,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "action_oi": action_oi,
                    "e2e_ms": e2e_time,
                    "frequency_hz": 1000 / e2e_time,
                })
        
        # ==== Setup B: Small Diffusion (Pi-0 action expert) ====
        logger.info(f"  Setup B: Small Diffusion (Pi-0 action expert)")
        
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                df_action = get_pi0_action_expert_perf(
                    config, [system], [num_devices], bits, steps,
                    vlm_sequence_length=config.vlm_sequence_length,
                    action_chunk_size=chunk_size
                )
                
                if not df_action.empty:
                    action_row = df_action[df_action["batch_size"] == 1].iloc[0]
                    action_time = action_row["time_ms"]
                    action_oi = action_row.get("op_intensity", 0)
                    e2e_time = vision_time + vlm_time + action_time
                    
                    results.append({
                        "system": system,
                        "setup": "B: Small Diffusion",
                        "denoising_steps": steps,
                        "action_chunk_size": chunk_size,
                        "dof": config.action_dof,
                        "vision_ms": vision_time,
                        "vlm_ms": vlm_time,
                        "action_ms": action_time,
                        "action_oi": action_oi,
                        "e2e_ms": e2e_time,
                        "frequency_hz": 1000 / e2e_time,
                    })
        
        # For DoF variations with chunk_size=1 (for DoF comparison plot)
        logger.info(f"    Testing DoF variations with chunk_size=1...")
        for dof in dof_values:
            for steps in denoising_steps_range:
                df_action = get_pi0_action_expert_perf(
                    config, [system], [num_devices], bits, steps,
                    vlm_sequence_length=config.vlm_sequence_length,
                    action_chunk_size=1
                )
                
                if not df_action.empty:
                    action_row = df_action[df_action["batch_size"] == 1].iloc[0]
                    action_time = action_row["time_ms"]
                    action_oi = action_row.get("op_intensity", 0)
                    e2e_time = vision_time + vlm_time + action_time
                    
                    results.append({
                        "system": system,
                        "setup": "B: Small Diffusion (DoF Comparison)",
                        "denoising_steps": steps,
                        "action_chunk_size": 1,
                        "dof": dof,
                        "vision_ms": vision_time,
                        "vlm_ms": vlm_time,
                        "action_ms": action_time,
                        "action_oi": action_oi,
                        "e2e_ms": e2e_time,
                        "frequency_hz": 1000 / e2e_time,
                    })
        
        # ==== Setup C: Large Diffusion (same size as VLM backbone) ====
        logger.info(f"  Setup C: Large Diffusion (VLM-sized DiT as action predictor)")
        
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                # Use VLM action predictor as a proxy for a large DiT with similar parameters
                # The architecture is similar enough for performance estimation
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model,
                    system=system,
                    num_devices=num_devices,
                    input_tokens=config.vlm_sequence_length,
                    output_tokens_parallel=config.action_dof * chunk_size,  # Generate full action chunk in parallel
                    self_attention=True,
                    bits=bits,
                    max_batch_size=1,
                )
                
                if action_results:
                    action_row = [r for r in action_results if r["batch_size"] == 1][0]
                    single_step_time = action_row["time_ms"]
                    action_oi = action_row.get("op_intensity", 0)
                    action_time = single_step_time * steps  # Multiply by denoising steps
                    e2e_time = vision_time + vlm_time + action_time
                    
                    results.append({
                        "system": system,
                        "setup": "C: Large Diffusion",
                        "denoising_steps": steps,
                        "action_chunk_size": chunk_size,
                        "dof": config.action_dof,
                        "vision_ms": vision_time,
                        "vlm_ms": vlm_time,
                        "action_ms": action_time,
                        "action_oi": action_oi,
                        "e2e_ms": e2e_time,
                        "frequency_hz": 1000 / e2e_time,
                    })
        
        # For DoF variations with chunk_size=1 (for DoF comparison plot)
        logger.info(f"    Testing DoF variations with chunk_size=1...")
        for dof in dof_values:
            for steps in denoising_steps_range:
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model,
                    system=system,
                    num_devices=num_devices,
                    input_tokens=config.vlm_sequence_length,
                    output_tokens_parallel=dof * 1,  # chunk_size=1, vary DoF
                    self_attention=True,
                    bits=bits,
                    max_batch_size=1,
                )
                
                if action_results:
                    action_row = [r for r in action_results if r["batch_size"] == 1][0]
                    single_step_time = action_row["time_ms"]
                    action_oi = action_row.get("op_intensity", 0)
                    action_time = single_step_time * steps  # Multiply by denoising steps
                    e2e_time = vision_time + vlm_time + action_time
                    
                    results.append({
                        "system": system,
                        "setup": "C: Large Diffusion (DoF Comparison)",
                        "denoising_steps": steps,
                        "action_chunk_size": 1,
                        "dof": dof,
                        "vision_ms": vision_time,
                        "vlm_ms": vlm_time,
                        "action_ms": action_time,
                        "action_oi": action_oi,
                        "e2e_ms": e2e_time,
                        "frequency_hz": 1000 / e2e_time,
                    })
        
        # ==== Setup D: Autoregressive with Parallel Decoding ====
        # Generate all action chunks and DoF simultaneously in one pass
        logger.info(f"  Setup D: Autoregressive with Parallel Decoding (using VLM backbone as action predictor)")
        
        # For default DoF, test all chunk sizes
        for chunk_size in action_chunk_sizes:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.vlm_sequence_length,
                output_tokens_parallel=config.action_dof * chunk_size,  # Generate all tokens in parallel
                self_attention=True,
                bits=bits,
                max_batch_size=1,
            )
            
            if action_results:
                action_row = [r for r in action_results if r["batch_size"] == 1][0]
                action_time = action_row["time_ms"]
                action_oi = action_row.get("op_intensity", 0)
                e2e_time = vision_time + vlm_time + action_time
                
                results.append({
                    "system": system,
                    "setup": "D: Autoregressive Parallel",
                    "denoising_steps": "N/A",
                    "action_chunk_size": chunk_size,
                    "dof": config.action_dof,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "action_oi": action_oi,
                    "e2e_ms": e2e_time,
                    "frequency_hz": 1000 / e2e_time,
                })
        
        # For DoF variations with chunk_size=1
        logger.info(f"    Testing DoF variations with chunk_size=1...")
        for dof in dof_values:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.vlm_sequence_length,
                output_tokens_parallel=dof * 1,  # chunk_size=1, vary DoF
                self_attention=True,
                bits=bits,
                max_batch_size=1,
            )
            
            if action_results:
                action_row = [r for r in action_results if r["batch_size"] == 1][0]
                action_time = action_row["time_ms"]
                action_oi = action_row.get("op_intensity", 0)
                e2e_time = vision_time + vlm_time + action_time
                
                results.append({
                    "system": system,
                    "setup": "D: Autoregressive Parallel (DoF Comparison)",
                    "denoising_steps": "N/A",
                    "action_chunk_size": 1,
                    "dof": dof,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "action_oi": action_oi,
                    "e2e_ms": e2e_time,
                    "frequency_hz": 1000 / e2e_time,
                })
    
    results = pd.DataFrame(results)

    # Save results to CSV
    if not results.empty:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / "pi0_autoregressive_vs_diffusion.csv"
        results.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")
    
    return results


def print_autoregressive_vs_diffusion_summary(
    df: pd.DataFrame,
    logger=None,
    default_dof: int = 14,
    default_chunk_size: int = 1,
    default_denoising_steps: int = 10
) -> None:
    """
    Print formatted summary for autoregressive vs diffusion comparison.
    
    Six tables:
    - Table 1: VLA Total Latency vs Action Chunk Size (DoF=14)
    - Table 2: VLA Total Latency vs DoF (Chunk Size=1)
    - Table 3: Action Prediction Latency vs Action Chunk Size (DoF=14)
    - Table 4: Action Prediction Latency vs DoF (Chunk Size=1)
    - Table 5: Action Prediction OI vs Action Chunk Size (DoF=14)
    - Table 6: Action Prediction OI vs DoF (Chunk Size=1)
    
    Args:
        df: DataFrame with results
        logger: Logger instance
        default_dof: DoF value for Tables 1, 3, 5 (default: 14)
        default_chunk_size: Chunk size for Tables 2, 4, 6 (default: 1)
        default_denoising_steps: Denoising steps for diffusion models (default: 10)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.info("No results to display.")
        return
    
    # Get unique values
    systems = sorted(df["system"].unique())
    
    # Iterate over systems
    for system in systems:
        system_df = df[df["system"] == system].copy()
        
        if system_df.empty:
            continue
        
        logger.info("\n" + "=" * 120)
        logger.info(f"System: {system}")
        logger.info("=" * 120)
        
        # ===== TABLE 1: VLA Total Latency vs Action Chunk Size (DoF=14) =====
        logger.info(f"\nTABLE 1: VLA Total Latency vs Action Chunk Size (DoF={default_dof})")
        logger.info("-" * 120)
        
        # Filter for default DoF and default denoising steps
        table1_df = system_df[system_df["dof"] == default_dof].copy()
        
        # Get unique chunk sizes
        chunk_sizes = sorted(table1_df["action_chunk_size"].unique())
        
        # Build header
        header_parts = [f"{'Solution':<30}"]
        for chunk_size in chunk_sizes:
            header_parts.append(f"Chunk={chunk_size:>3}")
        logger.info(" | ".join(header_parts))
        logger.info("-" * 120)
        
        # Define solutions in the order we want (matching plot order)
        solutions = [
            ("Autoregressive", "A: Autoregressive", "N/A"),
            ("Diffusion", "B: Small Diffusion", default_denoising_steps),
            ("Autoregressive-Parallel", "D: Autoregressive Parallel", "N/A"),
            ("Diffusion-Large", "C: Large Diffusion", default_denoising_steps),
        ]
        
        for solution_name, setup_name, steps in solutions:
            # Filter for this solution
            if steps == "N/A":
                solution_df = table1_df[table1_df["setup"] == setup_name]
            else:
                solution_df = table1_df[
                    (table1_df["setup"] == setup_name) & 
                    (table1_df["denoising_steps"] == steps)
                ]
            
            if solution_df.empty:
                continue
            
            # Build row data
            row_parts = [f"{solution_name:<30}"]
            for chunk_size in chunk_sizes:
                chunk_df = solution_df[solution_df["action_chunk_size"] == chunk_size]
                if not chunk_df.empty:
                    latency = chunk_df["e2e_ms"].values[0]
                    row_parts.append(f"{latency:>10.2f}")
                else:
                    row_parts.append(f"{'N/A':>10}")
            
            logger.info(" | ".join(row_parts))
        
        # ===== TABLE 2: VLA Total Latency vs DoF (Chunk Size=1) =====
        logger.info(f"\nTABLE 2: VLA Total Latency vs DoF (Chunk Size={default_chunk_size})")
        logger.info("-" * 120)
        
        # Filter for default chunk size and use DoF Comparison setups
        table2_df = system_df[system_df["action_chunk_size"] == default_chunk_size].copy()
        
        # Get unique DoF values
        dof_values = sorted(table2_df["dof"].unique())
        
        # Build header
        header_parts = [f"{'Solution':<30}"]
        for dof in dof_values:
            header_parts.append(f"DoF={dof:>3}")
        logger.info(" | ".join(header_parts))
        logger.info("-" * 120)
        
        # Define solutions in the order we want (matching plot order)
        solutions_dof = [
            ("Autoregressive", "A: Autoregressive (DoF Comparison)", "N/A"),
            ("Diffusion", "B: Small Diffusion (DoF Comparison)", default_denoising_steps),
            ("Autoregressive-Parallel", "D: Autoregressive Parallel (DoF Comparison)", "N/A"),
            ("Diffusion-Large", "C: Large Diffusion (DoF Comparison)", default_denoising_steps),
        ]
        
        for solution_name, setup_name, steps in solutions_dof:
            # Filter for this solution
            if steps == "N/A":
                solution_df = table2_df[table2_df["setup"] == setup_name]
            else:
                solution_df = table2_df[
                    (table2_df["setup"] == setup_name) & 
                    (table2_df["denoising_steps"] == steps)
                ]
            
            if solution_df.empty:
                continue
            
            # Build row data
            row_parts = [f"{solution_name:<30}"]
            for dof in dof_values:
                dof_df = solution_df[solution_df["dof"] == dof]
                if not dof_df.empty:
                    latency = dof_df["e2e_ms"].values[0]
                    row_parts.append(f"{latency:>9.2f}")
                else:
                    row_parts.append(f"{'N/A':>9}")
            
            logger.info(" | ".join(row_parts))
        
        # ===== TABLE 3: Action Prediction Latency vs Action Chunk Size (DoF=14) =====
        logger.info(f"\nTABLE 3: Action Prediction Latency vs Action Chunk Size (DoF={default_dof})")
        logger.info("(Action prediction only, excluding Vision + VLM)")
        logger.info("-" * 120)
        
        # Build header
        header_parts = [f"{'Solution':<30}"]
        for chunk_size in chunk_sizes:
            header_parts.append(f"Chunk={chunk_size:>3}")
        logger.info(" | ".join(header_parts))
        logger.info("-" * 120)
        
        for solution_name, setup_name, steps in solutions:
            # Filter for this solution
            if steps == "N/A":
                solution_df = table1_df[table1_df["setup"] == setup_name]
            else:
                solution_df = table1_df[
                    (table1_df["setup"] == setup_name) & 
                    (table1_df["denoising_steps"] == steps)
                ]
            
            if solution_df.empty:
                continue
            
            # Build row data
            row_parts = [f"{solution_name:<30}"]
            for chunk_size in chunk_sizes:
                chunk_df = solution_df[solution_df["action_chunk_size"] == chunk_size]
                if not chunk_df.empty:
                    action_latency = chunk_df["action_ms"].values[0]
                    row_parts.append(f"{action_latency:>10.2f}")
                else:
                    row_parts.append(f"{'N/A':>10}")
            
            logger.info(" | ".join(row_parts))
        
        # ===== TABLE 4: Action Prediction Latency vs DoF (Chunk Size=1) =====
        logger.info(f"\nTABLE 4: Action Prediction Latency vs DoF (Chunk Size={default_chunk_size})")
        logger.info("(Action prediction only, excluding Vision + VLM)")
        logger.info("-" * 120)
        
        # Build header
        header_parts = [f"{'Solution':<30}"]
        for dof in dof_values:
            header_parts.append(f"DoF={dof:>3}")
        logger.info(" | ".join(header_parts))
        logger.info("-" * 120)
        
        for solution_name, setup_name, steps in solutions_dof:
            # Filter for this solution
            if steps == "N/A":
                solution_df = table2_df[table2_df["setup"] == setup_name]
            else:
                solution_df = table2_df[
                    (table2_df["setup"] == setup_name) & 
                    (table2_df["denoising_steps"] == steps)
                ]
            
            if solution_df.empty:
                continue
            
            # Build row data
            row_parts = [f"{solution_name:<30}"]
            for dof in dof_values:
                dof_df = solution_df[solution_df["dof"] == dof]
                if not dof_df.empty:
                    action_latency = dof_df["action_ms"].values[0]
                    row_parts.append(f"{action_latency:>9.2f}")
                else:
                    row_parts.append(f"{'N/A':>9}")
            
            logger.info(" | ".join(row_parts))
        
        # ===== TABLE 5: Action Prediction OI vs Action Chunk Size (DoF=14) =====
        logger.info(f"\nTABLE 5: Action Prediction Operational Intensity (OI) vs Action Chunk Size (DoF={default_dof})")
        logger.info("(FLOPs/Byte - Higher is better for compute-bound workloads)")
        logger.info("-" * 120)
        
        # Build header
        header_parts = [f"{'Solution':<30}"]
        for chunk_size in chunk_sizes:
            header_parts.append(f"Chunk={chunk_size:>3}")
        logger.info(" | ".join(header_parts))
        logger.info("-" * 120)
        
        for solution_name, setup_name, steps in solutions:
            # Filter for this solution
            if steps == "N/A":
                solution_df = table1_df[table1_df["setup"] == setup_name]
            else:
                solution_df = table1_df[
                    (table1_df["setup"] == setup_name) & 
                    (table1_df["denoising_steps"] == steps)
                ]
            
            if solution_df.empty:
                continue
            
            # Build row data
            row_parts = [f"{solution_name:<30}"]
            for chunk_size in chunk_sizes:
                chunk_df = solution_df[solution_df["action_chunk_size"] == chunk_size]
                if not chunk_df.empty:
                    action_oi = chunk_df["action_oi"].values[0]
                    row_parts.append(f"{action_oi:>10.2f}")
                else:
                    row_parts.append(f"{'N/A':>10}")
            
            logger.info(" | ".join(row_parts))
        
        # ===== TABLE 6: Action Prediction OI vs DoF (Chunk Size=1) =====
        logger.info(f"\nTABLE 6: Action Prediction Operational Intensity (OI) vs DoF (Chunk Size={default_chunk_size})")
        logger.info("(FLOPs/Byte - Higher is better for compute-bound workloads)")
        logger.info("-" * 120)
        
        # Build header
        header_parts = [f"{'Solution':<30}"]
        for dof in dof_values:
            header_parts.append(f"DoF={dof:>3}")
        logger.info(" | ".join(header_parts))
        logger.info("-" * 120)
        
        for solution_name, setup_name, steps in solutions_dof:
            # Filter for this solution
            if steps == "N/A":
                solution_df = table2_df[table2_df["setup"] == setup_name]
            else:
                solution_df = table2_df[
                    (table2_df["setup"] == setup_name) & 
                    (table2_df["denoising_steps"] == steps)
                ]
            
            if solution_df.empty:
                continue
            
            # Build row data
            row_parts = [f"{solution_name:<30}"]
            for dof in dof_values:
                dof_df = solution_df[solution_df["dof"] == dof]
                if not dof_df.empty:
                    action_oi = dof_df["action_oi"].values[0]
                    row_parts.append(f"{action_oi:>9.2f}")
                else:
                    row_parts.append(f"{'N/A':>9}")
            
            logger.info(" | ".join(row_parts))


# ==============================================================================
# Experiment 6: Device vs Server (including network latency)
# Compares on-device (Jetson), edge-server (RTX 4090), and cloud (B100)
# inference, accounting for network transfer of images and actions.
# ==============================================================================

def compare_datacenter_vs_edge(
    config: Pi0Config = PI0_CONFIG,
    denoising_steps: int = 10,
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """
    Compare Pi0 performance between data center GPUs and Jetson edge devices.
    
    All comparisons use:
    - Single device (num_devices=1)
    - Batch size = 1 (typical for robotics inference)
    
    Automatically selects the best available precision for each device
    (e.g., fp16 for older Jetsons that don't support bf16).
    
    Args:
        config: Pi0 model config to evaluate
        denoising_steps: Number of flow matching denoising steps
        bits: Preferred precision for computation (fallback to fp16 if unavailable)
        
    Returns:
        DataFrame with performance comparison across all systems
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Data center GPUs
    datacenter_systems = [
        "A100_80GB",
        "H100",
        "B100",
    ]
    
    # Jetson edge devices (sorted roughly by capability)
    jetson_systems = [
        "Jetson_AGX_Thor",       # Most powerful Jetson (supports bf16/fp8)
        "Jetson_AGX_Orin_64GB",
        "Jetson_AGX_Orin_32GB",
        "Jetson_Orin_NX_16GB",
        "Jetson_Orin_NX_8GB",
        "Jetson_Orin_Nano_8GB",
        "Jetson_Orin_Nano_4GB",
        "Jetson_AGX_Xavier",
        "Jetson_Xavier_NX",
    ]
    
    all_systems = datacenter_systems + jetson_systems
    num_devices = 1  # Single device comparison
    batch_size = 1   # Always use batch_size=1 for robotics
    
    results = []
    
    logger.info(f"\n\nSettings: {denoising_steps} denoising steps, batch_size=1")
    
    for system in all_systems:
        try:
            # Get component latencies (only batch_size=1)
            # Precision is automatically selected based on system capabilities
            df_vision = get_pi0_vision_perf(
                config, [system], [num_devices], bits, max_batch_size=1
            )
            df_vlm = get_pi0_vlm_perf(
                config, [system], [num_devices], bits, max_batch_size=1
            )
            df_action = get_pi0_action_expert_perf(
                config, [system], [num_devices], bits, denoising_steps,
                vlm_sequence_length=config.vlm_sequence_length,
                max_batch_size=1
            )
            
            if df_vision.empty or df_vlm.empty or df_action.empty:
                logger.warning(f"  {system}: Skipped (memory constraints)")
                continue
            
            # Results are already batch_size=1 only
            vision_time = df_vision["time_ms"].values[0]
            vlm_time = df_vlm["time_ms"].values[0]
            action_time = df_action["time_ms"].values[0]
            e2e_time = vision_time + vlm_time + action_time
            
            # Determine device category
            category = "Data Center" if system in datacenter_systems else "Edge (Jetson)"
            
            results.append({
                "model": config.name,
                "system": system,
                "category": category,
                "precision": bits,
                "vision_ms": vision_time,
                "vlm_ms": vlm_time,
                "action_ms": action_time,
                "e2e_ms": e2e_time,
                "frequency_hz": 1000 / e2e_time,
                "denoising_steps": denoising_steps,
            })
            
        except Exception as e:
            logger.warning(f"  {system}: Error - {str(e)[:50]}")
            continue
    
    return pd.DataFrame(results)


def print_datacenter_vs_edge_summary(df: pd.DataFrame, logger=None) -> None:
    """Print a formatted summary comparing datacenter vs edge performance."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.info("No results to display.")
        return
    
    # Compute speedup ratios first (summary)
    dc_df = df[df["category"] == "Data Center"]
    edge_df = df[df["category"] == "Edge (Jetson)"]
    
    if not dc_df.empty and not edge_df.empty:
        fastest_dc = dc_df["e2e_ms"].min()
        fastest_dc_sys = dc_df.loc[dc_df["e2e_ms"].idxmin(), "system"]
        fastest_edge = edge_df["e2e_ms"].min()
        fastest_edge_sys = edge_df.loc[edge_df["e2e_ms"].idxmin(), "system"]
        
        logger.info("\n" + "-" * 100)
        logger.info("Summary:")
        logger.info(f"  Fastest Data Center: {fastest_dc_sys} @ {fastest_dc:.2f} ms ({1000/fastest_dc:.1f} Hz)")
        logger.info(f"  Fastest Edge:        {fastest_edge_sys} @ {fastest_edge:.2f} ms ({1000/fastest_edge:.1f} Hz)")
        logger.info(f"  DC/Edge Speedup:     {fastest_edge/fastest_dc:.1f}x")
        
        # Check if any edge device meets real-time robotics requirement (e.g., 10 Hz)
        realtime_threshold_hz = 10  # Typical robot control frequency
        edge_realtime = edge_df[edge_df["frequency_hz"] >= realtime_threshold_hz]
        if not edge_realtime.empty:
            logger.info(f"\n  Edge devices meeting {realtime_threshold_hz} Hz real-time requirement:")
            for _, row in edge_realtime.iterrows():
                logger.info(f"    - {row['system']}: {row['frequency_hz']:.1f} Hz")
        else:
            logger.info(f"\n  No edge device meets {realtime_threshold_hz} Hz real-time requirement")
    
    # Then print the table
    logger.info("\nData Center vs Edge Performance Comparison")
    
    # Separate by category
    for category in ["Data Center", "Edge (Jetson)"]:
        cat_df = df[df["category"] == category].sort_values("e2e_ms")
        
        if cat_df.empty:
            continue
            
        logger.info(f"\n{category} Systems:")
        logger.info("-" * 100)
        logger.info(f"{'System':<25} {'Prec':<6} {'Vision':>10} {'VLM':>10} {'Action':>10} {'E2E':>10} {'Freq':>10}")
        logger.info(f"{'':25} {'':6} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(Hz)':>10}")
        logger.info("-" * 100)
        
        for _, row in cat_df.iterrows():
            prec = row.get('precision', 'bf16')
            logger.info(f"{row['system']:<25} "
                  f"{prec:<6} "
                  f"{row['vision_ms']:>10.2f} "
                  f"{row['vlm_ms']:>10.2f} "
                  f"{row['action_ms']:>10.2f} "
                  f"{row['e2e_ms']:>10.2f} "
                  f"{row['frequency_hz']:>10.1f}")


def run_datacenter_vs_edge_comparison(
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Run full datacenter vs edge comparison for all Pi0 family models.
    
    Args:
        output_dir: Directory to save results
        experiment_num: Experiment number for logging (optional)
        logger: Logger instance (optional)
    
    Returns:
        Dictionary mapping model names to comparison DataFrames
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    all_dfs = []
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}DATA CENTER vs EDGE (JETSON) COMPARISON")
    logger.info("=" * 150)
    
    for config in ALL_PI0_CONFIGS:
        df = compare_datacenter_vs_edge(
            config=config,
            denoising_steps=10,
            bits="bf16",
            logger=logger,
        )
        
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        
        # Print model name, then summary and table
        logger.info(f"\n{config.name}:")
        print_datacenter_vs_edge_summary(df, logger=logger)
    
    # Save combined results
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        output_file = output_path / "pi0_datacenter_vs_edge.csv"
        df_combined.to_csv(output_file, index=False)
        logger.info(f"\n\nResults saved to {output_file}")
    
    return all_results


def compare_datacenter_vs_edge_with_network(
    config: Pi0Config = PI0_CONFIG,
    denoising_steps: int = 10,
    bits: str = "bf16",
    image_resolution: int = 384,  # Typical Pi0 image resolution (SigLIP)
    image_compression_ratio: float = 0.1,  # JPEG compression
    action_dof: int = 8,  # 7-DoF arm + gripper
    action_chunk_size: int = 1,
    network_configs: list[NetworkConfig] = None,
    logger=None,
) -> pd.DataFrame:
    """
    Compare Pi0 performance between Jetson Thor (local) and datacenter GPUs with network latency.
    
    Edge device (Jetson Thor):
    - Runs inference locally, no network latency
    
    Datacenter GPUs (A100/H100/B100):
    - Remote inference with network latency:
      * Image upload: Robot → Server (upload bandwidth)
      * Action download: Server → Robot (download bandwidth)
    
    Args:
        config: Pi0 model config to evaluate
        denoising_steps: Number of flow matching denoising steps
        bits: Preferred precision for computation
        image_resolution: Image resolution (width/height, assuming square)
        image_compression_ratio: Image compression ratio (0.1 = JPEG 10:1)
        action_dof: Number of degrees of freedom for actions
        action_chunk_size: Number of future actions (action chunking)
        network_configs: List of network configurations to test (default: all)
        logger: Logger instance (optional)
        
    Returns:
        DataFrame with performance comparison including network latency
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if network_configs is None:
        # Use all network configs by default
        network_configs = ALL_NETWORK_CONFIGS
    
    # Edge device (local, no network)
    edge_system = "Jetson_AGX_Thor"
    
    # Datacenter GPUs
    datacenter_systems = [
        "A100_80GB",
        "H100",
        "B100",
    ]
    
    num_devices = 1  # Single device comparison
    batch_size = 1   # Always use batch_size=1 for robotics
    
    # Create image and action configs for network latency calculation
    image_config = ImageConfig(
        resolution=image_resolution,
        channels=3,
        bytes_per_pixel=1,
        compression_ratio=image_compression_ratio,
    )
    
    action_config = ActionConfig(
        num_dof=action_dof,
        action_chunk_size=action_chunk_size,
        bytes_per_value=4,  # float32
    )
    
    results = []
    
    logger.info(f"\n\nSettings: {denoising_steps} denoising steps, batch_size=1")
    logger.info(f"Image: {image_config.name}, Action: {action_config.name}")
    
    # Get edge device performance (no network latency)
    try:
        # Precision is automatically selected based on system capabilities
        df_vision = get_pi0_vision_perf(
            config, [edge_system], [num_devices], bits, max_batch_size=1
        )
        df_vlm = get_pi0_vlm_perf(
            config, [edge_system], [num_devices], bits, max_batch_size=1
        )
        df_action = get_pi0_action_expert_perf(
            config, [edge_system], [num_devices], bits, denoising_steps,
            vlm_sequence_length=config.vlm_sequence_length,
            max_batch_size=1
        )
        
        if df_vision.empty or df_vlm.empty or df_action.empty:
            logger.warning(f"  {edge_system}: Skipped (memory constraints)")
        else:
            vision_time = df_vision["time_ms"].values[0]
            vlm_time = df_vlm["time_ms"].values[0]
            action_time = df_action["time_ms"].values[0]
            e2e_time = vision_time + vlm_time + action_time
            inference_hz = 1000 / e2e_time
            
            results.append({
                "model": config.name,
                "system": edge_system,
                "category": "Edge (Local)",
                "network": "N/A (Local)",
                "precision": bits,
                "vision_ms": vision_time,
                "vlm_ms": vlm_time,
                "action_ms": action_time,
                "network_image_ms": 0.0,
                "network_action_ms": 0.0,
                "e2e_compute_ms": e2e_time,
                "e2e_total_ms": e2e_time,  # No network latency
                "frequency_hz": inference_hz,
                "freq_async_hz": inference_hz,  # No network bottleneck for local
                "denoising_steps": denoising_steps,
            })
    except Exception as e:
        logger.warning(f"  {edge_system}: Error - {str(e)[:50]}")
    
    # Get datacenter GPU performance with network latency
    for dc_system in datacenter_systems:
        try:
            # Precision is automatically selected based on system capabilities
            df_vision = get_pi0_vision_perf(
                config, [dc_system], [num_devices], bits, max_batch_size=1
            )
            df_vlm = get_pi0_vlm_perf(
                config, [dc_system], [num_devices], bits, max_batch_size=1
            )
            df_action = get_pi0_action_expert_perf(
                config, [dc_system], [num_devices], bits, denoising_steps,
                vlm_sequence_length=config.vlm_sequence_length,
                max_batch_size=1
            )
            
            if df_vision.empty or df_vlm.empty or df_action.empty:
                logger.warning(f"  {dc_system}: Skipped (memory constraints)")
                continue
            
            vision_time = df_vision["time_ms"].values[0]
            vlm_time = df_vlm["time_ms"].values[0]
            action_time = df_action["time_ms"].values[0]
            e2e_compute = vision_time + vlm_time + action_time
            inference_hz = 1000.0 / e2e_compute
            
            # Add network latency for each network config
            for net_config in network_configs:
                # Image upload latency (Robot → Server)
                image_latency = estimate_image_latency(net_config, image_config)
                network_image_ms = image_latency["total_latency_ms"]
                
                # Action download latency (Server → Robot)
                action_latency = estimate_action_latency(net_config, action_config)
                network_action_ms = action_latency["total_latency_ms"]
                
                # Total E2E latency = compute + network (image upload + action download)
                e2e_total = e2e_compute + network_image_ms + network_action_ms
                
                # Compute network throughput Hz (excluding base latency, for continuous streaming)
                network_hz = compute_network_throughput_hz(
                    network_config=net_config,
                    image_config=image_config,
                    action_config=action_config,
                )
                
                # Async frequency: min(inference_hz, network_hz)
                # This assumes continuous streaming where network and inference can overlap
                freq_async_hz = min(inference_hz, network_hz)
                
                results.append({
                    "model": config.name,
                    "system": dc_system,
                    "category": "Data Center (Remote)",
                    "network": net_config.name,
                    "precision": bits,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "network_image_ms": network_image_ms,
                    "network_action_ms": network_action_ms,
                    "e2e_compute_ms": e2e_compute,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": freq_async_hz,
                    "denoising_steps": denoising_steps,
                })
                
        except Exception as e:
            logger.warning(f"  {dc_system}: Error - {str(e)[:50]}")
            continue
    
    return pd.DataFrame(results)


def print_datacenter_vs_edge_with_network_summary(df: pd.DataFrame, logger=None) -> None:
    """Print a formatted summary comparing datacenter vs edge performance with network latency."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.info("No results to display.")
        return
    
    # Find best configurations
    edge_df = df[df["category"] == "Edge (Local)"]
    dc_df = df[df["category"] == "Data Center (Remote)"]
    
    if not edge_df.empty and not dc_df.empty:
        best_edge = edge_df["e2e_total_ms"].min()
        best_edge_sys = edge_df.loc[edge_df["e2e_total_ms"].idxmin(), "system"]
        
        best_dc = dc_df["e2e_total_ms"].min()
        best_dc_row = dc_df.loc[dc_df["e2e_total_ms"].idxmin()]
        best_dc_sys = best_dc_row["system"]
        best_dc_net = best_dc_row["network"]
        
        # Also find best async frequency
        best_edge_async = edge_df["freq_async_hz"].max() if "freq_async_hz" in edge_df.columns else edge_df["frequency_hz"].max()
        best_dc_async_row = dc_df.loc[dc_df["freq_async_hz"].idxmax()] if "freq_async_hz" in dc_df.columns else dc_df.loc[dc_df["frequency_hz"].idxmax()]
        best_dc_async_sys = best_dc_async_row["system"]
        best_dc_async_net = best_dc_async_row["network"]
        best_dc_async = best_dc_async_row.get("freq_async_hz", best_dc_async_row["frequency_hz"])
        
        logger.info("\n" + "-" * 140)
        logger.info("Summary:")
        logger.info(f"  Best Edge (Local):  {best_edge_sys} @ {best_edge:.2f} ms ({1000/best_edge:.1f} Hz sync, {best_edge_async:.1f} Hz async)")
        logger.info(f"  Best Data Center:   {best_dc_sys} + {best_dc_net} @ {best_dc:.2f} ms ({1000/best_dc:.1f} Hz sync, {best_dc_async:.1f} Hz async)")
        logger.info(f"  DC/Edge Speed Ratio (Sync Inference):      {best_edge/best_dc:.2f}x")
        logger.info(f"  DC/Edge Speed Ratio (Async Inference):      {best_dc_async/best_edge_async:.2f}x")
        
    # Print edge device results
    if not edge_df.empty:
        logger.info("\n" + "-" * 130)
        logger.info("Edge Device (Local - No Network Latency)")
        logger.info("-" * 130)
        logger.info(f"{'System':<25} {'Prec':<6} {'Vision':>10} {'VLM':>10} {'Action':>10} {'Compute':>10} {'Total':>10} {'Freq (Sync)':>10} {'Freq (Async)':>12}")
        logger.info(f"{'':25} {'':6} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(Hz)':>10} {'(Hz)':>12}")
        logger.info("-" * 130)
        
        for _, row in edge_df.iterrows():
            freq_async = row.get('freq_async_hz', row['frequency_hz'])
            logger.info(f"{row['system']:<25} "
                  f"{row['precision']:<6} "
                  f"{row['vision_ms']:>10.2f} "
                  f"{row['vlm_ms']:>10.2f} "
                  f"{row['action_ms']:>10.2f} "
                  f"{row['e2e_compute_ms']:>10.2f} "
                  f"{row['e2e_total_ms']:>10.2f} "
                  f"{row['frequency_hz']:>10.1f} "
                  f"{freq_async:>12.1f}")
    
    # Print datacenter results grouped by system
    if not dc_df.empty:
        logger.info("\n" + "-" * 140)
        logger.info("Data Center GPUs (Remote - With Network Latency)")
        logger.info("-" * 140)
        
        for dc_system in dc_df["system"].unique():
            sys_df = dc_df[dc_df["system"] == dc_system].sort_values("e2e_total_ms")
            
            logger.info(f"\n{dc_system}:")
            logger.info("-" * 140)
            logger.info(f"{'Network':<30} {'Prec':<6} {'Compute':>10} {'Img Net':>10} {'Act Net':>10} {'Total':>10} {'Freq (Sync)':>10} {'Freq (Async)':>12}")
            logger.info(f"{'':30} {'':6} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(Hz)':>10} {'(Hz)':>12}")
            logger.info("-" * 140)
            
            for _, row in sys_df.iterrows():
                freq_async = row.get('freq_async_hz', row['frequency_hz'])
                logger.info(f"{row['network']:<30} "
                      f"{row['precision']:<6} "
                      f"{row['e2e_compute_ms']:>10.2f} "
                      f"{row['network_image_ms']:>10.2f} "
                      f"{row['network_action_ms']:>10.2f} "
                      f"{row['e2e_total_ms']:>10.2f} "
                      f"{row['frequency_hz']:>10.1f} "
                      f"{freq_async:>12.1f}")


def print_minimum_config_for_frequencies(df: pd.DataFrame, logger=None) -> None:
    """
    Print the minimum hardware configuration needed to reach 10Hz, 100Hz, and 1000Hz
    for both synchronous and asynchronous inference.
    
    Shows all devices and their configurations that can barely meet each target
    (lowest frequency that still meets the requirement, not the best performance).
    
    Args:
        df: DataFrame with performance results
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.info("No results to analyze.")
        return
    
    target_frequencies = [10, 100, 1000]
    
    logger.info("\n" + "=" * 140)
    logger.info("MINIMUM CONFIGURATIONS TO REACH TARGET FREQUENCIES")
    logger.info("(Shows configuration that can barely meet the target - lowest Hz above threshold)")
    logger.info("=" * 140)
    
    # Analyze for Edge (Local) configurations
    edge_df = df[df["category"] == "Edge (Local)"].copy()
    if not edge_df.empty:
        logger.info("\n" + "-" * 140)
        logger.info("Edge Devices (Local Inference)")
        logger.info("-" * 140)
        
        for freq_target in target_frequencies:
            logger.info(f"\nTarget: {freq_target} Hz")
            logger.info(f"  {'Mode':<15} {'Device':<25} {'Precision':<10} {'Frequency (Hz)':>15} {'Latency (ms)':>15}")
            logger.info(f"  {'-'*15} {'-'*25} {'-'*10} {'-'*15} {'-'*15}")
            
            # Synchronous inference - show all devices
            sync_capable = edge_df[edge_df["frequency_hz"] >= freq_target]
            if not sync_capable.empty:
                # Group by system and find the config that barely meets target for each
                for system in sorted(edge_df["system"].unique()):
                    system_capable = sync_capable[sync_capable["system"] == system]
                    if not system_capable.empty:
                        # Find the one with lowest frequency (barely meets requirement)
                        min_config = system_capable.loc[system_capable["frequency_hz"].idxmin()]
                        logger.info(f"  {'Synchronous':<15} {min_config['system']:<25} {min_config['precision']:<10} "
                                  f"{min_config['frequency_hz']:>15.1f} {min_config['e2e_total_ms']:>15.2f}")
            else:
                logger.info(f"  {'Synchronous':<15} No device can reach {freq_target} Hz")
            
            # Asynchronous inference - show all devices
            if "freq_async_hz" in edge_df.columns:
                async_capable = edge_df[edge_df["freq_async_hz"] >= freq_target]
                if not async_capable.empty:
                    # Group by system and find the config that barely meets target for each
                    for system in sorted(edge_df["system"].unique()):
                        system_capable = async_capable[async_capable["system"] == system]
                        if not system_capable.empty:
                            # Find the one with lowest frequency (barely meets requirement)
                            min_config = system_capable.loc[system_capable["freq_async_hz"].idxmin()]
                            logger.info(f"  {'Asynchronous':<15} {min_config['system']:<25} {min_config['precision']:<10} "
                                      f"{min_config['freq_async_hz']:>15.1f} {min_config['e2e_total_ms']:>15.2f}")
                else:
                    logger.info(f"  {'Asynchronous':<15} No device can reach {freq_target} Hz")
    
    # Analyze for Data Center (Remote) configurations
    dc_df = df[df["category"] == "Data Center (Remote)"].copy()
    if not dc_df.empty:
        logger.info("\n" + "-" * 140)
        logger.info("Data Center GPUs (Remote Inference with Network)")
        logger.info("-" * 140)
        
        for freq_target in target_frequencies:
            logger.info(f"\nTarget: {freq_target} Hz")
            logger.info(f"  {'Mode':<15} {'Device':<25} {'Network':<30} {'Precision':<10} {'Frequency (Hz)':>15} {'Latency (ms)':>15}")
            logger.info(f"  {'-'*15} {'-'*25} {'-'*30} {'-'*10} {'-'*15} {'-'*15}")
            
            # Synchronous inference - show all devices
            sync_capable = dc_df[dc_df["frequency_hz"] >= freq_target]
            if not sync_capable.empty:
                # Group by system and find the config that barely meets target for each
                for system in sorted(dc_df["system"].unique()):
                    system_capable = sync_capable[sync_capable["system"] == system]
                    if not system_capable.empty:
                        # Find the one with lowest frequency (barely meets requirement)
                        min_config = system_capable.loc[system_capable["frequency_hz"].idxmin()]
                        logger.info(f"  {'Synchronous':<15} {min_config['system']:<25} {min_config['network']:<30} "
                                  f"{min_config['precision']:<10} {min_config['frequency_hz']:>15.1f} {min_config['e2e_total_ms']:>15.2f}")
            else:
                logger.info(f"  {'Synchronous':<15} No device can reach {freq_target} Hz")
            
            # Asynchronous inference - show all devices
            if "freq_async_hz" in dc_df.columns:
                async_capable = dc_df[dc_df["freq_async_hz"] >= freq_target]
                if not async_capable.empty:
                    # Group by system and find the config that barely meets target for each
                    for system in sorted(dc_df["system"].unique()):
                        system_capable = async_capable[async_capable["system"] == system]
                        if not system_capable.empty:
                            # Find the one with lowest frequency (barely meets requirement)
                            min_config = system_capable.loc[system_capable["freq_async_hz"].idxmin()]
                            logger.info(f"  {'Asynchronous':<15} {min_config['system']:<25} {min_config['network']:<30} "
                                      f"{min_config['precision']:<10} {min_config['freq_async_hz']:>15.1f} {min_config['e2e_total_ms']:>15.2f}")
                else:
                    logger.info(f"  {'Asynchronous':<15} No device can reach {freq_target} Hz")
    
    logger.info("\n" + "=" * 140)


def run_datacenter_vs_edge_with_network_comparison(
    output_dir: str = "perf_results",
    network_configs: list[NetworkConfig] = None,
    experiment_num: int = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Run datacenter vs edge comparison with network latency for all Pi0 family models.
    
    Args:
        output_dir: Directory to save results
        network_configs: List of network configurations to test (default: all)
        experiment_num: Experiment number for logging (optional)
        logger: Logger instance (optional)
    
    Returns:
        Dictionary mapping model names to comparison DataFrames
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if network_configs is None:
        network_configs = ALL_NETWORK_CONFIGS
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    all_dfs = []
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}DATA CENTER + NETWORK vs EDGE COMPARISON")
    logger.info("=" * 150)
    logger.info(f"Testing {len(network_configs)} network configurations")
    
    for config in ALL_PI0_CONFIGS:
        df = compare_datacenter_vs_edge_with_network(
            config=config,
            denoising_steps=10,
            bits="bf16",
            network_configs=network_configs,
            logger=logger,
        )
        
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        
        logger.info(f"\n{config.name}:")
        print_datacenter_vs_edge_with_network_summary(df, logger=logger)
        print_minimum_config_for_frequencies(df, logger=logger)
    
    # Save combined results
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        output_file = output_path / "pi0_datacenter_vs_edge_with_network.csv"
        df_combined.to_csv(output_file, index=False)
        logger.info(f"\n\nResults saved to {output_file}")
    
    return all_results


def compare_device_vs_server(
    config: Pi0Config = PI0_CONFIG,
    denoising_steps: int = 10,
    bits: str = "bf16",
    image_resolution: int = 384,  # Typical Pi0 image resolution (SigLIP)
    image_compression_ratio: float = 0.1,  # JPEG compression
    action_dof: int = 8,  # 7-DoF arm + gripper
    action_chunk_size: int = 1,
    logger=None,
) -> pd.DataFrame:
    """
    Compare on-device vs edge-server vs cloud inference for VLA:
    
    1. On-device inference: 
       - Device: Jetson Thor
       - Inference: Local on robot (no network latency)
       
    2. Edge-server inference:
       - Device: RTX 4090, B100
       - Network: Wired (Ethernet 1GbE, 10GbE) or Wireless (WiFi 6/7)
       - Communication: Robot ↔ Edge server
       
    3. Cloud inference:
       - Device: B100
       - Network: Local network (Ethernet/WiFi) + Cloud network (Fast: 10Gbps 10ms, Slow: 1Gbps 100ms)
       - Communication: Robot → Gateway → Cloud → Gateway → Robot
    
    Args:
        config: Pi0 model config to evaluate
        denoising_steps: Number of flow matching denoising steps
        bits: Preferred precision for computation
        image_resolution: Image resolution (width/height, assuming square)
        image_compression_ratio: Image compression ratio (0.1 = JPEG 10:1)
        action_dof: Number of degrees of freedom for actions
        action_chunk_size: Number of future actions (action chunking)
        logger: Logger instance (optional)
        
    Returns:
        DataFrame with performance comparison across deployment scenarios
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    num_devices = 1  # Single device comparison
    batch_size = 1   # Always use batch_size=1 for robotics
    
    # Create image and action configs for network latency calculation
    image_config = ImageConfig(
        resolution=image_resolution,
        channels=3,
        bytes_per_pixel=1,
        compression_ratio=image_compression_ratio,
    )
    
    action_config = ActionConfig(
        num_dof=action_dof,
        action_chunk_size=action_chunk_size,
        bytes_per_value=4,  # float32
    )
    
    results = []
    
    logger.info(f"\n\nSettings: {denoising_steps} denoising steps, batch_size=1")
    logger.info(f"Image: {image_config.name}, Action: {action_config.name}")
    
    # ========================================================================
    # CATEGORY 1: ON-DEVICE INFERENCE (Jetson Thor, RTX 4090, B100)
    # ========================================================================
    on_device_systems = ["Jetson_AGX_Thor", "RTX_4090", "B100"]
    
    for on_device_system in on_device_systems:
        try:
            df_vision = get_pi0_vision_perf(
                config, [on_device_system], [num_devices], bits, max_batch_size=1
            )
            df_vlm = get_pi0_vlm_perf(
                config, [on_device_system], [num_devices], bits, max_batch_size=1
            )
            df_action = get_pi0_action_expert_perf(
                config, [on_device_system], [num_devices], bits, denoising_steps,
                vlm_sequence_length=config.vlm_sequence_length,
                max_batch_size=1
            )
            
            if not (df_vision.empty or df_vlm.empty or df_action.empty):
                vision_time = df_vision["time_ms"].values[0]
                vlm_time = df_vlm["time_ms"].values[0]
                action_time = df_action["time_ms"].values[0]
                e2e_time = vision_time + vlm_time + action_time
                inference_hz = 1000 / e2e_time
                
                results.append({
                    "model": config.name,
                    "category": "On-device",
                    "system": on_device_system,
                    "network": "N/A (Local)",
                    "precision": bits,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "network_image_ms": 0.0,
                    "network_action_ms": 0.0,
                    "e2e_compute_ms": e2e_time,
                    "e2e_total_ms": e2e_time,
                    "frequency_hz": inference_hz,
                    "freq_async_hz": inference_hz,
                    "denoising_steps": denoising_steps,
                })
            else:
                logger.warning(f"  {on_device_system}: Skipped (memory constraints)")
        except Exception as e:
            logger.warning(f"  {on_device_system}: Error - {str(e)[:50]}")
    
    # ========================================================================
    # CATEGORY 2: EDGE-SERVER INFERENCE (RTX 4090, B100)
    # ========================================================================
    edge_server_systems = ["RTX_4090", "B100"]
    edge_networks = [
        ETHERNET_1G_CONFIG,   # Wired: 1GbE Ethernet
        ETHERNET_10G_CONFIG,  # Wired: 10GbE Ethernet
        WIFI_6_CONFIG,  # Wireless: WiFi 6
        WIFI_7_CONFIG,  # Wireless: WiFi 7
        CELL_5G_SA_CONFIG,  # Cellular: 5G
        CELL_4G_LTE_CONFIG,  # Cellular: 4G LTE
    ]
    
    for edge_server_system in edge_server_systems:
        try:
            df_vision = get_pi0_vision_perf(
                config, [edge_server_system], [num_devices], bits, max_batch_size=1
            )
            df_vlm = get_pi0_vlm_perf(
                config, [edge_server_system], [num_devices], bits, max_batch_size=1
            )
            df_action = get_pi0_action_expert_perf(
                config, [edge_server_system], [num_devices], bits, denoising_steps,
                vlm_sequence_length=config.vlm_sequence_length,
                max_batch_size=1
            )
            
            if df_vision.empty or df_vlm.empty or df_action.empty:
                logger.warning(f"  {edge_server_system}: Skipped (memory constraints)")
                continue
                
            vision_time = df_vision["time_ms"].values[0]
            vlm_time = df_vlm["time_ms"].values[0]
            action_time = df_action["time_ms"].values[0]
            e2e_compute = vision_time + vlm_time + action_time
            inference_hz = 1000.0 / e2e_compute
            
            # Add network latency for each edge network config
            for net_config in edge_networks:
                # Image upload latency (Robot → Server)
                image_latency = estimate_image_latency(net_config, image_config)
                network_image_ms = image_latency["total_latency_ms"]
                
                # Action download latency (Server → Robot)
                action_latency = estimate_action_latency(net_config, action_config)
                network_action_ms = action_latency["total_latency_ms"]
                
                # Total E2E latency = compute + network
                e2e_total = e2e_compute + network_image_ms + network_action_ms
                
                # Compute network throughput Hz
                network_hz = compute_network_throughput_hz(
                    network_config=net_config,
                    image_config=image_config,
                    action_config=action_config,
                )
                
                # Async frequency: min(inference_hz, network_hz)
                freq_async_hz = min(inference_hz, network_hz)
                
                results.append({
                    "model": config.name,
                    "category": "Edge-server",
                    "system": edge_server_system,
                    "network": net_config.name,
                    "precision": bits,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "network_image_ms": network_image_ms,
                    "network_action_ms": network_action_ms,
                    "e2e_compute_ms": e2e_compute,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": freq_async_hz,
                    "denoising_steps": denoising_steps,
                })
                    
        except Exception as e:
            logger.warning(f"  {edge_server_system}: Error - {str(e)[:50]}")
    
    # ========================================================================
    # CATEGORY 3: CLOUD INFERENCE (B100)
    # ========================================================================
    cloud_system = "B100"
    
    # For cloud inference, we use fixed network pairs:
    # 1. Wired (Ethernet 10GbE) + Fast cloud (10Gbps, 10ms)
    # 2. Cellular (4G LTE) + Slow cloud (1Gbps, 100ms)
    
    network_pairs = [
        ("Wired + Fast Cloud", ETHERNET_10G_CONFIG, CLOUD_FAST_CONFIG),
        ("4G + Slow Cloud", CELL_4G_LTE_CONFIG, CLOUD_SLOW_CONFIG),
    ]
    
    try:
        df_vision = get_pi0_vision_perf(
            config, [cloud_system], [num_devices], bits, max_batch_size=1
        )
        df_vlm = get_pi0_vlm_perf(
            config, [cloud_system], [num_devices], bits, max_batch_size=1
        )
        df_action = get_pi0_action_expert_perf(
            config, [cloud_system], [num_devices], bits, denoising_steps,
            vlm_sequence_length=config.vlm_sequence_length,
            max_batch_size=1
        )
        
        if df_vision.empty or df_vlm.empty or df_action.empty:
            logger.warning(f"  {cloud_system}: Skipped (memory constraints)")
        else:
            vision_time = df_vision["time_ms"].values[0]
            vlm_time = df_vlm["time_ms"].values[0]
            action_time = df_action["time_ms"].values[0]
            e2e_compute = vision_time + vlm_time + action_time
            inference_hz = 1000.0 / e2e_compute
            
            # Test fixed network pairs
            for network_name, local_net, cloud_net in network_pairs:
                # Local network latency (Robot → Gateway)
                image_latency_local = estimate_image_latency(local_net, image_config)
                action_latency_local = estimate_action_latency(local_net, action_config)
                
                # Cloud network latency (Gateway → Cloud)
                image_latency_cloud = estimate_image_latency(cloud_net, image_config)
                action_latency_cloud = estimate_action_latency(cloud_net, action_config)
                
                # Total network latency = local + cloud (bidirectional)
                network_image_ms = image_latency_local["total_latency_ms"] + image_latency_cloud["total_latency_ms"]
                network_action_ms = action_latency_local["total_latency_ms"] + action_latency_cloud["total_latency_ms"]
                
                # Total E2E latency = compute + network
                e2e_total = e2e_compute + network_image_ms + network_action_ms
                
                # Network throughput considering both local and cloud hops
                # For simplicity, we take the bottleneck (min of both)
                network_hz_local = compute_network_throughput_hz(
                    network_config=local_net,
                    image_config=image_config,
                    action_config=action_config,
                )
                network_hz_cloud = compute_network_throughput_hz(
                    network_config=cloud_net,
                    image_config=image_config,
                    action_config=action_config,
                )
                network_hz = min(network_hz_local, network_hz_cloud)
                
                # Async frequency: min(inference_hz, network_hz)
                freq_async_hz = min(inference_hz, network_hz)
                
                results.append({
                    "model": config.name,
                    "category": "Cloud",
                    "system": cloud_system,
                    "network": network_name,
                    "precision": bits,
                    "vision_ms": vision_time,
                    "vlm_ms": vlm_time,
                    "action_ms": action_time,
                    "network_image_ms": network_image_ms,
                    "network_action_ms": network_action_ms,
                    "e2e_compute_ms": e2e_compute,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": freq_async_hz,
                    "denoising_steps": denoising_steps,
                })
                    
    except Exception as e:
        logger.warning(f"  {cloud_system}: Error - {str(e)[:50]}")
    
    return pd.DataFrame(results)


def print_device_vs_server_summary(df: pd.DataFrame, logger=None) -> None:
    """Print a formatted summary comparing on-device vs edge-server vs cloud inference."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.info("No results to display.")
        return
    
    # Print summary statistics
    logger.info("\n" + "-" * 160)
    logger.info("Summary by Category:")
    logger.info("-" * 160)
    
    for category in ["On-device", "Edge-server", "Cloud"]:
        cat_df = df[df["category"] == category]
        if not cat_df.empty:
            best_total = cat_df["e2e_total_ms"].min()
            best_row = cat_df.loc[cat_df["e2e_total_ms"].idxmin()]
            best_async = cat_df["freq_async_hz"].max()
            best_async_row = cat_df.loc[cat_df["freq_async_hz"].idxmax()]
            
            logger.info(f"\n{category}:")
            logger.info(f"  Best System:       {best_row['system']}")
            logger.info(f"  Best Network:      {best_row['network']}")
            logger.info(f"  Best Latency:      {best_total:.2f} ms ({1000/best_total:.1f} Hz sync)")
            logger.info(f"  Best Async Freq:   {best_async:.1f} Hz ({best_async_row['network']})")
    
    # Print detailed results by category
    for category in ["On-device", "Edge-server", "Cloud"]:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue
            
        logger.info("\n" + "-" * 160)
        logger.info(f"{category} Inference")
        logger.info("-" * 160)
        logger.info(f"{'System':<20} {'Network':<35} {'Prec':<6} {'Compute':>10} {'Img Net':>10} {'Act Net':>10} {'Total':>10} {'Freq(Sync)':>12} {'Freq(Async)':>12}")
        logger.info(f"{'':20} {'':35} {'':6} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(Hz)':>12} {'(Hz)':>12}")
        logger.info("-" * 160)
        
        # Sort by total latency
        cat_df_sorted = cat_df.sort_values("e2e_total_ms")
        
        for _, row in cat_df_sorted.iterrows():
            logger.info(f"{row['system']:<20} "
                  f"{row['network']:<35} "
                  f"{row['precision']:<6} "
                  f"{row['e2e_compute_ms']:>10.2f} "
                  f"{row['network_image_ms']:>10.2f} "
                  f"{row['network_action_ms']:>10.2f} "
                  f"{row['e2e_total_ms']:>10.2f} "
                  f"{row['frequency_hz']:>12.1f} "
                  f"{row['freq_async_hz']:>12.1f}")


def run_device_vs_server_comparison(
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Run on-device vs edge-server vs cloud comparison for all Pi0 family models.
    
    Evaluates three types of VLA inference systems:
    1. On-device inference: Jetson Thor (local execution)
    2. Edge-server inference: RTX 4090, B100 with wired/wireless networks
    3. Cloud inference: B100 with local + cloud network latency
    
    Args:
        output_dir: Directory to save results
        experiment_num: Experiment number for logging (optional)
        logger: Logger instance (optional)
    
    Returns:
        Dictionary mapping model names to comparison DataFrames
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    all_dfs = []
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 160)
    logger.info(f"{exp_header}DEVICE VS SERVER COMPARISON")
    logger.info("=" * 160)
    logger.info("\nEvaluating three categories:")
    logger.info("  1. On-device inference:    Jetson Thor (local, no network)")
    logger.info("  2. Edge-server inference:  RTX 4090, B100 + Wired/WiFi networks")
    logger.info("  3. Cloud inference:        B100 + Local network + Cloud network")
    logger.info("")
    logger.info("Network configurations:")
    logger.info("  - On-device:    No network (local inference)")
    logger.info("  - Edge-server:  Ethernet 1GbE, Ethernet 10GbE (wired), WiFi 6, WiFi 7 (wireless), 5G, 4G (cellular)")
    logger.info("  - Cloud:        Fixed pairs: Wired (Ethernet 10GbE) + Fast Cloud, 4G + Slow Cloud")
    
    for config in ALL_PI0_CONFIGS:
        df = compare_device_vs_server(
            config=config,
            denoising_steps=10,
            bits="bf16",
            logger=logger,
        )
        
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        
        logger.info(f"\n{config.name}:")
        print_device_vs_server_summary(df, logger=logger)
    
    # Save combined results
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        output_file = output_path / "pi0_device_vs_server.csv"
        df_combined.to_csv(output_file, index=False)
        logger.info(f"\n\nResults saved to {output_file}")
    
    return all_results


# ==============================================================================
# Experiment 7: Device-Server Collaboration (Helix-style split inference)
# Vision runs on-device, VLM on server, KV cache sent back, action on-device.
# Compares against server-only and device-only baselines.
# ==============================================================================

def compare_device_server_collaboration(
    config: Pi0Config = PI0_CONFIG,
    denoising_steps: int = 10,
    bits: str = "bf16",
    image_resolution: int = 384,  # Typical Pi0 image resolution (SigLIP)
    image_compression_ratio: float = 0.1,  # JPEG compression
    action_dof: int = 8,  # 7-DoF arm + gripper
    action_chunk_size: int = 1,
    server_system: str = "B100",  # Default server GPU
    device_system: str = "Jetson_AGX_Thor",  # Default device GPU
    network_configs: list[NetworkConfig] = None,
    logger=None,
) -> pd.DataFrame:
    """
    Compare Device-Server Collaboration performance for Pi0, including baselines.
    
    Three scenarios:
    1. Device-Server Collaboration:
       - Device robot uploads image to server (Robot → Server)
       - Server GPU runs VLM backbone inference
       - KV cache of VLM is downloaded to robot (Server → Robot)
       - Device GPU runs diffusion inference (action expert)
    
    2. Server Only:
       - Device robot uploads image to server (Robot → Server)
       - Server GPU runs vision + VLM + action expert inference
       - Action is downloaded to robot (Server → Robot)
    
    3. Device Only:
       - Device GPU runs vision + VLM + action expert locally (no network)
    
    Args:
        config: Pi0 model config to evaluate
        denoising_steps: Number of flow matching denoising steps
        bits: Preferred precision for computation
        image_resolution: Image resolution (width/height, assuming square)
        image_compression_ratio: Image compression ratio (0.1 = JPEG 10:1)
        action_dof: Number of degrees of freedom for actions
        action_chunk_size: Number of future actions (action chunking)
        server_system: Server GPU system (default: B100)
        device_system: Device GPU system (default: Jetson_AGX_Thor)
        network_configs: List of network configurations to test (default: all)
        logger: Logger instance (optional)
        
    Returns:
        DataFrame with performance comparison for all three scenarios
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if network_configs is None:
        # Use all network configs by default
        network_configs = ALL_NETWORK_CONFIGS
    
    num_devices = 1  # Single device comparison
    
    # Create image and action configs for network latency calculation
    image_config = ImageConfig(
        resolution=image_resolution,
        channels=3,
        bytes_per_pixel=1,
        compression_ratio=image_compression_ratio,
    )
    
    action_config = ActionConfig(
        num_dof=action_dof,
        action_chunk_size=action_chunk_size,
        bytes_per_value=4,  # float32
    )
    
    # Create KV cache config for the VLM model
    kv_cache_configs = get_kvcache_configs_from_model_config(
        model_name=config.vlm_model,
        seq_lengths=[config.vlm_sequence_length],
        pretty_name=f"{config.name}-VLM",
    )
    
    if not kv_cache_configs:
        logger.warning(f"Could not create KV cache config for {config.vlm_model}")
        return pd.DataFrame()
    
    kv_cache_config = kv_cache_configs[0]  # Use first config (should only be one)
    
    results = []
    
    logger.info(f"\n\nSettings: {denoising_steps} denoising steps, batch_size=1")
    logger.info(f"Image: {image_config.name}, Action: {action_config.name}, KV Cache: {kv_cache_config.name}")
    logger.info(f"Server GPU: {server_system}, Device GPU: {device_system}")
    
    # ===== SCENARIO 1: Device Only (baseline) =====
    try:
        # Precision is automatically selected based on system capabilities
        df_vision_device = get_pi0_vision_perf(
            config, [device_system], [num_devices], bits, max_batch_size=1
        )
        df_vlm_device = get_pi0_vlm_perf(
            config, [device_system], [num_devices], bits, max_batch_size=1
        )
        df_action_device = get_pi0_action_expert_perf(
            config, [device_system], [num_devices], bits, denoising_steps,
            vlm_sequence_length=config.vlm_sequence_length,
            max_batch_size=1
        )
        
        if df_vision_device.empty or df_vlm_device.empty or df_action_device.empty:
            logger.warning(f"  {device_system}: Skipped (memory constraints)")
        else:
            vision_time_device = df_vision_device["time_ms"].values[0]
            vlm_time_device = df_vlm_device["time_ms"].values[0]
            action_time_device = df_action_device["time_ms"].values[0]
            e2e_device = vision_time_device + vlm_time_device + action_time_device
            inference_hz_device = 1000.0 / e2e_device
            
            results.append({
                "model": config.name,
                "category": "Device Only",
                "server_system": "N/A",
                "device_system": device_system,
                "network": "N/A (Local)",
                "precision": bits,
                "vision_ms": vision_time_device,
                "vlm_ms": vlm_time_device,
                "action_ms": action_time_device,
                "network_image_ms": 0.0,
                "network_action_ms": 0.0,
                "network_kv_cache_ms": 0.0,
                "e2e_total_ms": e2e_device,
                "frequency_hz": inference_hz_device,
                "freq_async_hz": inference_hz_device,
                "denoising_steps": denoising_steps,
            })
    except Exception as e:
        logger.warning(f"  {device_system} (Device Only): Error - {str(e)[:50]}")
    
    # ===== SCENARIO 2: Server Only =====
    try:
        # Precision is automatically selected based on system capabilities
        df_vision_server = get_pi0_vision_perf(
            config, [server_system], [num_devices], bits, max_batch_size=1
        )
        df_vlm_server = get_pi0_vlm_perf(
            config, [server_system], [num_devices], bits, max_batch_size=1
        )
        df_action_server = get_pi0_action_expert_perf(
            config, [server_system], [num_devices], bits, denoising_steps,
            vlm_sequence_length=config.vlm_sequence_length,
            max_batch_size=1
        )
        
        if df_vision_server.empty or df_vlm_server.empty or df_action_server.empty:
            logger.warning(f"  {server_system}: Skipped (memory constraints)")
        else:
            vision_time_server = df_vision_server["time_ms"].values[0]
            vlm_time_server = df_vlm_server["time_ms"].values[0]
            action_time_server = df_action_server["time_ms"].values[0]
            e2e_compute_server = vision_time_server + vlm_time_server + action_time_server
            inference_hz_server = 1000.0 / e2e_compute_server
            
            # Add network latency for each network config
            for net_config in network_configs:
                # Image upload latency (Robot → Server)
                image_latency = estimate_image_latency(net_config, image_config)
                network_image_ms = image_latency["total_latency_ms"]
                
                # Action download latency (Server → Robot)
                action_latency = estimate_action_latency(net_config, action_config)
                network_action_ms = action_latency["total_latency_ms"]
                
                # Total E2E latency = network (image upload) + compute + network (action download)
                e2e_total = network_image_ms + e2e_compute_server + network_action_ms
                
                # Compute network throughput Hz
                network_hz = compute_network_throughput_hz(
                    network_config=net_config,
                    image_config=image_config,
                    action_config=action_config,
                )
                
                # Async frequency: min(inference_hz, network_hz)
                freq_async_hz = min(inference_hz_server, network_hz)
                
                results.append({
                    "model": config.name,
                    "category": "Server Only",
                    "server_system": server_system,
                    "device_system": "N/A",
                    "network": net_config.name,
                    "precision": bits,
                    "vision_ms": vision_time_server,
                    "vlm_ms": vlm_time_server,
                    "action_ms": action_time_server,
                    "network_image_ms": network_image_ms,
                    "network_action_ms": network_action_ms,
                    "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": freq_async_hz,
                    "denoising_steps": denoising_steps,
                })
            
            # Add cloud network pairs (two-hop: local + cloud) for Server Only as well
            cloud_network_pairs = [
                ("Wired + Fast Cloud", ETHERNET_10G_CONFIG, CLOUD_FAST_CONFIG),
                ("4G + Slow Cloud", CELL_4G_LTE_CONFIG, CLOUD_SLOW_CONFIG),
            ]
            
            for network_name, local_net, cloud_net in cloud_network_pairs:
                # Local network latency (Robot → Gateway)
                image_latency_local = estimate_image_latency(local_net, image_config)
                action_latency_local = estimate_action_latency(local_net, action_config)
                
                # Cloud network latency (Gateway → Server)
                image_latency_cloud = estimate_image_latency(cloud_net, image_config)
                action_latency_cloud = estimate_action_latency(cloud_net, action_config)
                
                # Total network latency = local + cloud (bidirectional)
                network_image_ms = image_latency_local["total_latency_ms"] + image_latency_cloud["total_latency_ms"]
                network_action_ms = action_latency_local["total_latency_ms"] + action_latency_cloud["total_latency_ms"]
                
                # Total E2E latency = compute + network
                e2e_total = e2e_compute_server + network_image_ms + network_action_ms
                
                # Network throughput considering both local and cloud hops
                network_hz_local = compute_network_throughput_hz(
                    network_config=local_net,
                    image_config=image_config,
                    action_config=action_config,
                )
                network_hz_cloud = compute_network_throughput_hz(
                    network_config=cloud_net,
                    image_config=image_config,
                    action_config=action_config,
                )
                network_hz = min(network_hz_local, network_hz_cloud)
                
                # Async frequency: min(inference_hz, network_hz)
                freq_async_hz = min(inference_hz_server, network_hz)
                
                results.append({
                    "model": config.name,
                    "category": "Server Only",
                    "server_system": server_system,
                    "device_system": "N/A",
                    "network": network_name,
                    "precision": bits,
                    "vision_ms": vision_time_server,
                    "vlm_ms": vlm_time_server,
                    "action_ms": action_time_server,
                    "network_image_ms": network_image_ms,
                    "network_action_ms": network_action_ms,
                    "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": freq_async_hz,
                    "denoising_steps": denoising_steps,
                })
    except Exception as e:
        logger.warning(f"  {server_system} (Server Only): Error - {str(e)[:50]}")
    
    # ===== SCENARIO 3: Device-Server Collaboration =====
    try:
        # Precision is automatically selected based on system capabilities
        # Server GPU VLM performance
        df_vlm_server = get_pi0_vlm_perf(
            config, [server_system], [num_devices], bits, max_batch_size=1
        )
        
        if df_vlm_server.empty:
            logger.warning(f"  {server_system}: Skipped (VLM memory constraints)")
        else:
            vlm_time_server = df_vlm_server["time_ms"].values[0]
            
            # Device GPU action expert performance
            df_action_device = get_pi0_action_expert_perf(
                config, [device_system], [num_devices], bits, denoising_steps,
                vlm_sequence_length=config.vlm_sequence_length,
                max_batch_size=1
            )
            
            if df_action_device.empty:
                logger.warning(f"  {device_system}: Skipped (action expert memory constraints)")
            else:
                action_time_device = df_action_device["time_ms"].values[0]
                
                # Add network latency for each network config
                for net_config in network_configs:
                    # Image upload latency (Robot → Server)
                    image_latency = estimate_image_latency(net_config, image_config)
                    network_image_ms = image_latency["total_latency_ms"]
                    
                    # KV cache download latency (Server → Robot)
                    kv_cache_latency = estimate_kvcache_latency(net_config, kv_cache_config)
                    network_kv_cache_ms = kv_cache_latency["total_latency_ms"]
                    
                    # Total E2E latency = image upload + server VLM + KV cache download + device action
                    e2e_total = network_image_ms + vlm_time_server + network_kv_cache_ms + action_time_device
                    
                    # Compute network throughput Hz (for continuous streaming)
                    # Uses time-multiplexed network: upload image, then download KV cache
                    network_hz = compute_network_throughput_hz(
                        network_config=net_config,
                        image_config=image_config,
                        kvcache_config=kv_cache_config,
                    )
                    
                    # Compute frequency: min(server VLM Hz, network Hz, device action Hz)
                    server_vlm_hz = 1000.0 / vlm_time_server
                    device_action_hz = 1000.0 / action_time_device
                    freq_async_hz = min(server_vlm_hz, network_hz, device_action_hz)
                    
                    results.append({
                        "model": config.name,
                        "category": "Device-Server Collaboration",
                        "server_system": server_system,
                        "device_system": device_system,
                        "network": net_config.name,
                        "precision": bits,
                        "vision_ms": 0.0,  # Vision not used in collaboration
                        "vlm_ms": vlm_time_server,
                        "action_ms": action_time_device,
                        "network_image_ms": network_image_ms,
                        "network_action_ms": 0.0,  # No action download in collaboration
                        "network_kv_cache_ms": network_kv_cache_ms,
                        "e2e_total_ms": e2e_total,
                        "frequency_hz": 1000 / e2e_total,
                        "freq_async_hz": freq_async_hz,
                        "denoising_steps": denoising_steps,
                    })
                
                # Add cloud network pairs (two-hop: local + cloud)
                # Same pairs as in device_vs_server experiment
                cloud_network_pairs = [
                    ("Wired + Fast Cloud", ETHERNET_10G_CONFIG, CLOUD_FAST_CONFIG),
                    ("4G + Slow Cloud", CELL_4G_LTE_CONFIG, CLOUD_SLOW_CONFIG),
                ]
                
                for network_name, local_net, cloud_net in cloud_network_pairs:
                    # Local network latency (Robot → Gateway)
                    image_latency_local = estimate_image_latency(local_net, image_config)
                    kv_cache_latency_local = estimate_kvcache_latency(local_net, kv_cache_config)
                    
                    # Cloud network latency (Gateway → Server)
                    image_latency_cloud = estimate_image_latency(cloud_net, image_config)
                    kv_cache_latency_cloud = estimate_kvcache_latency(cloud_net, kv_cache_config)
                    
                    # Total network latency = local + cloud (bidirectional)
                    network_image_ms = image_latency_local["total_latency_ms"] + image_latency_cloud["total_latency_ms"]
                    network_kv_cache_ms = kv_cache_latency_local["total_latency_ms"] + kv_cache_latency_cloud["total_latency_ms"]
                    
                    # Total E2E latency = image upload + server VLM + KV cache download + device action
                    e2e_total = network_image_ms + vlm_time_server + network_kv_cache_ms + action_time_device
                    
                    # Network throughput considering both local and cloud hops
                    # For simplicity, we take the bottleneck (min of both)
                    network_hz_local = compute_network_throughput_hz(
                        network_config=local_net,
                        image_config=image_config,
                        kvcache_config=kv_cache_config,
                    )
                    network_hz_cloud = compute_network_throughput_hz(
                        network_config=cloud_net,
                        image_config=image_config,
                        kvcache_config=kv_cache_config,
                    )
                    network_hz = min(network_hz_local, network_hz_cloud)
                    
                    # Async frequency: min(inference_hz, network_hz)
                    server_vlm_hz = 1000.0 / vlm_time_server
                    device_action_hz = 1000.0 / action_time_device
                    freq_async_hz = min(server_vlm_hz, network_hz, device_action_hz)
                    
                    results.append({
                        "model": config.name,
                        "category": "Device-Server Collaboration",
                        "server_system": server_system,
                        "device_system": device_system,
                        "network": network_name,
                        "precision": bits,
                        "vision_ms": 0.0,  # Vision not used in collaboration
                        "vlm_ms": vlm_time_server,
                        "action_ms": action_time_device,
                        "network_image_ms": network_image_ms,
                        "network_action_ms": 0.0,  # No action download in collaboration
                        "network_kv_cache_ms": network_kv_cache_ms,
                        "e2e_total_ms": e2e_total,
                        "frequency_hz": 1000 / e2e_total,
                        "freq_async_hz": freq_async_hz,
                        "denoising_steps": denoising_steps,
                    })
    except Exception as e:
        logger.warning(f"  Device-Server Collaboration: Error - {str(e)[:50]}")
    
    return pd.DataFrame(results)


def print_device_server_collaboration_summary(df: pd.DataFrame, logger=None) -> None:
    """Print a formatted summary comparing device-server collaboration performance with baselines."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.info("No results to display.")
        return
    
    # Separate by category
    device_only_df = df[df["category"] == "Device Only"]
    server_only_df = df[df["category"] == "Server Only"]
    collaboration_df = df[df["category"] == "Device-Server Collaboration"]
    
    # Find best configurations for each category
    best_device_only = device_only_df["e2e_total_ms"].min() if not device_only_df.empty else None
    best_server_only = server_only_df["e2e_total_ms"].min() if not server_only_df.empty else None
    best_collaboration = collaboration_df["e2e_total_ms"].min() if not collaboration_df.empty else None
    
    logger.info("\n" + "-" * 150)
    logger.info("Summary:")
    
    # Collect all results for speed comparison
    results_for_comparison = []
    
    if best_device_only is not None:
        device_row = device_only_df.loc[device_only_df["e2e_total_ms"].idxmin()]
        device_async = device_row.get('freq_async_hz', device_row['frequency_hz'])
        device_sync = 1000/best_device_only
        logger.info(f"  Device Only:        {device_row['device_system']} @ {best_device_only:.2f} ms ({device_sync:.1f} Hz sync, {device_async:.1f} Hz async)")
        results_for_comparison.append(("Device Only", device_sync, device_async))
    
    if best_server_only is not None:
        server_row = server_only_df.loc[server_only_df["e2e_total_ms"].idxmin()]
        server_async = server_row.get('freq_async_hz', server_row['frequency_hz'])
        server_sync = 1000/best_server_only
        logger.info(f"  Server Only:        {server_row['server_system']} + {server_row['network']} @ {best_server_only:.2f} ms ({server_sync:.1f} Hz sync, {server_async:.1f} Hz async)")
        results_for_comparison.append(("Server Only", server_sync, server_async))
    
    if best_collaboration is not None:
        collab_row = collaboration_df.loc[collaboration_df["e2e_total_ms"].idxmin()]
        collab_async = collab_row.get('freq_async_hz', collab_row['frequency_hz'])
        collab_sync = 1000/best_collaboration
        logger.info(f"  Device-Server Collab: {collab_row['server_system']} (server) + {collab_row['device_system']} (device) + {collab_row['network']} @ {best_collaboration:.2f} ms ({collab_sync:.1f} Hz sync, {collab_async:.1f} Hz async)")
        results_for_comparison.append(("Device-Server Collab", collab_sync, collab_async))
    
    # Print speed comparison (sorted by async frequency, descending)
    if len(results_for_comparison) > 1:
        # Sort by async frequency (descending)
        results_for_comparison_sorted_async = sorted(results_for_comparison, key=lambda x: x[2], reverse=True)
        
        # Build async comparison string
        async_comparison_parts = []
        for name, sync_hz, async_hz in results_for_comparison_sorted_async:
            async_comparison_parts.append(f"{name} ({async_hz:.1f} Hz)")
        
        logger.info(f"  Speed Comparison (Async): {' > '.join(async_comparison_parts)}")
        
        # Sort by sync frequency (descending)
        results_for_comparison_sorted_sync = sorted(results_for_comparison, key=lambda x: x[1], reverse=True)
        
        # Build sync comparison string
        sync_comparison_parts = []
        for name, sync_hz, async_hz in results_for_comparison_sorted_sync:
            sync_comparison_parts.append(f"{name} ({sync_hz:.1f} Hz)")
        
        logger.info(f"  Speed Comparison (Sync):  {' > '.join(sync_comparison_parts)}")
    
    # Print Device Only results
    if not device_only_df.empty:
        logger.info("\n" + "-" * 130)
        logger.info("Device Only (Local - No Network Latency)")
        logger.info("-" * 130)
        logger.info(f"{'System':<25} {'Prec':<6} {'Vision':>10} {'VLM':>10} {'Action':>10} {'Total':>10} {'Freq (Sync)':>12} {'Freq (Async)':>12}")
        logger.info(f"{'':25} {'':6} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(Hz)':>12} {'(Hz)':>12}")
        logger.info("-" * 130)
        
        for _, row in device_only_df.iterrows():
            freq_async = row.get('freq_async_hz', row['frequency_hz'])
            logger.info(f"{row['device_system']:<25} "
                  f"{row['precision']:<6} "
                  f"{row['vision_ms']:>10.2f} "
                  f"{row['vlm_ms']:>10.2f} "
                  f"{row['action_ms']:>10.2f} "
                  f"{row['e2e_total_ms']:>10.2f} "
                  f"{row['frequency_hz']:>12.1f} "
                  f"{freq_async:>12.1f}")
    
    # Print Server Only results
    if not server_only_df.empty:
        logger.info("\n" + "-" * 150)
        logger.info("Server Only (Remote - Image Upload + Full Inference + Action Download)")
        logger.info("-" * 150)
        logger.info(f"{'Network':<30} {'Prec':<6} {'Vision':>10} {'VLM':>10} {'Action':>10} {'Img Net':>10} {'Act Net':>10} {'Total':>10} {'Freq (Sync)':>12} {'Freq (Async)':>12}")
        logger.info(f"{'':30} {'':6} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(Hz)':>12} {'(Hz)':>12}")
        logger.info("-" * 150)
        
        for _, row in server_only_df.sort_values("e2e_total_ms").iterrows():
            freq_async = row.get('freq_async_hz', row['frequency_hz'])
            logger.info(f"{row['network']:<30} "
                  f"{row['precision']:<6} "
                  f"{row['vision_ms']:>10.2f} "
                  f"{row['vlm_ms']:>10.2f} "
                  f"{row['action_ms']:>10.2f} "
                  f"{row['network_image_ms']:>10.2f} "
                  f"{row['network_action_ms']:>10.2f} "
                  f"{row['e2e_total_ms']:>10.2f} "
                  f"{row['frequency_hz']:>12.1f} "
                  f"{freq_async:>12.1f}")
    
    # Print Device-Server Collaboration results
    if not collaboration_df.empty:
        logger.info("\n" + "-" * 150)
        logger.info("Device-Server Collaboration (Image Upload → Server VLM → KV Cache Download → Device Action)")
        logger.info("-" * 150)
        logger.info(f"{'Network':<30} {'Prec':<10} {'Img Up':>10} {'Server VLM':>10} {'KV Down':>10} {'Device Act':>10} {'Total':>10} {'Freq (Sync)':>12} {'Freq (Async)':>12}")
        logger.info(f"{'':30} {'':10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(Hz)':>12} {'(Hz)':>12}")
        logger.info("-" * 150)
        
        for _, row in collaboration_df.sort_values("e2e_total_ms").iterrows():
            freq_async = row.get('freq_async_hz', row['frequency_hz'])
            logger.info(f"{row['network']:<30} "
                  f"{row['precision']:<10} "
                  f"{row['network_image_ms']:>10.2f} "
                  f"{row['vlm_ms']:>10.2f} "
                  f"{row['network_kv_cache_ms']:>10.2f} "
                  f"{row['action_ms']:>10.2f} "
                  f"{row['e2e_total_ms']:>10.2f} "
                  f"{row['frequency_hz']:>12.1f} "
                  f"{freq_async:>12.1f}")


def run_device_server_collaboration_comparison(
    output_dir: str = "perf_results",
    network_configs: list[NetworkConfig] = None,
    server_system: str = "B100",
    device_system: str = "Jetson_AGX_Thor",
    experiment_num: int = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Run device-server collaboration comparison for all Pi0 family models.
    
    Args:
        output_dir: Directory to save results
        network_configs: List of network configurations to test (default: all)
        server_system: Server GPU system (default: B100)
        device_system: Device GPU system (default: Jetson_AGX_Thor)
        experiment_num: Experiment number for logging (optional)
        logger: Logger instance (optional)
    
    Returns:
        Dictionary mapping model names to comparison DataFrames
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if network_configs is None:
        network_configs = ALL_NETWORK_CONFIGS
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    all_dfs = []
    
    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 150)
    logger.info(f"{exp_header}DEVICE-SERVER COLLABORATION COMPARISON")
    logger.info("=" * 150)
    logger.info(f"Server GPU: {server_system}, Device GPU: {device_system}")
    logger.info(f"Testing {len(network_configs)} network configurations")
    logger.info(f"Expected conclusion: Device / Server-only solution can be faster or slower to each other, "
                 "but Device-Server Collaboration always be slower than Server Only, "
                 "because KV cache download is always slower than action download.")
    
    for config in ALL_PI0_CONFIGS:
        df = compare_device_server_collaboration(
            config=config,
            denoising_steps=10,
            bits="bf16",
            server_system=server_system,
            device_system=device_system,
            network_configs=network_configs,
            logger=logger,
        )
        
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        
        logger.info(f"\n{config.name}:")
        print_device_server_collaboration_summary(df, logger=logger)
    
    # Save combined results
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        output_file = output_path / "pi0_device_server_collaboration.csv"
        df_combined.to_csv(output_file, index=False)
        logger.info(f"\n\nResults saved to {output_file}")
    
    return all_results





if __name__ == "__main__":
    # Set up logging
    logger = setup_logging("perf_results/pi0_perf.log")
    
    # Default configuration
    system_list_all = ["A100_80GB", "H100", "B100", "RTX_4090", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"
    denoising_steps = 10  # Default flow matching steps
    
    # ============================================
    # EXPERIMENT CONTROL FLAGS
    # ============================================
    # Set runall to True to run all experiments, or False to selectively run experiments
    runall = True
    
    # Individual experiment flags (only used if runall=False)
    run_exp_1_base_pi0 = True                # Base Pi0 family performance
    run_exp_2_model_size_scaling = True      # Model size scaling
    run_exp_3_long_context = True            # Long context experiment
    run_exp_4_denoise_steps_action_lengths = True   # Denoising steps & action lengths
    run_exp_5_autoregressive_vs_diffusion = True    # Autoregressive vs Diffusion comparison
    run_exp_6_device_vs_server = True        # On-device vs Edge-server vs Cloud inference
    run_exp_7_device_server_collaboration = True    # Device-server collaboration
    # ============================================
    
    # Log experiment configuration
    if runall:
        logger.info("=" * 100)
        logger.info("RUNNING ALL EXPERIMENTS")
        logger.info("=" * 100)
    else:
        logger.info("=" * 100)
        logger.info("SELECTIVE EXPERIMENT EXECUTION")
        logger.info("-" * 100)
        logger.info(f"Exp 1 - Base Pi0 Family:              {'ENABLED' if run_exp_1_base_pi0 else 'DISABLED'}")
        logger.info(f"Exp 2 - Model Size Scaling:           {'ENABLED' if run_exp_2_model_size_scaling else 'DISABLED'}")
        logger.info(f"Exp 3 - Long Context:                 {'ENABLED' if run_exp_3_long_context else 'DISABLED'}")
        logger.info(f"Exp 4 - Denoise Steps & Action Lens:  {'ENABLED' if run_exp_4_denoise_steps_action_lengths else 'DISABLED'}")
        logger.info(f"Exp 5 - Autoregressive vs Diffusion:  {'ENABLED' if run_exp_5_autoregressive_vs_diffusion else 'DISABLED'}")
        logger.info(f"Exp 6 - Device vs Server:             {'ENABLED' if run_exp_6_device_vs_server else 'DISABLED'}")
        logger.info(f"Exp 7 - Device-Server Collaboration:  {'ENABLED' if run_exp_7_device_server_collaboration else 'DISABLED'}")
        logger.info("=" * 100)
    
    # Experiment counter
    exp_counter = 0
    
    # Experiment 1: Run evaluation for all Pi0 family models
    if runall or run_exp_1_base_pi0:
        logger.info("Starting Pi0 family performance evaluation...")
        exp_counter += 1
        all_results = get_all_pi0_perf(
            system_list=system_list_all,
            num_device_list=num_device_list,
            bits=bits,
            denoising_steps=denoising_steps,
            experiment_num=exp_counter,
            logger=logger,
        )
        
        # Print summary
        print_all_pi0_perf_summary(all_results, logger=logger)
    
    # Experiment 2: Evaluate model size scaling
    if runall or run_exp_2_model_size_scaling:
        logger.info("\n\n")
        exp_counter += 1
        size_scaling_results, size_scaling_configs = get_model_size_scaling_perf(
            system_list=system_list_all,
            num_device_list=[1],  # Single chip for fair comparison
            bits=bits,
            denoising_steps=denoising_steps,
            experiment_num=exp_counter,
            logger=logger,
        )
        print_model_size_scaling_summary(size_scaling_results, size_scaling_configs, logger=logger)
   
    # Experiment 3: Run long context experiment
    if runall or run_exp_3_long_context:
        logger.info("\n\n")
        exp_counter += 1
        run_long_context_experiment(
            experiment_num=exp_counter,
            logger=logger
        )
 
    # Experiment 4: Compare denoising steps for all models
    if runall or run_exp_4_denoise_steps_action_lengths:
        logger.info("\n\n")
        exp_counter += 1
        df_steps = compare_denoising_steps_action_lengths(
            experiment_num=exp_counter,
            logger=logger
        )
        print_denoising_steps_action_lengths_summarys(df_steps, logger=logger)
    
    # Experiment 5: Compare autoregressive vs diffusion action generators
    if runall or run_exp_5_autoregressive_vs_diffusion:
        logger.info("\n\n")
        exp_counter += 1
        df_auto_vs_diff = compare_autoregressive_vs_diffusion(
            experiment_num=exp_counter,
            logger=logger
        )
        print_autoregressive_vs_diffusion_summary(df_auto_vs_diff, logger=logger)
    
    # Experiment 6: Run device vs server comparison (On-device, Edge-server, Cloud)
    if runall or run_exp_6_device_vs_server:
        logger.info("\n\n")
        exp_counter += 1
        device_vs_server_results = run_device_vs_server_comparison(
            experiment_num=exp_counter,
            logger=logger
        )
    
    # Experiment 7: Run device-server collaboration experiment
    if runall or run_exp_7_device_server_collaboration:
        logger.info("\n\n")
        exp_counter += 1
        device_server_results = run_device_server_collaboration_comparison(
            server_system="B100",
            device_system="Jetson_AGX_Thor",
            experiment_num=exp_counter,
            logger=logger,
        )
    
    # Final summary
    logger.info("\n" + "=" * 100)
    logger.info(f"COMPLETED {exp_counter} EXPERIMENT(S)")
    logger.info("=" * 100)
    
