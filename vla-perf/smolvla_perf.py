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
SmolVLA Performance Evaluation Script

SmolVLA (~450M total) from Hugging Face:
- Vision:  SigLIP-SO400M/14 — processes 256 patches (224px / 14px patch = 16x16).
           Layer-skipping extracts features from an intermediate layer, so the ViT
           still runs all its compute but outputs only 64 tokens to the VLM.
- VLM:     SmolLM2-1.7B (24 layers, 2048 hidden) — processes 64 visual + language tokens
- Action:  Flow Matching Transformer (~100M, 10 layers, 768 hidden)
           Cross-attends to VLM hidden states; N denoising steps.

Performance modeling breakdown:
1. Vision encoding  (SigLIP prefill over 256 ViT patches)
2. VLM prefill      (SmolLM2-1.7B over 64 visual tokens + language tokens)
3. Action Expert    (flow matching parallel decode, N denoising steps)

Experiments (7 total, mirroring pi0_perf.py depth):
  1. Base E2E performance across hardware
  2. Model size scaling (SmolLM2-1.7B → 7B → 13B → 70B backbone variants)
  3. Long context (multi-camera: 1-5 SigLIP frames)
  4. Denoising steps x action chunk size sweep
  5. Autoregressive vs Diffusion action generation comparison
  6. Device vs Server (on-device / edge-server / cloud + network latency)
  7. Device-Server Collaboration (VLM on server, action expert on device)

Reference:
    arxiv.org/abs/2506.01844
    huggingface.co/blog/smolvla
    genz/GenZ/Models/Model_sets/vla_models.py -> SmolVLA model configs
"""

import pandas as pd
import logging
from pathlib import Path
from dataclasses import dataclass
import copy

from GenZ.Models.default_models import ModelConfig, MODEL_DICT

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
# SmolVLA Architecture Constants
# ==============================================================================
SMOLVLA_VISION_ENCODING_TOKENS = 256   # SigLIP ViT input: 224px / 14px patch = 16x16 = 256 patches
SMOLVLA_VLM_VISION_TOKENS = 64         # tokens fed to the VLM after layer-skipping extraction
SMOLVLA_LANGUAGE_TOKENS = 32           # typical instruction token count
SMOLVLA_TOTAL_VLM_TOKENS = SMOLVLA_VLM_VISION_TOKENS + SMOLVLA_LANGUAGE_TOKENS
SMOLVLA_ACTION_CHUNK = 16              # action tokens per flow matching pass
SMOLVLA_DENOISING_STEPS = 10          # default denoising iterations
SMOLVLA_ACTION_DOF = 7                 # 7-DoF action space


@dataclass
class SmolVLAConfig:
    """Configuration for a SmolVLA-family model variant."""
    name: str
    vision_model: str
    vlm_model: str
    action_model: str
    vision_tokens_per_frame: int = SMOLVLA_VISION_ENCODING_TOKENS
    vlm_vision_tokens_per_frame: int = SMOLVLA_VLM_VISION_TOKENS
    language_tokens: int = SMOLVLA_LANGUAGE_TOKENS
    num_frames: int = 1
    action_chunk: int = SMOLVLA_ACTION_CHUNK
    denoising_steps: int = SMOLVLA_DENOISING_STEPS
    action_dof: int = SMOLVLA_ACTION_DOF

    @property
    def total_vision_tokens(self) -> int:
        return self.vision_tokens_per_frame * self.num_frames

    @property
    def total_vlm_tokens(self) -> int:
        return self.vlm_vision_tokens_per_frame * self.num_frames + self.language_tokens


SMOLVLA_CONFIG = SmolVLAConfig(
    name="smolvla",
    vision_model="smolvla-vision",
    vlm_model="smollm2-1.7b",
    action_model="smolvla-action-expert",
)

ALL_SMOLVLA_CONFIGS = [SMOLVLA_CONFIG]


def create_action_expert_config_from_vlm(vlm_config: ModelConfig, name: str) -> ModelConfig:
    """Derive an action expert config from a VLM config (hidden//2, ffn//4)."""
    action_config = copy.deepcopy(vlm_config)
    action_config.model = name
    action_config.hidden_size = vlm_config.hidden_size // 2
    action_config.intermediate_size = vlm_config.intermediate_size // 4
    action_config.vocab_size = 0
    return action_config


# ==============================================================================
# Component perf functions
# ==============================================================================

def get_smolvla_vision_perf(
    config: SmolVLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """Evaluate SmolVLA SigLIP vision encoding."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model=config.vision_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vision_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_smolvla_vlm_perf(
    config: SmolVLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """Evaluate SmolLM2-1.7B VLM prefill for SmolVLA."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model=config.vlm_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_smolvla_action_perf(
    config: SmolVLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    denoising_steps: int = None,
    action_chunk: int = None,
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate SmolVLA action expert (flow matching transformer).

    Total action latency = denoising_steps * single-step parallel-decode latency.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if denoising_steps is None:
        denoising_steps = config.denoising_steps
    if action_chunk is None:
        action_chunk = config.action_chunk

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_parallel_decode_perf(
                model=config.action_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens_parallel=action_chunk,
                self_attention=True,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                for r in model_results:
                    r["time_ms"] *= denoising_steps
                    r["model.dec_steps"] = denoising_steps
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        return df
    return get_optimal_df(df, apply_pareto=True)


# ==============================================================================
# Experiment 1: Base E2E
# ==============================================================================

def get_smolvla_e2e_perf(
    config: SmolVLAConfig = SMOLVLA_CONFIG,
    system_list: list[str] = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Evaluate end-to-end SmolVLA performance across all components."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    df_vision = get_smolvla_vision_perf(config, system_list, num_device_list, bits, logger=logger)
    results["vision"] = df_vision
    df_vision.to_csv(output_path / f"{config.name}_vision_perf.csv", index=False)

    df_vlm = get_smolvla_vlm_perf(config, system_list, num_device_list, bits, logger=logger)
    results["vlm"] = df_vlm
    df_vlm.to_csv(output_path / f"{config.name}_vlm_perf.csv", index=False)

    df_action = get_smolvla_action_perf(config, system_list, num_device_list, bits, logger=logger)
    results["action"] = df_action
    df_action.to_csv(output_path / f"{config.name}_action_perf.csv", index=False)

    if df_vision.empty or df_vlm.empty or df_action.empty:
        logger.warning(f"One or more component DataFrames empty for {config.name} — skipping E2E.")
        results["e2e"] = pd.DataFrame()
        return results

    group_cols = ["hardware.name", "hardware.num_chips", "batch_size"]
    vision_times = df_vision[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "vision_time_ms"})
    vlm_times = df_vlm[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "vlm_time_ms"})
    action_times = df_action[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "action_time_ms"})

    df_merged = vision_times.merge(vlm_times, on=group_cols, how="inner")
    df_merged = df_merged.merge(action_times, on=group_cols, how="inner")
    df_merged["e2e_time_ms"] = df_merged["vision_time_ms"] + df_merged["vlm_time_ms"] + df_merged["action_time_ms"]
    df_merged["model.name"] = config.name
    df_merged["model.stage"] = "e2e"
    df_merged["model.dec_steps"] = config.denoising_steps
    df_merged["model.seq_len_inference_prefill"] = config.total_vlm_tokens

    df_e2e = df_merged[[
        "model.name", "model.stage", "model.dec_steps",
        "model.seq_len_inference_prefill",
        "hardware.name", "hardware.num_chips", "batch_size",
        "vision_time_ms", "vlm_time_ms", "action_time_ms", "e2e_time_ms",
    ]]
    results["e2e"] = df_e2e
    df_e2e.to_csv(output_path / f"{config.name}_e2e_perf.csv", index=False)

    return results


def get_all_smolvla_perf(
    system_list: list[str] = ["A100_80GB", "H100", "B100", "RTX_3090", "RTX_4090", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    denoising_steps: int = SMOLVLA_DENOISING_STEPS,
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run Exp 1: base E2E for all SmolVLA configs."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}SMOLVLA BASE PERFORMANCE EVALUATION")
    logger.info("=" * 130)

    all_results = {}
    for config in ALL_SMOLVLA_CONFIGS:
        cfg = SmolVLAConfig(
            name=config.name,
            vision_model=config.vision_model,
            vlm_model=config.vlm_model,
            action_model=config.action_model,
            denoising_steps=denoising_steps,
        )
        logger.info(f"\nEvaluating {cfg.name}...")
        results = get_smolvla_e2e_perf(cfg, system_list, num_device_list, bits, output_dir, logger)
        all_results[config.name] = results

    all_e2e = [r["e2e"] for r in all_results.values() if not r["e2e"].empty]
    if all_e2e:
        pd.concat(all_e2e, ignore_index=True).to_csv(
            Path(output_dir) / "smolvla_family_e2e_perf.csv", index=False
        )

    return all_results


# ==============================================================================
# Experiment 2: Model Size Scaling
# ==============================================================================

def get_model_size_scaling_perf(
    system_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    denoising_steps: int = SMOLVLA_DENOISING_STEPS,
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> tuple[dict, dict]:
    """
    Exp 2: Evaluate SmolVLA performance scaling with different VLM backbone sizes.

    Variants:
      smolvla:    smolvla-vision + smollm2-1.7b  + action expert (actual)
      smolvla-7b: smolvla-vision + llama2_7b     + derived action (hidden//2)
      smolvla-13b:smolvla-vision + llama2_13b    + derived action (hidden//2)
      smolvla-70b:smolvla-vision + llama2_70b    + derived action (hidden//2)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = [1]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}SMOLVLA MODEL SIZE SCALING EVALUATION")
    logger.info("=" * 130)

    model_variants = []

    # Baseline: actual SmolVLA (smollm2-1.7b)
    vlm_s = copy.deepcopy(MODEL_DICT.get_model("smollm2-1.7b"))
    vlm_s.vocab_size = 0
    vlm_s.model = "smollm2-1.7b-no-vocab"
    action_s = copy.deepcopy(MODEL_DICT.get_model("smolvla-action-expert"))
    action_s.model = "vla/smolvla-size-scaling-action-1.7b"
    MODEL_DICT.add_model(vlm_s)
    MODEL_DICT.add_model(action_s)
    model_variants.append(SmolVLAConfig(
        name="smolvla-1.7b", vision_model="smolvla-vision",
        vlm_model=vlm_s.model, action_model=action_s.model,
        denoising_steps=denoising_steps,
    ))

    # 7B variant
    vlm_7b = copy.deepcopy(MODEL_DICT.get_model("llama2_7b"))
    vlm_7b.vocab_size = 0
    vlm_7b.model = "llama2_7b-no-vocab"
    action_7b = create_action_expert_config_from_vlm(vlm_7b, "vla/smolvla-size-scaling-action-7b")
    MODEL_DICT.add_model(vlm_7b)
    MODEL_DICT.add_model(action_7b)
    model_variants.append(SmolVLAConfig(
        name="smolvla-7b", vision_model="smolvla-vision",
        vlm_model=vlm_7b.model, action_model=action_7b.model,
        denoising_steps=denoising_steps,
    ))

    # 13B variant
    vlm_13b = copy.deepcopy(MODEL_DICT.get_model("llama2_13b"))
    vlm_13b.vocab_size = 0
    vlm_13b.model = "llama2_13b-no-vocab"
    action_13b = create_action_expert_config_from_vlm(vlm_13b, "vla/smolvla-size-scaling-action-13b")
    MODEL_DICT.add_model(vlm_13b)
    MODEL_DICT.add_model(action_13b)
    model_variants.append(SmolVLAConfig(
        name="smolvla-13b", vision_model="smolvla-vision",
        vlm_model=vlm_13b.model, action_model=action_13b.model,
        denoising_steps=denoising_steps,
    ))

    # 70B variant
    vlm_70b = copy.deepcopy(MODEL_DICT.get_model("llama2_70b"))
    vlm_70b.vocab_size = 0
    vlm_70b.model = "llama2_70b-no-vocab"
    action_70b = create_action_expert_config_from_vlm(vlm_70b, "vla/smolvla-size-scaling-action-70b")
    MODEL_DICT.add_model(vlm_70b)
    MODEL_DICT.add_model(action_70b)
    model_variants.append(SmolVLAConfig(
        name="smolvla-70b", vision_model="smolvla-vision",
        vlm_model=vlm_70b.model, action_model=action_70b.model,
        denoising_steps=denoising_steps,
    ))

    # Log parameter counts
    logger.info("\n--- Model Parameter Counts ---")
    component_params_data = []
    for cfg in model_variants:
        vision_cfg = MODEL_DICT.get_model(cfg.vision_model)
        vlm_cfg = MODEL_DICT.get_model(cfg.vlm_model)
        act_cfg = MODEL_DICT.get_model(cfg.action_model)
        vp = calculate_transformer_params(vision_cfg)
        lp = calculate_transformer_params(vlm_cfg)
        ap = calculate_transformer_params(act_cfg)
        total = vp + lp + ap
        logger.info(f"  {cfg.name}: vision={format_param_count(vp)}, vlm={format_param_count(lp)}, "
                    f"action={format_param_count(ap)}, total={format_param_count(total)}")
        component_params_data.append({
            "model": cfg.name,
            "vision_params_M": vp / 1e6,
            "vlm_params_M": lp / 1e6,
            "action_params_M": ap / 1e6,
            "total_params_M": total / 1e6,
        })
    pd.DataFrame(component_params_data).to_csv(output_path / "smolvla_model_size_scaling_params.csv", index=False)

    # Run evaluations
    all_results = {}
    all_e2e = []
    for cfg in model_variants:
        logger.info(f"\n  Evaluating {cfg.name}...")
        results = get_smolvla_e2e_perf(cfg, system_list, num_device_list, bits, output_dir, logger)
        all_results[cfg.name] = results
        if not results["e2e"].empty:
            all_e2e.append(results["e2e"])

    if all_e2e:
        pd.concat(all_e2e, ignore_index=True).to_csv(
            output_path / "smolvla_model_size_scaling.csv", index=False
        )

    return all_results, {cfg.name: cfg for cfg in model_variants}


def print_model_size_scaling_summary(all_results: dict, model_configs: dict, logger=None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("\n--- SmolVLA Model Size Scaling Summary (Batch=1, Chips=1) ---")
    logger.info(f"{'Model':<18} {'Hardware':<20} {'Vision (ms)':>12} {'VLM (ms)':>12} {'Action (ms)':>12} {'E2E (ms)':>12} {'Hz':>8}")
    logger.info("-" * 100)
    for model_name, results in all_results.items():
        if "e2e" not in results or results["e2e"].empty:
            continue
        df = results["e2e"]
        sub = df[(df["hardware.num_chips"] == 1) & (df["batch_size"] == 1)]
        for _, row in sub.iterrows():
            e2e = row["e2e_time_ms"]
            logger.info(f"{model_name:<18} {row['hardware.name']:<20} "
                        f"{row['vision_time_ms']:>12.2f} {row['vlm_time_ms']:>12.2f} "
                        f"{row['action_time_ms']:>12.2f} {e2e:>12.2f} {1000/e2e:>8.1f}")


# ==============================================================================
# Experiment 3: Long Context (multi-camera / multi-frame)
# ==============================================================================

def run_long_context_experiment(
    config: SmolVLAConfig = SMOLVLA_CONFIG,
    system_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_frames_list: list[int] = [1, 2, 3, 4, 5],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 3: Evaluate SmolVLA latency as number of camera frames scales from 1 to 5.

    Each additional frame adds SMOLVLA_VISION_ENCODING_TOKENS to the SigLIP prefill
    and SMOLVLA_VLM_VISION_TOKENS to the VLM context.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}SMOLVLA LONG CONTEXT (MULTI-CAMERA) EXPERIMENT")
    logger.info("=" * 130)
    logger.info(f"Systems: {system_list}")
    logger.info(f"Frames: {num_frames_list}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = []
    for num_frames in num_frames_list:
        cfg = SmolVLAConfig(
            name=config.name,
            vision_model=config.vision_model,
            vlm_model=config.vlm_model,
            action_model=config.action_model,
            num_frames=num_frames,
            denoising_steps=config.denoising_steps,
        )
        total_vision_tokens = cfg.total_vision_tokens
        total_vlm_tokens = cfg.total_vlm_tokens

        kv_cache_mb = calculate_kv_cache_size_mb(
            model_name=config.vlm_model,
            seq_length=total_vlm_tokens,
            bits=bits,
        )
        logger.info(f"\n  Frames={num_frames}: vision_tokens={total_vision_tokens}, "
                    f"vlm_tokens={total_vlm_tokens}, KV cache={kv_cache_mb:.1f} MB")

        for system in system_list:
            df_vision = get_smolvla_vision_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)
            df_vlm = get_smolvla_vlm_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)
            df_action = get_smolvla_action_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)

            if df_vision.empty or df_vlm.empty or df_action.empty:
                continue

            vision_ms = df_vision["time_ms"].values[0]
            vlm_ms = df_vlm["time_ms"].values[0]
            action_ms = df_action["time_ms"].values[0]
            e2e_ms = vision_ms + vlm_ms + action_ms

            results.append({
                "model": config.name,
                "system": system,
                "num_frames": num_frames,
                "total_vision_tokens": total_vision_tokens,
                "total_vlm_tokens": total_vlm_tokens,
                "kv_cache_mb": kv_cache_mb,
                "vision_ms": vision_ms,
                "vlm_ms": vlm_ms,
                "action_ms": action_ms,
                "e2e_ms": e2e_ms,
                "frequency_hz": 1000 / e2e_ms,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        out = output_path / "smolvla_long_context.csv"
        df.to_csv(out, index=False)
        logger.info(f"\nLong context results saved to {out}")

        logger.info("\n--- Long Context Summary ---")
        logger.info(f"{'Frames':>7} {'Vision Tok':>11} {'VLM Tok':>9} {'KV MB':>8} "
                    f"{'System':<20} {'Vision':>10} {'VLM':>10} {'Action':>10} {'E2E':>10} {'Hz':>8}")
        logger.info("-" * 110)
        for _, row in df.iterrows():
            logger.info(f"{int(row['num_frames']):>7} {int(row['total_vision_tokens']):>11} "
                        f"{int(row['total_vlm_tokens']):>9} {row['kv_cache_mb']:>8.1f} "
                        f"{row['system']:<20} {row['vision_ms']:>10.2f} {row['vlm_ms']:>10.2f} "
                        f"{row['action_ms']:>10.2f} {row['e2e_ms']:>10.2f} {row['frequency_hz']:>8.1f}")

    return df


# ==============================================================================
# Experiment 4: Denoising Steps x Action Chunk Size sweep
# ==============================================================================

def compare_denoising_steps_action_lengths(
    config: SmolVLAConfig = SMOLVLA_CONFIG,
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    step_range: list[int] = [1, 5, 10, 20, 50],
    chunk_range: list[int] = [4, 8, 16, 32, 64],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 4: 2D sweep over denoising steps and action chunk sizes.
    Shows latency grid and speedup vs. baseline (steps=10, chunk=16).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}SMOLVLA DENOISING STEPS x ACTION CHUNK SIZE")
    logger.info("=" * 130)
    logger.info(f"Systems: {systems}, Steps: {step_range}, Chunks: {chunk_range}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = []
    for system in systems:
        logger.info(f"\n  System: {system}")
        df_vision = get_smolvla_vision_perf(config, [system], [1], bits, max_batch_size=1, logger=logger)
        df_vlm = get_smolvla_vlm_perf(config, [system], [1], bits, max_batch_size=1, logger=logger)

        if df_vision.empty or df_vlm.empty:
            logger.warning(f"    Skipped {system} (no vision/VLM data)")
            continue

        vision_ms = df_vision[df_vision["batch_size"] == 1]["time_ms"].values[0]
        vlm_ms = df_vlm[df_vlm["batch_size"] == 1]["time_ms"].values[0]

        for steps in step_range:
            for chunk in chunk_range:
                df_action = get_smolvla_action_perf(
                    config, [system], [1], bits,
                    denoising_steps=steps, action_chunk=chunk,
                    max_batch_size=1, logger=logger,
                )
                if df_action.empty:
                    continue
                action_ms = df_action[df_action["batch_size"] == 1]["time_ms"].values[0]
                e2e_ms = vision_ms + vlm_ms + action_ms
                results.append({
                    "model": config.name,
                    "system": system,
                    "denoising_steps": steps,
                    "action_chunk": chunk,
                    "vision_ms": vision_ms,
                    "vlm_ms": vlm_ms,
                    "action_ms": action_ms,
                    "e2e_ms": e2e_ms,
                    "frequency_hz": 1000 / e2e_ms,
                })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    out = output_path / "smolvla_denoising_steps_action_chunks.csv"
    df.to_csv(out, index=False)
    logger.info(f"\nDenoising steps x action chunk results saved to {out}")

    # Print 2D latency grid per system
    default_steps = config.denoising_steps
    default_chunk = config.action_chunk
    for system in df["system"].unique():
        sys_df = df[df["system"] == system]
        logger.info(f"\n  System: {system} — E2E latency (ms) grid")
        pivot = sys_df.pivot(index="denoising_steps", columns="action_chunk", values="e2e_ms")
        col_label = "Steps \\ Chunk"
        header = f"{col_label:<14}" + "".join(f"{c:>10}" for c in chunk_range)
        logger.info(header)
        logger.info("-" * (14 + 10 * len(chunk_range)))
        for steps in step_range:
            if steps in pivot.index:
                row_str = f"{steps:<14}" + "".join(
                    f"{pivot.loc[steps, c]:>10.2f}" if c in pivot.columns else f"{'N/A':>10}"
                    for c in chunk_range
                )
                logger.info(row_str)

        # Speedup grid vs baseline
        baseline_row = sys_df[(sys_df["denoising_steps"] == default_steps) & (sys_df["action_chunk"] == default_chunk)]
        if not baseline_row.empty:
            baseline_ms = baseline_row["e2e_ms"].values[0]
            pivot_speedup = baseline_ms / pivot
            logger.info(f"\n  Speedup vs baseline (steps={default_steps}, chunk={default_chunk})")
            logger.info(header)
            logger.info("-" * (14 + 10 * len(chunk_range)))
            for steps in step_range:
                if steps in pivot_speedup.index:
                    row_str = f"{steps:<14}" + "".join(
                        f"{pivot_speedup.loc[steps, c]:>10.2f}x" if c in pivot_speedup.columns else f"{'N/A':>10}"
                        for c in chunk_range
                    )
                    logger.info(row_str)

    return df


# ==============================================================================
# Experiment 5: Autoregressive vs Diffusion
# ==============================================================================

def compare_autoregressive_vs_diffusion(
    config: SmolVLAConfig = SMOLVLA_CONFIG,
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_devices: int = 1,
    bits: str = "bf16",
    denoising_steps_range: list[int] = [10],
    action_chunk_sizes: list[int] = [1, 4, 8, 16, 32],
    dof_values: list[int] = [7, 14, 21, 28, 35, 42],
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 5: Compare four action generation strategies for SmolVLA.

    A) Autoregressive (SmolLM2-1.7B backbone as AR action predictor)
    B) Small Diffusion (SmolVLA action expert ~100M flow matching)
    C) Large Diffusion (VLM-sized DiT ~1.7B)
    D) Autoregressive with Parallel Decoding (all action tokens at once)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}SMOLVLA AUTOREGRESSIVE VS DIFFUSION")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    action_predictor_model = "smolvla-vlm-action-predictor"
    vlm_ap_config = copy.deepcopy(MODEL_DICT.get_model(config.vlm_model))
    vlm_ap_config.model = action_predictor_model
    vlm_ap_config.vocab_size = 0
    MODEL_DICT.add_model(vlm_ap_config)

    results = []

    for system in systems:
        logger.info(f"\n  System: {system}")

        df_vision = get_smolvla_vision_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_vlm = get_smolvla_vlm_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)

        if df_vision.empty or df_vlm.empty:
            logger.warning(f"    Skipped {system}")
            continue

        vision_ms = df_vision[df_vision["batch_size"] == 1]["time_ms"].values[0]
        vlm_ms = df_vlm[df_vlm["batch_size"] == 1]["time_ms"].values[0]

        # Setup A: Autoregressive
        logger.info("    Setup A: Autoregressive VLA")
        for chunk_size in action_chunk_sizes:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens=config.action_dof * chunk_size,
                bits=bits,
                max_batch_size=1,
            )
            if decode_results:
                row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_ms = row["time_ms"] * config.action_dof * chunk_size
                e2e_ms = vision_ms + vlm_ms + action_ms
                results.append({
                    "system": system, "setup": "A: Autoregressive",
                    "denoising_steps": "N/A", "action_chunk_size": chunk_size,
                    "dof": config.action_dof,
                    "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                    "action_ms": action_ms, "action_oi": row.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        for dof in dof_values:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens=dof,
                bits=bits,
                max_batch_size=1,
            )
            if decode_results:
                row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_ms = row["time_ms"] * dof
                e2e_ms = vision_ms + vlm_ms + action_ms
                results.append({
                    "system": system, "setup": "A: Autoregressive (DoF Comparison)",
                    "denoising_steps": "N/A", "action_chunk_size": 1, "dof": dof,
                    "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                    "action_ms": action_ms, "action_oi": row.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        # Setup B: Small Diffusion (actual SmolVLA action expert)
        logger.info("    Setup B: Small Diffusion (SmolVLA action expert ~100M)")
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                df_act = get_smolvla_action_perf(
                    config, [system], [num_devices], bits,
                    denoising_steps=steps, action_chunk=chunk_size,
                    max_batch_size=1, logger=logger,
                )
                if not df_act.empty:
                    act_row = df_act[df_act["batch_size"] == 1].iloc[0]
                    action_ms = act_row["time_ms"]
                    e2e_ms = vision_ms + vlm_ms + action_ms
                    results.append({
                        "system": system, "setup": "B: Small Diffusion",
                        "denoising_steps": steps, "action_chunk_size": chunk_size,
                        "dof": config.action_dof,
                        "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                        "action_ms": action_ms, "action_oi": act_row.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        for dof in dof_values:
            for steps in denoising_steps_range:
                df_act = get_smolvla_action_perf(
                    config, [system], [num_devices], bits,
                    denoising_steps=steps, action_chunk=1,
                    max_batch_size=1, logger=logger,
                )
                if not df_act.empty:
                    act_row = df_act[df_act["batch_size"] == 1].iloc[0]
                    action_ms = act_row["time_ms"]
                    e2e_ms = vision_ms + vlm_ms + action_ms
                    results.append({
                        "system": system, "setup": "B: Small Diffusion (DoF Comparison)",
                        "denoising_steps": steps, "action_chunk_size": 1, "dof": dof,
                        "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                        "action_ms": action_ms, "action_oi": act_row.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        # Setup C: Large Diffusion (VLM-sized DiT)
        logger.info("    Setup C: Large Diffusion (VLM-sized ~1.7B)")
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model,
                    system=system,
                    num_devices=num_devices,
                    input_tokens=config.total_vlm_tokens,
                    output_tokens_parallel=config.action_dof * chunk_size,
                    self_attention=True,
                    bits=bits,
                    max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = vision_ms + vlm_ms + action_ms
                    results.append({
                        "system": system, "setup": "C: Large Diffusion",
                        "denoising_steps": steps, "action_chunk_size": chunk_size,
                        "dof": config.action_dof,
                        "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        for dof in dof_values:
            for steps in denoising_steps_range:
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model,
                    system=system,
                    num_devices=num_devices,
                    input_tokens=config.total_vlm_tokens,
                    output_tokens_parallel=dof,
                    self_attention=True,
                    bits=bits,
                    max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = vision_ms + vlm_ms + action_ms
                    results.append({
                        "system": system, "setup": "C: Large Diffusion (DoF Comparison)",
                        "denoising_steps": steps, "action_chunk_size": 1, "dof": dof,
                        "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        # Setup D: Autoregressive with Parallel Decoding
        logger.info("    Setup D: AR Parallel Decode")
        for chunk_size in action_chunk_sizes:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens_parallel=config.action_dof * chunk_size,
                self_attention=True,
                bits=bits,
                max_batch_size=1,
            )
            if action_results:
                ar = [r for r in action_results if r["batch_size"] == 1][0]
                action_ms = ar["time_ms"]
                e2e_ms = vision_ms + vlm_ms + action_ms
                results.append({
                    "system": system, "setup": "D: Autoregressive Parallel",
                    "denoising_steps": "N/A", "action_chunk_size": chunk_size,
                    "dof": config.action_dof,
                    "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                    "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        for dof in dof_values:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens_parallel=dof,
                self_attention=True,
                bits=bits,
                max_batch_size=1,
            )
            if action_results:
                ar = [r for r in action_results if r["batch_size"] == 1][0]
                action_ms = ar["time_ms"]
                e2e_ms = vision_ms + vlm_ms + action_ms
                results.append({
                    "system": system, "setup": "D: Autoregressive Parallel (DoF Comparison)",
                    "denoising_steps": "N/A", "action_chunk_size": 1, "dof": dof,
                    "vision_ms": vision_ms, "vlm_ms": vlm_ms,
                    "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

    df = pd.DataFrame(results)
    if not df.empty:
        out = output_path / "smolvla_autoregressive_vs_diffusion.csv"
        df.to_csv(out, index=False)
        logger.info(f"\nAR vs Diffusion results saved to {out}")

        for system in df["system"].unique():
            sys_df = df[df["system"] == system]
            logger.info(f"\n  System: {system} — E2E latency (ms) vs action chunk (DoF={config.action_dof}, steps={denoising_steps_range[0]})")
            logger.info(f"  {'Solution':<30}" + "".join(f"Chunk={c:>3}" for c in action_chunk_sizes))
            logger.info("  " + "-" * (30 + 9 * len(action_chunk_sizes)))
            for setup_label, setup_name, steps in [
                ("Autoregressive", "A: Autoregressive", "N/A"),
                ("Diffusion (Small)", "B: Small Diffusion", denoising_steps_range[0]),
                ("Diffusion (Large)", "C: Large Diffusion", denoising_steps_range[0]),
                ("AR Parallel", "D: Autoregressive Parallel", "N/A"),
            ]:
                fdf = sys_df[(sys_df["setup"] == setup_name) & (sys_df["dof"] == config.action_dof)]
                if steps != "N/A":
                    fdf = fdf[fdf["denoising_steps"] == steps]
                row_str = f"  {setup_label:<30}"
                for c in action_chunk_sizes:
                    v = fdf[fdf["action_chunk_size"] == c]["e2e_ms"]
                    row_str += f"{v.values[0]:>9.2f}" if not v.empty else f"{'N/A':>9}"
                logger.info(row_str)

    return df


# ==============================================================================
# Experiment 6: Device vs Server
# ==============================================================================

def compare_device_vs_server(
    config: SmolVLAConfig = SMOLVLA_CONFIG,
    denoising_steps: int = SMOLVLA_DENOISING_STEPS,
    bits: str = "bf16",
    image_resolution: int = 224,
    image_compression_ratio: float = 0.1,
    action_dof: int = SMOLVLA_ACTION_DOF,
    action_chunk_size: int = 1,
    logger=None,
) -> pd.DataFrame:
    """
    Compare on-device vs edge-server vs cloud inference for SmolVLA.

    Categories:
      1. On-device:    Jetson Thor, RTX 4090, B100 (no network)
      2. Edge-server:  RTX 4090, B100 + wired/WiFi/cellular networks
      3. Cloud:        B100 + local + cloud network
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    image_config = ImageConfig(resolution=image_resolution, channels=3,
                               bytes_per_pixel=1, compression_ratio=image_compression_ratio)
    action_config = ActionConfig(num_dof=action_dof, action_chunk_size=action_chunk_size,
                                 bytes_per_value=4)
    num_devices = 1
    results = []

    logger.info(f"\n  Image: {image_config.name}, Action: {action_config.name}")

    # Category 1: On-device
    for system in ["Jetson_AGX_Thor", "RTX_4090", "B100"]:
        try:
            df_v = get_smolvla_vision_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
            df_l = get_smolvla_vlm_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
            df_a = get_smolvla_action_perf(config, [system], [num_devices], bits,
                                           denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
            if df_v.empty or df_l.empty or df_a.empty:
                continue
            v_ms = df_v["time_ms"].values[0]
            l_ms = df_l["time_ms"].values[0]
            a_ms = df_a["time_ms"].values[0]
            e2e = v_ms + l_ms + a_ms
            results.append({
                "model": config.name, "category": "On-device", "system": system,
                "network": "N/A (Local)", "precision": bits,
                "vision_ms": v_ms, "vlm_ms": l_ms, "action_ms": a_ms,
                "network_image_ms": 0.0, "network_action_ms": 0.0,
                "e2e_compute_ms": e2e, "e2e_total_ms": e2e,
                "frequency_hz": 1000 / e2e, "freq_async_hz": 1000 / e2e,
                "denoising_steps": denoising_steps,
            })
        except Exception as e:
            logger.warning(f"    {system} on-device: {str(e)[:50]}")

    edge_networks = [ETHERNET_1G_CONFIG, ETHERNET_10G_CONFIG, WIFI_6_CONFIG, WIFI_7_CONFIG,
                     CELL_5G_SA_CONFIG, CELL_4G_LTE_CONFIG]

    # Category 2: Edge-server
    for system in ["RTX_4090", "B100"]:
        try:
            df_v = get_smolvla_vision_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
            df_l = get_smolvla_vlm_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
            df_a = get_smolvla_action_perf(config, [system], [num_devices], bits,
                                           denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
            if df_v.empty or df_l.empty or df_a.empty:
                continue
            v_ms = df_v["time_ms"].values[0]
            l_ms = df_l["time_ms"].values[0]
            a_ms = df_a["time_ms"].values[0]
            e2e_compute = v_ms + l_ms + a_ms
            inf_hz = 1000.0 / e2e_compute
            for net in edge_networks:
                img_lat = estimate_image_latency(net, image_config)["total_latency_ms"]
                act_lat = estimate_action_latency(net, action_config)["total_latency_ms"]
                e2e_total = e2e_compute + img_lat + act_lat
                net_hz = compute_network_throughput_hz(net, image_config, action_config)
                results.append({
                    "model": config.name, "category": "Edge-server", "system": system,
                    "network": net.name, "precision": bits,
                    "vision_ms": v_ms, "vlm_ms": l_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat,
                    "e2e_compute_ms": e2e_compute, "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": denoising_steps,
                })
        except Exception as e:
            logger.warning(f"    {system} edge-server: {str(e)[:50]}")

    # Category 3: Cloud
    try:
        system = "B100"
        df_v = get_smolvla_vision_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_l = get_smolvla_vlm_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_a = get_smolvla_action_perf(config, [system], [num_devices], bits,
                                       denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
        if not (df_v.empty or df_l.empty or df_a.empty):
            v_ms = df_v["time_ms"].values[0]
            l_ms = df_l["time_ms"].values[0]
            a_ms = df_a["time_ms"].values[0]
            e2e_compute = v_ms + l_ms + a_ms
            inf_hz = 1000.0 / e2e_compute
            for net_name, local_net, cloud_net in [
                ("Wired + Fast Cloud", ETHERNET_10G_CONFIG, CLOUD_FAST_CONFIG),
                ("4G + Slow Cloud", CELL_4G_LTE_CONFIG, CLOUD_SLOW_CONFIG),
            ]:
                img_lat = (estimate_image_latency(local_net, image_config)["total_latency_ms"] +
                           estimate_image_latency(cloud_net, image_config)["total_latency_ms"])
                act_lat = (estimate_action_latency(local_net, action_config)["total_latency_ms"] +
                           estimate_action_latency(cloud_net, action_config)["total_latency_ms"])
                e2e_total = e2e_compute + img_lat + act_lat
                net_hz = min(
                    compute_network_throughput_hz(local_net, image_config, action_config),
                    compute_network_throughput_hz(cloud_net, image_config, action_config),
                )
                results.append({
                    "model": config.name, "category": "Cloud", "system": system,
                    "network": net_name, "precision": bits,
                    "vision_ms": v_ms, "vlm_ms": l_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat,
                    "e2e_compute_ms": e2e_compute, "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": denoising_steps,
                })
    except Exception as e:
        logger.warning(f"    Cloud B100: {str(e)[:50]}")

    return pd.DataFrame(results)


def print_device_vs_server_summary(df: pd.DataFrame, logger=None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    if df.empty:
        return
    for category in ["On-device", "Edge-server", "Cloud"]:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue
        logger.info(f"\n  {category}:")
        logger.info(f"  {'System':<20} {'Network':<30} {'Compute':>10} {'Total':>10} {'Hz Sync':>10} {'Hz Async':>10}")
        for _, row in cat_df.sort_values("e2e_total_ms").iterrows():
            logger.info(f"  {row['system']:<20} {row['network']:<30} "
                        f"{row['e2e_compute_ms']:>10.2f} {row['e2e_total_ms']:>10.2f} "
                        f"{row['frequency_hz']:>10.1f} {row.get('freq_async_hz', row['frequency_hz']):>10.1f}")


def run_device_vs_server_comparison(
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> dict:
    """Exp 6: Run device vs server comparison for all SmolVLA configs."""
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}SMOLVLA DEVICE VS SERVER COMPARISON")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_results = {}
    all_dfs = []

    for config in ALL_SMOLVLA_CONFIGS:
        logger.info(f"\n  {config.name}:")
        df = compare_device_vs_server(config=config, logger=logger)
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        print_device_vs_server_summary(df, logger=logger)

    if all_dfs:
        out = output_path / "smolvla_device_vs_server.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(out, index=False)
        logger.info(f"\nResults saved to {out}")

    return all_results


# ==============================================================================
# Experiment 7: Device-Server Collaboration
# VLM on server, action expert on device
# ==============================================================================

def compare_device_server_collaboration(
    config: SmolVLAConfig = SMOLVLA_CONFIG,
    denoising_steps: int = SMOLVLA_DENOISING_STEPS,
    bits: str = "bf16",
    image_resolution: int = 224,
    image_compression_ratio: float = 0.1,
    action_dof: int = SMOLVLA_ACTION_DOF,
    action_chunk_size: int = 1,
    server_system: str = "B100",
    device_system: str = "Jetson_AGX_Thor",
    network_configs: list = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 7: Device-Server Collaboration for SmolVLA.

    Scenarios:
      1. Device Only:              Jetson runs vision + VLM + action expert locally
      2. Server Only:              Server runs all; sends action back to robot
      3. Device-Server Collab:     Server runs VLM; KV cache sent to device; device runs action expert
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if network_configs is None:
        network_configs = ALL_NETWORK_CONFIGS

    image_config = ImageConfig(resolution=image_resolution, channels=3,
                               bytes_per_pixel=1, compression_ratio=image_compression_ratio)
    action_config = ActionConfig(num_dof=action_dof, action_chunk_size=action_chunk_size,
                                 bytes_per_value=4)
    num_devices = 1

    kv_cache_configs = get_kvcache_configs_from_model_config(
        model_name=config.vlm_model,
        seq_lengths=[config.total_vlm_tokens],
        pretty_name=f"{config.name}-VLM",
    )
    if not kv_cache_configs:
        logger.warning(f"Could not create KV cache config for {config.vlm_model}")
        return pd.DataFrame()
    kv_cache_config = kv_cache_configs[0]

    results = []
    logger.info(f"\n  Image: {image_config.name}, KV Cache: {kv_cache_config.name}")
    logger.info(f"  Server: {server_system}, Device: {device_system}")

    # Scenario 1: Device Only
    try:
        df_v = get_smolvla_vision_perf(config, [device_system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_l = get_smolvla_vlm_perf(config, [device_system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_a = get_smolvla_action_perf(config, [device_system], [num_devices], bits,
                                       denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
        if not (df_v.empty or df_l.empty or df_a.empty):
            v_ms = df_v["time_ms"].values[0]
            l_ms = df_l["time_ms"].values[0]
            a_ms = df_a["time_ms"].values[0]
            e2e = v_ms + l_ms + a_ms
            results.append({
                "model": config.name, "category": "Device Only",
                "server_system": "N/A", "device_system": device_system,
                "network": "N/A (Local)", "precision": bits,
                "vision_ms": v_ms, "vlm_ms": l_ms, "action_ms": a_ms,
                "network_image_ms": 0.0, "network_action_ms": 0.0, "network_kv_cache_ms": 0.0,
                "e2e_total_ms": e2e, "frequency_hz": 1000 / e2e, "freq_async_hz": 1000 / e2e,
                "denoising_steps": denoising_steps,
            })
    except Exception as e:
        logger.warning(f"    Device Only: {str(e)[:50]}")

    # Scenario 2: Server Only
    try:
        df_v = get_smolvla_vision_perf(config, [server_system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_l = get_smolvla_vlm_perf(config, [server_system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_a = get_smolvla_action_perf(config, [server_system], [num_devices], bits,
                                       denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
        if not (df_v.empty or df_l.empty or df_a.empty):
            v_ms = df_v["time_ms"].values[0]
            l_ms = df_l["time_ms"].values[0]
            a_ms = df_a["time_ms"].values[0]
            e2e_compute = v_ms + l_ms + a_ms
            inf_hz = 1000.0 / e2e_compute
            for net in network_configs:
                img_lat = estimate_image_latency(net, image_config)["total_latency_ms"]
                act_lat = estimate_action_latency(net, action_config)["total_latency_ms"]
                e2e_total = e2e_compute + img_lat + act_lat
                net_hz = compute_network_throughput_hz(net, image_config, action_config)
                results.append({
                    "model": config.name, "category": "Server Only",
                    "server_system": server_system, "device_system": "N/A",
                    "network": net.name, "precision": bits,
                    "vision_ms": v_ms, "vlm_ms": l_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat, "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": denoising_steps,
                })
            # Cloud pairs
            for net_name, local_net, cloud_net in [
                ("Wired + Fast Cloud", ETHERNET_10G_CONFIG, CLOUD_FAST_CONFIG),
                ("4G + Slow Cloud", CELL_4G_LTE_CONFIG, CLOUD_SLOW_CONFIG),
            ]:
                img_lat = (estimate_image_latency(local_net, image_config)["total_latency_ms"] +
                           estimate_image_latency(cloud_net, image_config)["total_latency_ms"])
                act_lat = (estimate_action_latency(local_net, action_config)["total_latency_ms"] +
                           estimate_action_latency(cloud_net, action_config)["total_latency_ms"])
                e2e_total = e2e_compute + img_lat + act_lat
                net_hz = min(
                    compute_network_throughput_hz(local_net, image_config, action_config),
                    compute_network_throughput_hz(cloud_net, image_config, action_config),
                )
                results.append({
                    "model": config.name, "category": "Server Only",
                    "server_system": server_system, "device_system": "N/A",
                    "network": net_name, "precision": bits,
                    "vision_ms": v_ms, "vlm_ms": l_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat, "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": denoising_steps,
                })
    except Exception as e:
        logger.warning(f"    Server Only: {str(e)[:50]}")

    # Scenario 3: Device-Server Collaboration (VLM on server, action expert on device)
    try:
        df_l = get_smolvla_vlm_perf(config, [server_system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_a = get_smolvla_action_perf(config, [device_system], [num_devices], bits,
                                       denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
        if not (df_l.empty or df_a.empty):
            l_ms = df_l["time_ms"].values[0]
            a_ms = df_a["time_ms"].values[0]
            server_hz = 1000.0 / l_ms
            device_hz = 1000.0 / a_ms
            for net in network_configs:
                img_lat = estimate_image_latency(net, image_config)["total_latency_ms"]
                kv_lat = estimate_kvcache_latency(net, kv_cache_config)["total_latency_ms"]
                e2e_total = img_lat + l_ms + kv_lat + a_ms
                net_hz = compute_network_throughput_hz(net, image_config, kvcache_config=kv_cache_config)
                results.append({
                    "model": config.name, "category": "Device-Server Collaboration",
                    "server_system": server_system, "device_system": device_system,
                    "network": net.name, "precision": bits,
                    "vision_ms": 0.0, "vlm_ms": l_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": 0.0, "network_kv_cache_ms": kv_lat,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(server_hz, net_hz, device_hz),
                    "denoising_steps": denoising_steps,
                })
            for net_name, local_net, cloud_net in [
                ("Wired + Fast Cloud", ETHERNET_10G_CONFIG, CLOUD_FAST_CONFIG),
                ("4G + Slow Cloud", CELL_4G_LTE_CONFIG, CLOUD_SLOW_CONFIG),
            ]:
                img_lat = (estimate_image_latency(local_net, image_config)["total_latency_ms"] +
                           estimate_image_latency(cloud_net, image_config)["total_latency_ms"])
                kv_lat = (estimate_kvcache_latency(local_net, kv_cache_config)["total_latency_ms"] +
                          estimate_kvcache_latency(cloud_net, kv_cache_config)["total_latency_ms"])
                e2e_total = img_lat + l_ms + kv_lat + a_ms
                net_hz = min(
                    compute_network_throughput_hz(local_net, image_config, kvcache_config=kv_cache_config),
                    compute_network_throughput_hz(cloud_net, image_config, kvcache_config=kv_cache_config),
                )
                results.append({
                    "model": config.name, "category": "Device-Server Collaboration",
                    "server_system": server_system, "device_system": device_system,
                    "network": net_name, "precision": bits,
                    "vision_ms": 0.0, "vlm_ms": l_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": 0.0, "network_kv_cache_ms": kv_lat,
                    "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(server_hz, net_hz, device_hz),
                    "denoising_steps": denoising_steps,
                })
    except Exception as e:
        logger.warning(f"    Device-Server Collaboration: {str(e)[:50]}")

    return pd.DataFrame(results)


def print_device_server_collaboration_summary(df: pd.DataFrame, logger=None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    if df.empty:
        return
    for category in ["Device Only", "Server Only", "Device-Server Collaboration"]:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue
        best = cat_df.loc[cat_df["e2e_total_ms"].idxmin()]
        logger.info(f"  {category}: best={best['e2e_total_ms']:.2f} ms "
                    f"({1000/best['e2e_total_ms']:.1f} Hz sync, "
                    f"{best.get('freq_async_hz', 1000/best['e2e_total_ms']):.1f} Hz async)")


def run_device_server_collaboration_comparison(
    output_dir: str = "perf_results",
    network_configs: list = None,
    server_system: str = "B100",
    device_system: str = "Jetson_AGX_Thor",
    experiment_num: int = None,
    logger=None,
) -> dict:
    """Exp 7: Device-Server Collaboration for all SmolVLA configs."""
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}SMOLVLA DEVICE-SERVER COLLABORATION")
    logger.info("=" * 130)
    logger.info(f"Server: {server_system}, Device: {device_system}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_results = {}
    all_dfs = []

    for config in ALL_SMOLVLA_CONFIGS:
        logger.info(f"\n  {config.name}:")
        df = compare_device_server_collaboration(
            config=config,
            server_system=server_system,
            device_system=device_system,
            network_configs=network_configs,
            logger=logger,
        )
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        print_device_server_collaboration_summary(df, logger=logger)

    if all_dfs:
        out = output_path / "smolvla_device_server_collaboration.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(out, index=False)
        logger.info(f"\nResults saved to {out}")

    return all_results


# ==============================================================================
# Summary
# ==============================================================================

def print_summary(results: dict[str, pd.DataFrame], logger=None) -> None:
    """Print a summary table of SmolVLA E2E performance."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\nSmolVLA model characteristics:")
    logger.info(f"  Vision:  SigLIP-SO400M/14 ({SMOLVLA_VISION_ENCODING_TOKENS} ViT patches -> {SMOLVLA_VLM_VISION_TOKENS} tokens to VLM via layer skipping)")
    logger.info(f"  VLM:     SmolLM2-1.7B ({SMOLVLA_TOTAL_VLM_TOKENS} prefill tokens)")
    logger.info(f"  Action:  Flow Matching Transformer ~100M, {SMOLVLA_DENOISING_STEPS} steps, chunk={SMOLVLA_ACTION_CHUNK}")

    logger.info("\n" + "=" * 120)
    logger.info("SmolVLA Performance Summary")
    logger.info("=" * 120)

    if "e2e" in results and not results["e2e"].empty:
        df = results["e2e"]
        logger.info("-" * 120)
        logger.info(
            f"{'Hardware':<18} {'Chips':<6} {'Batch':<6} "
            f"{'Vision (ms)':>14} {'VLM (ms)':>14} {'Action (ms)':>14} "
            f"{'E2E (ms)':>12} {'Hz':>10}"
        )
        logger.info("-" * 120)
        for _, row in df.iterrows():
            e2e = row["e2e_time_ms"]
            hz = 1000 / e2e if e2e > 0 else 0
            logger.info(
                f"{row['hardware.name']:<18} {int(row['hardware.num_chips']):<6} "
                f"{int(row['batch_size']):<6} "
                f"{row['vision_time_ms']:>14.2f} {row['vlm_time_ms']:>14.2f} "
                f"{row['action_time_ms']:>14.2f} {e2e:>12.2f} {hz:>10.1f}"
            )
        logger.info("-" * 120)
    else:
        logger.warning("No E2E results available.")


if __name__ == "__main__":
    logger = setup_logging("perf_results/smolvla_perf.log")

    system_list = ["A100_80GB", "H100", "B100", "RTX_3090", "RTX_4090", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"
    denoising_steps = SMOLVLA_DENOISING_STEPS

    runall = True
    run_exp_1 = True
    run_exp_2 = True
    run_exp_3 = True
    run_exp_4 = True
    run_exp_5 = True
    run_exp_6 = True
    run_exp_7 = True

    logger.info("=" * 100)
    logger.info("Starting SmolVLA Performance Evaluation (7 Experiments)")
    logger.info("=" * 100)
    logger.info(f"Systems: {system_list}")
    logger.info(f"Devices: {num_device_list}")
    logger.info(f"Precision: {bits}")

    exp_counter = 0

    if runall or run_exp_1:
        exp_counter += 1
        all_results = get_all_smolvla_perf(
            system_list=system_list,
            num_device_list=num_device_list,
            bits=bits,
            denoising_steps=denoising_steps,
            experiment_num=exp_counter,
            logger=logger,
        )
        print_summary(all_results.get("smolvla", {}), logger=logger)

    if runall or run_exp_2:
        exp_counter += 1
        size_results, size_configs = get_model_size_scaling_perf(
            system_list=["B100", "RTX_4090", "Jetson_AGX_Thor"],
            num_device_list=[1],
            bits=bits,
            denoising_steps=denoising_steps,
            experiment_num=exp_counter,
            logger=logger,
        )
        print_model_size_scaling_summary(size_results, size_configs, logger=logger)

    if runall or run_exp_3:
        exp_counter += 1
        run_long_context_experiment(
            experiment_num=exp_counter,
            logger=logger,
        )

    if runall or run_exp_4:
        exp_counter += 1
        compare_denoising_steps_action_lengths(
            experiment_num=exp_counter,
            logger=logger,
        )

    if runall or run_exp_5:
        exp_counter += 1
        compare_autoregressive_vs_diffusion(
            experiment_num=exp_counter,
            logger=logger,
        )

    if runall or run_exp_6:
        exp_counter += 1
        run_device_vs_server_comparison(
            experiment_num=exp_counter,
            logger=logger,
        )

    if runall or run_exp_7:
        exp_counter += 1
        run_device_server_collaboration_comparison(
            server_system="B100",
            device_system="Jetson_AGX_Thor",
            experiment_num=exp_counter,
            logger=logger,
        )

    logger.info("\n" + "=" * 100)
    logger.info(f"COMPLETED {exp_counter} EXPERIMENT(S)")
    logger.info("=" * 100)
