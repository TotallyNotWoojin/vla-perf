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
Qwen2-VL-Based VLA Performance Evaluation Script

Models Qwen2-VL-7B as a VLA backbone (as used in, e.g., CogACT).
- Vision:  Qwen2-VL ViT-G/14 (~675M, 32 layers, 1152 hidden)
           224px input -> 16x16=256 raw patches -> 64 merged tokens (2x2 merge)
- VLM:     Qwen2-7B LLM (28 layers, 3584 hidden, GQA 28/4 heads)
           Processes merged visual tokens + language instruction tokens
- Action:  Autoregressive decode of 7 continuous action tokens
           (can swap for a diffusion head as in CogACT)

Performance modeling breakdown:
1. Vision encoding (Qwen2-VL ViT prefill)
2. VLM prefill     (Qwen2-7B processes merged visual + text tokens)
3. Action decode   (7 action tokens generated autoregressively)

Experiments (7 total, mirroring pi0_perf.py depth):
  1. Base E2E performance across hardware
  2. Model size scaling (Qwen2-7B -> 13B -> 70B backbone variants)
  3. Long context (multi-camera: 1-5 image frames)
  4. Action tokens x context length sweep (AR-specific analog of steps x chunks)
  5. Autoregressive vs Diffusion comparison (AR baseline vs hypothetical diffusion head)
  6. Device vs Server (on-device / edge-server / cloud + network latency)
  7. Device-Server Collaboration (VLM prefill on server, AR decode on device)

Reference:
    Qwen2-VL: arxiv.org/abs/2409.12191
    CogACT:   arxiv.org/abs/2411.19650
    genz/GenZ/Models/Model_sets/vla_models.py -> Qwen2-VL model configs
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
# Qwen2-VL-7B VLA Architecture Constants
# ==============================================================================
QWEN2VL_RAW_VISION_TOKENS = 256    # 224px / 14px patch = 16x16 = 256 raw tokens
QWEN2VL_MERGED_VISION_TOKENS = 64  # after 2x2 spatial merge in the VLM projector
QWEN2VL_LANGUAGE_TOKENS = 32       # typical instruction length
QWEN2VL_TOTAL_VLM_TOKENS = QWEN2VL_MERGED_VISION_TOKENS + QWEN2VL_LANGUAGE_TOKENS
QWEN2VL_ACTION_TOKENS = 7          # 7-DoF action output (same convention as OpenVLA)
QWEN2VL_ACTION_DOF = 7


@dataclass
class Qwen2VLAConfig:
    """Configuration for a Qwen2-VL-based VLA model variant."""
    name: str
    vision_model: str
    vlm_model: str
    raw_vision_tokens_per_frame: int = QWEN2VL_RAW_VISION_TOKENS
    merged_vision_tokens_per_frame: int = QWEN2VL_MERGED_VISION_TOKENS
    language_tokens: int = QWEN2VL_LANGUAGE_TOKENS
    num_frames: int = 1
    action_tokens: int = QWEN2VL_ACTION_TOKENS
    action_dof: int = QWEN2VL_ACTION_DOF

    @property
    def total_raw_vision_tokens(self) -> int:
        return self.raw_vision_tokens_per_frame * self.num_frames

    @property
    def total_vlm_tokens(self) -> int:
        return self.merged_vision_tokens_per_frame * self.num_frames + self.language_tokens


QWEN2VLA_CONFIG = Qwen2VLAConfig(
    name="qwen2-vl-7b-vla",
    vision_model="qwen2-vl-7b-vision",
    vlm_model="qwen2-vl-7b-llm",
)

ALL_QWEN2VLA_CONFIGS = [QWEN2VLA_CONFIG]


def create_action_expert_config_from_vlm(vlm_config: ModelConfig, name: str) -> ModelConfig:
    """Derive a diffusion action expert from a VLM config (hidden//2, ffn//4)."""
    action_config = copy.deepcopy(vlm_config)
    action_config.model = name
    action_config.hidden_size = vlm_config.hidden_size // 2
    action_config.intermediate_size = vlm_config.intermediate_size // 4
    action_config.vocab_size = 0
    return action_config


# ==============================================================================
# Component perf functions
# ==============================================================================

def get_qwen2vla_vision_perf(
    config: Qwen2VLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """Evaluate Qwen2-VL ViT vision encoding."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model=config.vision_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_raw_vision_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_qwen2vla_vlm_prefill_perf(
    config: Qwen2VLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """Evaluate Qwen2-7B VLM prefill (merged visual + language tokens)."""
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


def get_qwen2vla_action_decode_perf(
    config: Qwen2VLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    action_tokens: int = None,
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """Evaluate Qwen2-7B autoregressive action token decode."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if action_tokens is None:
        action_tokens = config.action_tokens

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_decode_perf(
                model=config.vlm_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens=action_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        return df
    return get_optimal_df(df, apply_pareto=True)


# ==============================================================================
# Experiment 1: Base E2E
# ==============================================================================

def get_qwen2vla_e2e_perf(
    config: Qwen2VLAConfig = QWEN2VLA_CONFIG,
    system_list: list[str] = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Evaluate end-to-end Qwen2-VL VLA performance across all components."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    df_vision = get_qwen2vla_vision_perf(config, system_list, num_device_list, bits, logger=logger)
    results["vision"] = df_vision
    df_vision.to_csv(output_path / f"{config.name}_vision_perf.csv", index=False)

    df_prefill = get_qwen2vla_vlm_prefill_perf(config, system_list, num_device_list, bits, logger=logger)
    results["vlm_prefill"] = df_prefill
    df_prefill.to_csv(output_path / f"{config.name}_vlm_prefill_perf.csv", index=False)

    df_decode = get_qwen2vla_action_decode_perf(config, system_list, num_device_list, bits, logger=logger)
    results["action_decode"] = df_decode
    df_decode.to_csv(output_path / f"{config.name}_action_decode_perf.csv", index=False)

    if df_vision.empty or df_prefill.empty or df_decode.empty:
        logger.warning(f"One or more component DataFrames empty for {config.name} — skipping E2E.")
        results["e2e"] = pd.DataFrame()
        return results

    group_cols = ["hardware.name", "hardware.num_chips", "batch_size"]
    vision_times = df_vision[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "vision_time_ms"})
    prefill_times = df_prefill[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "prefill_time_ms"})
    decode_times = df_decode[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "decode_time_ms"})
    decode_times["decode_time_ms"] *= config.action_tokens  # per-token latency -> total

    df_merged = vision_times.merge(prefill_times, on=group_cols, how="inner")
    df_merged = df_merged.merge(decode_times, on=group_cols, how="inner")
    df_merged["e2e_time_ms"] = (df_merged["vision_time_ms"] + df_merged["prefill_time_ms"] +
                                df_merged["decode_time_ms"])
    df_merged["model.name"] = config.name
    df_merged["model.stage"] = "e2e"
    df_merged["model.dec_steps"] = config.action_tokens
    df_merged["model.seq_len_inference_prefill"] = config.total_vlm_tokens

    df_e2e = df_merged[[
        "model.name", "model.stage", "model.dec_steps",
        "model.seq_len_inference_prefill",
        "hardware.name", "hardware.num_chips", "batch_size",
        "vision_time_ms", "prefill_time_ms", "decode_time_ms", "e2e_time_ms",
    ]]
    results["e2e"] = df_e2e
    df_e2e.to_csv(output_path / f"{config.name}_e2e_perf.csv", index=False)

    return results


def get_all_qwen2vla_perf(
    system_list: list[str] = ["A100_80GB", "H100", "B100", "RTX_3090", "RTX_4090", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run Exp 1: base E2E for all Qwen2-VL VLA configs."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}QWEN2-VL VLA BASE PERFORMANCE EVALUATION")
    logger.info("=" * 130)

    all_results = {}
    for config in ALL_QWEN2VLA_CONFIGS:
        logger.info(f"\nEvaluating {config.name}...")
        results = get_qwen2vla_e2e_perf(config, system_list, num_device_list, bits, output_dir, logger)
        all_results[config.name] = results

    all_e2e = [r["e2e"] for r in all_results.values() if not r["e2e"].empty]
    if all_e2e:
        pd.concat(all_e2e, ignore_index=True).to_csv(
            Path(output_dir) / "qwen2vla_family_e2e_perf.csv", index=False
        )

    return all_results


# ==============================================================================
# Experiment 2: Model Size Scaling
# ==============================================================================

def get_model_size_scaling_perf(
    system_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> tuple[dict, dict]:
    """
    Exp 2: Evaluate Qwen2-VL VLA scaling with different LLM backbone sizes.

    Variants:
      qwen2-vl-7b:  qwen2-vl-7b-vision + qwen2-vl-7b-llm   (actual)
      qwen2-vl-13b: qwen2-vl-7b-vision + llama2_13b proxy   (hypothetical)
      qwen2-vl-70b: qwen2-vl-7b-vision + llama2_70b proxy   (hypothetical)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = [1]

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}QWEN2-VL VLA MODEL SIZE SCALING EVALUATION")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model_variants = []

    # 7B baseline (actual Qwen2-VL-7B)
    vlm_7b = copy.deepcopy(MODEL_DICT.get_model("qwen2-vl-7b-llm"))
    vlm_7b.vocab_size = 0
    vlm_7b.model = "qwen2-vl-7b-llm-no-vocab"
    MODEL_DICT.add_model(vlm_7b)
    model_variants.append(Qwen2VLAConfig(
        name="qwen2-vl-7b", vision_model="qwen2-vl-7b-vision", vlm_model=vlm_7b.model,
    ))

    # 13B variant (llama2_13b as proxy for 13B class LLM)
    vlm_13b = copy.deepcopy(MODEL_DICT.get_model("llama2_13b"))
    vlm_13b.vocab_size = 0
    vlm_13b.model = "qwen2-vl-13b-proxy-no-vocab"
    MODEL_DICT.add_model(vlm_13b)
    model_variants.append(Qwen2VLAConfig(
        name="qwen2-vl-13b", vision_model="qwen2-vl-7b-vision", vlm_model=vlm_13b.model,
    ))

    # 70B variant
    vlm_70b = copy.deepcopy(MODEL_DICT.get_model("llama2_70b"))
    vlm_70b.vocab_size = 0
    vlm_70b.model = "qwen2-vl-70b-proxy-no-vocab"
    MODEL_DICT.add_model(vlm_70b)
    model_variants.append(Qwen2VLAConfig(
        name="qwen2-vl-70b", vision_model="qwen2-vl-7b-vision", vlm_model=vlm_70b.model,
    ))

    # Log parameter counts
    logger.info("\n--- Model Parameter Counts ---")
    component_params_data = []
    for cfg in model_variants:
        vision_cfg = MODEL_DICT.get_model(cfg.vision_model)
        vlm_cfg = MODEL_DICT.get_model(cfg.vlm_model)
        vp = calculate_transformer_params(vision_cfg)
        lp = calculate_transformer_params(vlm_cfg)
        total = vp + lp
        logger.info(f"  {cfg.name}: vision={format_param_count(vp)}, vlm={format_param_count(lp)}, "
                    f"total={format_param_count(total)}")
        component_params_data.append({
            "model": cfg.name, "vision_params_M": vp / 1e6,
            "vlm_params_M": lp / 1e6, "total_params_M": total / 1e6,
        })
    pd.DataFrame(component_params_data).to_csv(
        output_path / "qwen2vla_model_size_scaling_params.csv", index=False
    )

    all_results = {}
    all_e2e = []
    for cfg in model_variants:
        logger.info(f"\n  Evaluating {cfg.name}...")
        results = get_qwen2vla_e2e_perf(cfg, system_list, num_device_list, bits, output_dir, logger)
        all_results[cfg.name] = results
        if not results["e2e"].empty:
            all_e2e.append(results["e2e"])

    if all_e2e:
        pd.concat(all_e2e, ignore_index=True).to_csv(
            output_path / "qwen2vla_model_size_scaling.csv", index=False
        )

    return all_results, {cfg.name: cfg for cfg in model_variants}


def print_model_size_scaling_summary(all_results: dict, model_configs: dict, logger=None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("\n--- Qwen2-VL VLA Model Size Scaling Summary (Batch=1, Chips=1) ---")
    logger.info(f"{'Model':<18} {'Hardware':<20} {'Vision (ms)':>12} {'Prefill (ms)':>13} {'Decode (ms)':>12} {'E2E (ms)':>12} {'Hz':>8}")
    logger.info("-" * 100)
    for model_name, results in all_results.items():
        if "e2e" not in results or results["e2e"].empty:
            continue
        df = results["e2e"]
        sub = df[(df["hardware.num_chips"] == 1) & (df["batch_size"] == 1)]
        for _, row in sub.iterrows():
            e2e = row["e2e_time_ms"]
            logger.info(f"{model_name:<18} {row['hardware.name']:<20} "
                        f"{row['vision_time_ms']:>12.2f} {row['prefill_time_ms']:>13.2f} "
                        f"{row['decode_time_ms']:>12.2f} {e2e:>12.2f} {1000/e2e:>8.1f}")


# ==============================================================================
# Experiment 3: Long Context (multi-camera / multi-frame)
# ==============================================================================

def run_long_context_experiment(
    config: Qwen2VLAConfig = QWEN2VLA_CONFIG,
    system_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_frames_list: list[int] = [1, 2, 3, 4, 5],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 3: Evaluate Qwen2-VL VLA latency as number of camera frames scales.

    Each additional frame adds 256 ViT tokens (encoded) and 64 merged tokens to the VLM context.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}QWEN2-VL VLA LONG CONTEXT (MULTI-CAMERA) EXPERIMENT")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = []
    for num_frames in num_frames_list:
        cfg = Qwen2VLAConfig(
            name=config.name,
            vision_model=config.vision_model,
            vlm_model=config.vlm_model,
            num_frames=num_frames,
        )
        total_raw = cfg.total_raw_vision_tokens
        total_vlm = cfg.total_vlm_tokens
        kv_mb = calculate_kv_cache_size_mb(config.vlm_model, total_vlm, bits)
        logger.info(f"\n  Frames={num_frames}: raw_vision={total_raw}, vlm_tokens={total_vlm}, KV cache={kv_mb:.1f} MB")

        for system in system_list:
            df_v = get_qwen2vla_vision_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)
            df_p = get_qwen2vla_vlm_prefill_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)
            df_d = get_qwen2vla_action_decode_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)

            if df_v.empty or df_p.empty or df_d.empty:
                continue

            v_ms = df_v["time_ms"].values[0]
            p_ms = df_p["time_ms"].values[0]
            d_ms = df_d["time_ms"].values[0] * config.action_tokens
            e2e_ms = v_ms + p_ms + d_ms

            results.append({
                "model": config.name, "system": system, "num_frames": num_frames,
                "total_raw_vision_tokens": total_raw, "total_vlm_tokens": total_vlm,
                "kv_cache_mb": kv_mb,
                "vision_ms": v_ms, "prefill_ms": p_ms, "decode_ms": d_ms,
                "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        out = output_path / "qwen2vla_long_context.csv"
        df.to_csv(out, index=False)
        logger.info(f"\nLong context results saved to {out}")
        logger.info("\n--- Long Context Summary ---")
        logger.info(f"{'Frames':>7} {'Vis Tok':>8} {'VLM Tok':>8} {'KV MB':>7} "
                    f"{'System':<20} {'Vision':>10} {'Prefill':>10} {'Decode':>10} {'E2E':>10} {'Hz':>8}")
        logger.info("-" * 110)
        for _, row in df.iterrows():
            logger.info(f"{int(row['num_frames']):>7} {int(row['total_raw_vision_tokens']):>8} "
                        f"{int(row['total_vlm_tokens']):>8} {row['kv_cache_mb']:>7.1f} "
                        f"{row['system']:<20} {row['vision_ms']:>10.2f} {row['prefill_ms']:>10.2f} "
                        f"{row['decode_ms']:>10.2f} {row['e2e_ms']:>10.2f} {row['frequency_hz']:>8.1f}")

    return df


# ==============================================================================
# Experiment 4: Action Tokens x Context Length sweep
# (AR-model analog of denoising steps x action chunk)
# ==============================================================================

def compare_action_tokens_context_lengths(
    config: Qwen2VLAConfig = QWEN2VLA_CONFIG,
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    action_tokens_range: list[int] = [1, 7, 14, 28, 56],
    context_lengths_range: list[int] = [64, 96, 192, 384, 768],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 4: 2D sweep over action token count and VLM context length for Qwen2-VL VLA.

    For a pure-AR model, the latency-scaling axes are:
      - action_tokens: number of tokens decoded autoregressively (DoF x chunk_size)
      - context_length: total prefill tokens (vision + language)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}QWEN2-VL VLA ACTION TOKENS x CONTEXT LENGTH SWEEP")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = []
    for system in systems:
        logger.info(f"\n  System: {system}")
        for context_len in context_lengths_range:
            for action_tokens in action_tokens_range:
                # Vision prefill cost scales with context
                # Approximate: extra context beyond base goes entirely to VLM
                vision_toks = QWEN2VL_RAW_VISION_TOKENS  # Vision encoder is fixed
                vlm_toks = context_len  # Vary total VLM context

                model_results_p = collect_prefill_perf(
                    model=config.vlm_model,
                    system=system,
                    num_devices=1,
                    input_tokens=vlm_toks,
                    bits=bits,
                    max_batch_size=1,
                )
                model_results_d = collect_decode_perf(
                    model=config.vlm_model,
                    system=system,
                    num_devices=1,
                    input_tokens=vlm_toks,
                    output_tokens=action_tokens,
                    bits=bits,
                    max_batch_size=1,
                )
                if not model_results_p or not model_results_d:
                    continue

                prefill_row = [r for r in model_results_p if r["batch_size"] == 1]
                decode_row = [r for r in model_results_d if r["batch_size"] == 1]
                if not prefill_row or not decode_row:
                    continue

                p_ms = prefill_row[0]["time_ms"]
                d_ms = decode_row[0]["time_ms"] * action_tokens
                e2e_ms = p_ms + d_ms

                results.append({
                    "model": config.name, "system": system,
                    "context_length": context_len, "action_tokens": action_tokens,
                    "prefill_ms": p_ms, "decode_ms": d_ms,
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    out = output_path / "qwen2vla_action_tokens_context_lengths.csv"
    df.to_csv(out, index=False)
    logger.info(f"\nAction tokens x context length results saved to {out}")

    # Print 2D latency grid per system
    default_action = config.action_tokens
    default_context = QWEN2VL_TOTAL_VLM_TOKENS
    for system in df["system"].unique():
        sys_df = df[df["system"] == system]
        pivot = sys_df.pivot(index="action_tokens", columns="context_length", values="e2e_ms")
        logger.info(f"\n  System: {system} — E2E latency (ms): rows=action_tokens, cols=context_length")
        header = f"{'Act Tok \\ Ctx':<14}" + "".join(f"{c:>10}" for c in context_lengths_range)
        logger.info(header)
        logger.info("-" * (14 + 10 * len(context_lengths_range)))
        for at in action_tokens_range:
            if at in pivot.index:
                row_str = f"{at:<14}" + "".join(
                    f"{pivot.loc[at, c]:>10.2f}" if c in pivot.columns else f"{'N/A':>10}"
                    for c in context_lengths_range
                )
                logger.info(row_str)

        # Speedup vs baseline
        baseline_row = sys_df[
            (sys_df["action_tokens"] == default_action) &
            (sys_df["context_length"] == default_context)
        ]
        if not baseline_row.empty:
            baseline_ms = baseline_row["e2e_ms"].values[0]
            logger.info(f"\n  Speedup vs baseline (action_tokens={default_action}, context={default_context})")
            logger.info(header)
            logger.info("-" * (14 + 10 * len(context_lengths_range)))
            for at in action_tokens_range:
                if at in pivot.index:
                    row_str = f"{at:<14}" + "".join(
                        f"{baseline_ms / pivot.loc[at, c]:>10.2f}x" if c in pivot.columns else f"{'N/A':>10}"
                        for c in context_lengths_range
                    )
                    logger.info(row_str)

    return df


# ==============================================================================
# Experiment 5: Autoregressive vs Diffusion
# ==============================================================================

def compare_autoregressive_vs_diffusion(
    config: Qwen2VLAConfig = QWEN2VLA_CONFIG,
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_devices: int = 1,
    bits: str = "bf16",
    denoising_steps_range: list[int] = [10],
    action_chunk_sizes: list[int] = [1, 7, 14, 28, 56],
    dof_values: list[int] = [7, 14, 21, 28, 35, 42],
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 5: Compare action generation strategies for Qwen2-VL VLA.

    A) Autoregressive (Qwen2-7B, the actual model)
    B) Hypothetical Small Diffusion (derived 3.5B action expert)
    C) Hypothetical Large Diffusion (VLM-sized ~7B DiT)
    D) Autoregressive with Parallel Decoding
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}QWEN2-VL VLA AUTOREGRESSIVE VS DIFFUSION")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    action_predictor_model = "qwen2-vl-7b-action-predictor"
    vlm_ap_config = copy.deepcopy(MODEL_DICT.get_model(config.vlm_model))
    vlm_ap_config.model = action_predictor_model
    vlm_ap_config.vocab_size = 0
    MODEL_DICT.add_model(vlm_ap_config)

    small_diff_model = "qwen2-vl-small-diffusion-expert"
    small_diff_config = create_action_expert_config_from_vlm(vlm_ap_config, small_diff_model)
    MODEL_DICT.add_model(small_diff_config)

    results = []

    for system in systems:
        logger.info(f"\n  System: {system}")

        df_vision = get_qwen2vla_vision_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_prefill = get_qwen2vla_vlm_prefill_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)

        if df_vision.empty or df_prefill.empty:
            logger.warning(f"    Skipped {system}")
            continue

        vision_ms = df_vision[df_vision["batch_size"] == 1]["time_ms"].values[0]
        prefill_ms = df_prefill[df_prefill["batch_size"] == 1]["time_ms"].values[0]

        # Setup A: Autoregressive (actual Qwen2-VL VLA)
        logger.info("    Setup A: Autoregressive (actual)")
        for chunk_size in action_chunk_sizes:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system, num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens=config.action_dof * chunk_size,
                bits=bits, max_batch_size=1,
            )
            if decode_results:
                row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_ms = row["time_ms"] * config.action_dof * chunk_size
                e2e_ms = vision_ms + prefill_ms + action_ms
                results.append({
                    "system": system, "setup": "A: Autoregressive",
                    "denoising_steps": "N/A", "action_chunk_size": chunk_size,
                    "dof": config.action_dof, "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                    "action_ms": action_ms, "action_oi": row.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        for dof in dof_values:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system, num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens=dof, bits=bits, max_batch_size=1,
            )
            if decode_results:
                row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_ms = row["time_ms"] * dof
                e2e_ms = vision_ms + prefill_ms + action_ms
                results.append({
                    "system": system, "setup": "A: Autoregressive (DoF Comparison)",
                    "denoising_steps": "N/A", "action_chunk_size": 1, "dof": dof,
                    "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                    "action_ms": action_ms, "action_oi": row.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        # Setup B: Small Diffusion (hypothetical derived action expert, ~hidden//2)
        logger.info("    Setup B: Small Diffusion (hypothetical ~3.5B expert)")
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                action_results = collect_parallel_decode_perf(
                    model=small_diff_model, system=system, num_devices=num_devices,
                    input_tokens=config.total_vlm_tokens,
                    output_tokens_parallel=config.action_dof * chunk_size,
                    self_attention=True, bits=bits, max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = vision_ms + prefill_ms + action_ms
                    results.append({
                        "system": system, "setup": "B: Small Diffusion",
                        "denoising_steps": steps, "action_chunk_size": chunk_size,
                        "dof": config.action_dof, "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        for dof in dof_values:
            for steps in denoising_steps_range:
                action_results = collect_parallel_decode_perf(
                    model=small_diff_model, system=system, num_devices=num_devices,
                    input_tokens=config.total_vlm_tokens,
                    output_tokens_parallel=dof,
                    self_attention=True, bits=bits, max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = vision_ms + prefill_ms + action_ms
                    results.append({
                        "system": system, "setup": "B: Small Diffusion (DoF Comparison)",
                        "denoising_steps": steps, "action_chunk_size": 1, "dof": dof,
                        "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        # Setup C: Large Diffusion (VLM-sized DiT ~7B)
        logger.info("    Setup C: Large Diffusion (VLM-sized ~7B)")
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model, system=system, num_devices=num_devices,
                    input_tokens=config.total_vlm_tokens,
                    output_tokens_parallel=config.action_dof * chunk_size,
                    self_attention=True, bits=bits, max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = vision_ms + prefill_ms + action_ms
                    results.append({
                        "system": system, "setup": "C: Large Diffusion",
                        "denoising_steps": steps, "action_chunk_size": chunk_size,
                        "dof": config.action_dof, "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        for dof in dof_values:
            for steps in denoising_steps_range:
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model, system=system, num_devices=num_devices,
                    input_tokens=config.total_vlm_tokens,
                    output_tokens_parallel=dof,
                    self_attention=True, bits=bits, max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = vision_ms + prefill_ms + action_ms
                    results.append({
                        "system": system, "setup": "C: Large Diffusion (DoF Comparison)",
                        "denoising_steps": steps, "action_chunk_size": 1, "dof": dof,
                        "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        # Setup D: Autoregressive with Parallel Decoding
        logger.info("    Setup D: AR Parallel Decode")
        for chunk_size in action_chunk_sizes:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model, system=system, num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens_parallel=config.action_dof * chunk_size,
                self_attention=True, bits=bits, max_batch_size=1,
            )
            if action_results:
                ar = [r for r in action_results if r["batch_size"] == 1][0]
                action_ms = ar["time_ms"]
                e2e_ms = vision_ms + prefill_ms + action_ms
                results.append({
                    "system": system, "setup": "D: Autoregressive Parallel",
                    "denoising_steps": "N/A", "action_chunk_size": chunk_size,
                    "dof": config.action_dof, "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                    "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        for dof in dof_values:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model, system=system, num_devices=num_devices,
                input_tokens=config.total_vlm_tokens,
                output_tokens_parallel=dof,
                self_attention=True, bits=bits, max_batch_size=1,
            )
            if action_results:
                ar = [r for r in action_results if r["batch_size"] == 1][0]
                action_ms = ar["time_ms"]
                e2e_ms = vision_ms + prefill_ms + action_ms
                results.append({
                    "system": system, "setup": "D: Autoregressive Parallel (DoF Comparison)",
                    "denoising_steps": "N/A", "action_chunk_size": 1, "dof": dof,
                    "vision_ms": vision_ms, "vlm_ms": prefill_ms,
                    "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

    df = pd.DataFrame(results)
    if not df.empty:
        out = output_path / "qwen2vla_autoregressive_vs_diffusion.csv"
        df.to_csv(out, index=False)
        logger.info(f"\nAR vs Diffusion results saved to {out}")

        for system in df["system"].unique():
            sys_df = df[df["system"] == system]
            logger.info(f"\n  System: {system} — E2E latency (ms) vs action chunk (DoF={config.action_dof})")
            logger.info(f"  {'Solution':<30}" + "".join(f"Chunk={c:>3}" for c in action_chunk_sizes))
            logger.info("  " + "-" * (30 + 9 * len(action_chunk_sizes)))
            for setup_label, setup_name, steps in [
                ("Autoregressive (actual)", "A: Autoregressive", "N/A"),
                ("Diffusion (Small, hyp.)", "B: Small Diffusion", denoising_steps_range[0]),
                ("Diffusion (Large, hyp.)", "C: Large Diffusion", denoising_steps_range[0]),
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
    config: Qwen2VLAConfig = QWEN2VLA_CONFIG,
    denoising_steps: int = 1,
    bits: str = "bf16",
    image_resolution: int = 224,
    image_compression_ratio: float = 0.1,
    action_dof: int = QWEN2VL_ACTION_DOF,
    action_chunk_size: int = 1,
    logger=None,
) -> pd.DataFrame:
    """Exp 6: Compare on-device / edge-server / cloud inference for Qwen2-VL VLA."""
    if logger is None:
        logger = logging.getLogger(__name__)

    image_config = ImageConfig(resolution=image_resolution, channels=3,
                               bytes_per_pixel=1, compression_ratio=image_compression_ratio)
    action_config = ActionConfig(num_dof=action_dof, action_chunk_size=action_chunk_size,
                                 bytes_per_value=4)
    num_devices = 1
    results = []

    logger.info(f"\n  Image: {image_config.name}, Action: {action_config.name}")

    def get_e2e(system):
        df_v = get_qwen2vla_vision_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_p = get_qwen2vla_vlm_prefill_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_d = get_qwen2vla_action_decode_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        if df_v.empty or df_p.empty or df_d.empty:
            return None, None, None, None
        v = df_v["time_ms"].values[0]
        p = df_p["time_ms"].values[0]
        d = df_d["time_ms"].values[0] * config.action_tokens
        return v, p, d, v + p + d

    for system in ["Jetson_AGX_Thor", "RTX_4090", "B100"]:
        try:
            v_ms, p_ms, d_ms, e2e = get_e2e(system)
            if e2e is None:
                continue
            results.append({
                "model": config.name, "category": "On-device", "system": system,
                "network": "N/A (Local)", "precision": bits,
                "vision_ms": v_ms, "vlm_ms": p_ms, "action_ms": d_ms,
                "network_image_ms": 0.0, "network_action_ms": 0.0,
                "e2e_compute_ms": e2e, "e2e_total_ms": e2e,
                "frequency_hz": 1000 / e2e, "freq_async_hz": 1000 / e2e,
                "denoising_steps": "AR",
            })
        except Exception as exc:
            logger.warning(f"    {system} on-device: {str(exc)[:50]}")

    edge_networks = [ETHERNET_1G_CONFIG, ETHERNET_10G_CONFIG, WIFI_6_CONFIG, WIFI_7_CONFIG,
                     CELL_5G_SA_CONFIG, CELL_4G_LTE_CONFIG]

    for system in ["RTX_4090", "B100"]:
        try:
            v_ms, p_ms, d_ms, e2e_compute = get_e2e(system)
            if e2e_compute is None:
                continue
            inf_hz = 1000.0 / e2e_compute
            for net in edge_networks:
                img_lat = estimate_image_latency(net, image_config)["total_latency_ms"]
                act_lat = estimate_action_latency(net, action_config)["total_latency_ms"]
                e2e_total = e2e_compute + img_lat + act_lat
                net_hz = compute_network_throughput_hz(net, image_config, action_config)
                results.append({
                    "model": config.name, "category": "Edge-server", "system": system,
                    "network": net.name, "precision": bits,
                    "vision_ms": v_ms, "vlm_ms": p_ms, "action_ms": d_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat,
                    "e2e_compute_ms": e2e_compute, "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total, "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": "AR",
                })
        except Exception as exc:
            logger.warning(f"    {system} edge-server: {str(exc)[:50]}")

    try:
        system = "B100"
        v_ms, p_ms, d_ms, e2e_compute = get_e2e(system)
        if e2e_compute is not None:
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
                    "vision_ms": v_ms, "vlm_ms": p_ms, "action_ms": d_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat,
                    "e2e_compute_ms": e2e_compute, "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total, "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": "AR",
                })
    except Exception as exc:
        logger.warning(f"    Cloud B100: {str(exc)[:50]}")

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
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}QWEN2-VL VLA DEVICE VS SERVER COMPARISON")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_results = {}
    all_dfs = []

    for config in ALL_QWEN2VLA_CONFIGS:
        logger.info(f"\n  {config.name}:")
        df = compare_device_vs_server(config=config, logger=logger)
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        print_device_vs_server_summary(df, logger=logger)

    if all_dfs:
        out = output_path / "qwen2vla_device_vs_server.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(out, index=False)
        logger.info(f"\nResults saved to {out}")

    return all_results


# ==============================================================================
# Experiment 7: Device-Server Collaboration
# VLM prefill on server -> KV cache download -> AR decode on device
# ==============================================================================

def compare_device_server_collaboration(
    config: Qwen2VLAConfig = QWEN2VLA_CONFIG,
    bits: str = "bf16",
    image_resolution: int = 224,
    image_compression_ratio: float = 0.1,
    action_dof: int = QWEN2VL_ACTION_DOF,
    action_chunk_size: int = 1,
    server_system: str = "B100",
    device_system: str = "Jetson_AGX_Thor",
    network_configs: list = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 7: Device-Server Collaboration for Qwen2-VL VLA.

    Note: For pure-AR VLA (same model weights for prefill + decode), collab means:
      - Server: VLM prefill (context encoding)
      - Network: KV cache download (server -> device)
      - Device: AR decode of action tokens

    This shows the overhead of KV cache transfer vs server-only inference.
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

    def get_latencies(system):
        df_v = get_qwen2vla_vision_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_p = get_qwen2vla_vlm_prefill_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_d = get_qwen2vla_action_decode_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        if df_v.empty or df_p.empty or df_d.empty:
            return None
        v = df_v["time_ms"].values[0]
        p = df_p["time_ms"].values[0]
        d = df_d["time_ms"].values[0] * config.action_tokens
        return v, p, d

    # Scenario 1: Device Only
    try:
        latencies = get_latencies(device_system)
        if latencies:
            v_ms, p_ms, d_ms = latencies
            e2e = v_ms + p_ms + d_ms
            results.append({
                "model": config.name, "category": "Device Only",
                "server_system": "N/A", "device_system": device_system,
                "network": "N/A (Local)", "precision": bits,
                "vision_ms": v_ms, "vlm_ms": p_ms, "action_ms": d_ms,
                "network_image_ms": 0.0, "network_action_ms": 0.0, "network_kv_cache_ms": 0.0,
                "e2e_total_ms": e2e, "frequency_hz": 1000 / e2e, "freq_async_hz": 1000 / e2e,
            })
    except Exception as e:
        logger.warning(f"    Device Only: {str(e)[:50]}")

    # Scenario 2: Server Only
    try:
        latencies = get_latencies(server_system)
        if latencies:
            v_ms, p_ms, d_ms = latencies
            e2e_compute = v_ms + p_ms + d_ms
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
                    "vision_ms": v_ms, "vlm_ms": p_ms, "action_ms": d_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat, "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz),
                })
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
                    "vision_ms": v_ms, "vlm_ms": p_ms, "action_ms": d_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat, "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz),
                })
    except Exception as e:
        logger.warning(f"    Server Only: {str(e)[:50]}")

    # Scenario 3: Device-Server Collaboration (VLM prefill on server, AR decode on device)
    try:
        df_p_server = get_qwen2vla_vlm_prefill_perf(
            config, [server_system], [num_devices], bits, max_batch_size=1, logger=logger
        )
        df_d_device = get_qwen2vla_action_decode_perf(
            config, [device_system], [num_devices], bits, max_batch_size=1, logger=logger
        )
        if not (df_p_server.empty or df_d_device.empty):
            p_ms = df_p_server["time_ms"].values[0]
            d_ms = df_d_device["time_ms"].values[0] * config.action_tokens
            server_hz = 1000.0 / p_ms
            device_hz = 1000.0 / d_ms if d_ms > 0 else float("inf")
            for net in network_configs:
                img_lat = estimate_image_latency(net, image_config)["total_latency_ms"]
                kv_lat = estimate_kvcache_latency(net, kv_cache_config)["total_latency_ms"]
                e2e_total = img_lat + p_ms + kv_lat + d_ms
                net_hz = compute_network_throughput_hz(net, image_config, kvcache_config=kv_cache_config)
                results.append({
                    "model": config.name, "category": "Device-Server Collaboration",
                    "server_system": server_system, "device_system": device_system,
                    "network": net.name, "precision": bits,
                    "vision_ms": 0.0, "vlm_ms": p_ms, "action_ms": d_ms,
                    "network_image_ms": img_lat, "network_action_ms": 0.0, "network_kv_cache_ms": kv_lat,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(server_hz, net_hz, device_hz),
                })
            for net_name, local_net, cloud_net in [
                ("Wired + Fast Cloud", ETHERNET_10G_CONFIG, CLOUD_FAST_CONFIG),
                ("4G + Slow Cloud", CELL_4G_LTE_CONFIG, CLOUD_SLOW_CONFIG),
            ]:
                img_lat = (estimate_image_latency(local_net, image_config)["total_latency_ms"] +
                           estimate_image_latency(cloud_net, image_config)["total_latency_ms"])
                kv_lat = (estimate_kvcache_latency(local_net, kv_cache_config)["total_latency_ms"] +
                          estimate_kvcache_latency(cloud_net, kv_cache_config)["total_latency_ms"])
                e2e_total = img_lat + p_ms + kv_lat + d_ms
                net_hz = min(
                    compute_network_throughput_hz(local_net, image_config, kvcache_config=kv_cache_config),
                    compute_network_throughput_hz(cloud_net, image_config, kvcache_config=kv_cache_config),
                )
                results.append({
                    "model": config.name, "category": "Device-Server Collaboration",
                    "server_system": server_system, "device_system": device_system,
                    "network": net_name, "precision": bits,
                    "vision_ms": 0.0, "vlm_ms": p_ms, "action_ms": d_ms,
                    "network_image_ms": img_lat, "network_action_ms": 0.0, "network_kv_cache_ms": kv_lat,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(server_hz, net_hz, device_hz),
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
                    f"({1000/best['e2e_total_ms']:.1f} Hz sync)")


def run_device_server_collaboration_comparison(
    output_dir: str = "perf_results",
    network_configs: list = None,
    server_system: str = "B100",
    device_system: str = "Jetson_AGX_Thor",
    experiment_num: int = None,
    logger=None,
) -> dict:
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}QWEN2-VL VLA DEVICE-SERVER COLLABORATION")
    logger.info("=" * 130)
    logger.info(f"Server: {server_system}, Device: {device_system}")
    logger.info("Note: For pure-AR VLA, collaboration means VLM prefill on server + AR decode on device.")
    logger.info("      KV cache download overhead typically dominates action download for these token counts.")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_results = {}
    all_dfs = []

    for config in ALL_QWEN2VLA_CONFIGS:
        logger.info(f"\n  {config.name}:")
        df = compare_device_server_collaboration(
            config=config, server_system=server_system, device_system=device_system,
            network_configs=network_configs, logger=logger,
        )
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        print_device_server_collaboration_summary(df, logger=logger)

    if all_dfs:
        out = output_path / "qwen2vla_device_server_collaboration.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(out, index=False)
        logger.info(f"\nResults saved to {out}")

    return all_results


# ==============================================================================
# Summary
# ==============================================================================

def print_summary(results: dict[str, pd.DataFrame], logger=None) -> None:
    """Print a summary table of Qwen2-VL VLA E2E performance."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\nQwen2-VL-7B VLA model characteristics:")
    logger.info(f"  Vision:  Qwen2-VL ViT ({QWEN2VL_RAW_VISION_TOKENS} raw -> {QWEN2VL_MERGED_VISION_TOKENS} merged tokens)")
    logger.info(f"  VLM:     Qwen2-7B ({QWEN2VL_TOTAL_VLM_TOKENS} prefill tokens, 28 layers, 3584 hidden, GQA 28/4)")
    logger.info(f"  Action:  {QWEN2VL_ACTION_TOKENS} autoregressive tokens (7-DoF)")

    logger.info("\n" + "=" * 130)
    logger.info("Qwen2-VL-7B VLA Performance Summary")
    logger.info("=" * 130)

    if "e2e" in results and not results["e2e"].empty:
        df = results["e2e"]
        logger.info("-" * 130)
        logger.info(
            f"{'Hardware':<18} {'Chips':<6} {'Batch':<6} "
            f"{'Vision (ms)':>14} {'Prefill (ms)':>14} {'Decode (ms)':>14} "
            f"{'E2E (ms)':>12} {'Hz':>10}"
        )
        logger.info("-" * 130)
        for _, row in df.iterrows():
            e2e = row["e2e_time_ms"]
            hz = 1000 / e2e if e2e > 0 else 0
            logger.info(
                f"{row['hardware.name']:<18} {int(row['hardware.num_chips']):<6} "
                f"{int(row['batch_size']):<6} "
                f"{row['vision_time_ms']:>14.2f} {row['prefill_time_ms']:>14.2f} "
                f"{row['decode_time_ms']:>14.2f} {e2e:>12.2f} {hz:>10.1f}"
            )
        logger.info("-" * 130)
    else:
        logger.warning("No E2E results available.")


if __name__ == "__main__":
    logger = setup_logging("perf_results/qwen2vla_perf.log")

    system_list = ["A100_80GB", "H100", "B100", "RTX_3090", "RTX_4090", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"

    runall = True
    run_exp_1 = True
    run_exp_2 = True
    run_exp_3 = True
    run_exp_4 = True
    run_exp_5 = True
    run_exp_6 = True
    run_exp_7 = True

    logger.info("=" * 100)
    logger.info("Starting Qwen2-VL-7B VLA Performance Evaluation (7 Experiments)")
    logger.info("=" * 100)
    logger.info(f"Systems: {system_list}")
    logger.info(f"Devices: {num_device_list}")
    logger.info(f"Precision: {bits}")

    exp_counter = 0

    if runall or run_exp_1:
        exp_counter += 1
        all_results = get_all_qwen2vla_perf(
            system_list=system_list,
            num_device_list=num_device_list,
            bits=bits,
            experiment_num=exp_counter,
            logger=logger,
        )
        print_summary(all_results.get("qwen2-vl-7b-vla", {}), logger=logger)

    if runall or run_exp_2:
        exp_counter += 1
        size_results, size_configs = get_model_size_scaling_perf(
            system_list=["B100", "RTX_4090", "Jetson_AGX_Thor"],
            num_device_list=[1],
            bits=bits,
            experiment_num=exp_counter,
            logger=logger,
        )
        print_model_size_scaling_summary(size_results, size_configs, logger=logger)

    if runall or run_exp_3:
        exp_counter += 1
        run_long_context_experiment(experiment_num=exp_counter, logger=logger)

    if runall or run_exp_4:
        exp_counter += 1
        compare_action_tokens_context_lengths(experiment_num=exp_counter, logger=logger)

    if runall or run_exp_5:
        exp_counter += 1
        compare_autoregressive_vs_diffusion(experiment_num=exp_counter, logger=logger)

    if runall or run_exp_6:
        exp_counter += 1
        run_device_vs_server_comparison(experiment_num=exp_counter, logger=logger)

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
