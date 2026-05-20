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
X-VLA Performance Evaluation Script

X-VLA (~0.9B total) from arxiv:2510.10274:
- VLM:     Florence-2-Large (DaViT vision encoder + BART text encoder, 0.77B)
           Modeled as a single encoder-only prefill over fused vision + language tokens.
- Policy:  SoftPromptedTransformer (24 layers, 1024 hidden)
           Processes Florence-2 outputs with learnable embodiment soft prompts.
- Action:  Flow Matching denoiser (30 steps default; folded into policy transformer
           as parallel decode — the denoiser shares policy transformer weights).

Performance modeling breakdown:
1. Florence-2 encoding (joint vision + text prefill)
2. Policy transformer prefill (soft prompts + Florence-2 tokens)
3. Flow matching parallel decode (N denoising steps over action tokens)

Experiments (7 total, mirroring pi0_perf.py depth):
  1. Base E2E performance across hardware
  2. Policy transformer size scaling
  3. Long context (multi-camera Florence-2 frames)
  4. Denoising steps x action dimension sweep
  5. Autoregressive vs Diffusion comparison
  6. Device vs Server (on-device / edge-server / cloud + network latency)
  7. Device-Server Collaboration (policy prefill on server, flow matching on device)

Reference:
    arxiv.org/abs/2510.10274
    github.com/2toinf/X-VLA
    genz/GenZ/Models/Model_sets/vla_models.py -> X-VLA model configs
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
# X-VLA Architecture Constants
# ==============================================================================
XVLA_FLORENCE2_TOKENS = 576        # fused vision + text tokens from Florence-2-Large (768px input)
XVLA_SOFT_PROMPT_TOKENS = 16       # learnable embodiment soft prompt tokens
XVLA_POLICY_CONTEXT_TOKENS = XVLA_FLORENCE2_TOKENS + XVLA_SOFT_PROMPT_TOKENS
XVLA_ACTION_DIM = 20               # ee6d action space: 20D
XVLA_ACTION_TOKENS = XVLA_ACTION_DIM
XVLA_DENOISING_STEPS = 30         # default flow matching steps (paper uses 30)
XVLA_IMAGE_RESOLUTION = 768       # Florence-2 input resolution


@dataclass
class XVLAConfig:
    """Configuration for an X-VLA-family model variant."""
    name: str
    encoder_model: str   # Florence-2 encoder
    policy_model: str    # SoftPromptedTransformer (also runs as flow matching denoiser)
    florence2_tokens_per_frame: int = XVLA_FLORENCE2_TOKENS
    soft_prompt_tokens: int = XVLA_SOFT_PROMPT_TOKENS
    num_frames: int = 1
    action_tokens: int = XVLA_ACTION_TOKENS
    denoising_steps: int = XVLA_DENOISING_STEPS
    action_dof: int = XVLA_ACTION_DIM

    @property
    def total_florence2_tokens(self) -> int:
        return self.florence2_tokens_per_frame * self.num_frames

    @property
    def total_policy_context_tokens(self) -> int:
        return self.total_florence2_tokens + self.soft_prompt_tokens


XVLA_CONFIG = XVLAConfig(
    name="xvla-0.9b",
    encoder_model="florence2-large-encoder",
    policy_model="xvla-policy",
)

ALL_XVLA_CONFIGS = [XVLA_CONFIG]


def create_action_expert_config_from_policy(policy_config: ModelConfig, name: str) -> ModelConfig:
    """Derive a smaller action expert from policy config (hidden//2, ffn//4)."""
    action_config = copy.deepcopy(policy_config)
    action_config.model = name
    action_config.hidden_size = policy_config.hidden_size // 2
    action_config.intermediate_size = policy_config.intermediate_size // 4
    action_config.vocab_size = 0
    return action_config


# ==============================================================================
# Component perf functions
# ==============================================================================

def get_xvla_florence2_perf(
    config: XVLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """Evaluate Florence-2-Large joint vision + text encoding."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model=config.encoder_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_florence2_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_xvla_policy_prefill_perf(
    config: XVLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """Evaluate X-VLA SoftPromptedTransformer policy prefill."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model=config.policy_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_policy_context_tokens,
                bits=bits,
                max_batch_size=max_batch_size,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_xvla_action_perf(
    config: XVLAConfig,
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    denoising_steps: int = None,
    action_tokens: int = None,
    max_batch_size: int = 1024,
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate X-VLA flow matching action denoiser.

    Each denoising step is a parallel decode pass over action_tokens tokens
    using the policy transformer (shared weights). Total = steps * single-step latency.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if denoising_steps is None:
        denoising_steps = config.denoising_steps
    if action_tokens is None:
        action_tokens = config.action_tokens

    results = []
    for system in system_list:
        for num_devices in num_device_list:
            model_results = collect_parallel_decode_perf(
                model=config.policy_model,
                system=system,
                num_devices=num_devices,
                input_tokens=config.total_policy_context_tokens,
                output_tokens_parallel=action_tokens,
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

def get_xvla_e2e_perf(
    config: XVLAConfig = XVLA_CONFIG,
    system_list: list[str] = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    denoising_steps: int = None,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Evaluate end-to-end X-VLA performance across all components."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)
    if denoising_steps is None:
        denoising_steps = config.denoising_steps

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    df_f2 = get_xvla_florence2_perf(config, system_list, num_device_list, bits, logger=logger)
    results["florence2"] = df_f2
    df_f2.to_csv(output_path / f"{config.name}_florence2_perf.csv", index=False)

    df_policy = get_xvla_policy_prefill_perf(config, system_list, num_device_list, bits, logger=logger)
    results["policy_prefill"] = df_policy
    df_policy.to_csv(output_path / f"{config.name}_policy_prefill_perf.csv", index=False)

    df_action = get_xvla_action_perf(
        config, system_list, num_device_list, bits, denoising_steps=denoising_steps, logger=logger
    )
    results["action"] = df_action
    df_action.to_csv(output_path / f"{config.name}_action_perf.csv", index=False)

    if df_f2.empty or df_policy.empty or df_action.empty:
        logger.warning(f"One or more component DataFrames empty for {config.name} — skipping E2E.")
        results["e2e"] = pd.DataFrame()
        return results

    group_cols = ["hardware.name", "hardware.num_chips", "batch_size"]
    f2_times = df_f2[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "florence2_time_ms"})
    policy_times = df_policy[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "policy_time_ms"})
    action_times = df_action[group_cols + ["time_ms"]].copy().rename(columns={"time_ms": "action_time_ms"})

    df_merged = f2_times.merge(policy_times, on=group_cols, how="inner")
    df_merged = df_merged.merge(action_times, on=group_cols, how="inner")
    df_merged["e2e_time_ms"] = (df_merged["florence2_time_ms"] + df_merged["policy_time_ms"] +
                                df_merged["action_time_ms"])
    df_merged["model.name"] = config.name
    df_merged["model.stage"] = "e2e"
    df_merged["model.dec_steps"] = denoising_steps
    df_merged["model.seq_len_inference_prefill"] = config.total_policy_context_tokens

    df_e2e = df_merged[[
        "model.name", "model.stage", "model.dec_steps",
        "model.seq_len_inference_prefill",
        "hardware.name", "hardware.num_chips", "batch_size",
        "florence2_time_ms", "policy_time_ms", "action_time_ms", "e2e_time_ms",
    ]]
    results["e2e"] = df_e2e
    df_e2e.to_csv(output_path / f"{config.name}_e2e_perf.csv", index=False)

    return results


def get_all_xvla_perf(
    system_list: list[str] = ["A100_80GB", "H100", "B100", "RTX_3090", "RTX_4090", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    denoising_steps: int = XVLA_DENOISING_STEPS,
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run Exp 1: base E2E for all X-VLA configs."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}X-VLA BASE PERFORMANCE EVALUATION")
    logger.info("=" * 130)

    all_results = {}
    for config in ALL_XVLA_CONFIGS:
        logger.info(f"\nEvaluating {config.name}...")
        results = get_xvla_e2e_perf(config, system_list, num_device_list, denoising_steps, bits, output_dir, logger)
        all_results[config.name] = results

    all_e2e = [r["e2e"] for r in all_results.values() if not r["e2e"].empty]
    if all_e2e:
        pd.concat(all_e2e, ignore_index=True).to_csv(
            Path(output_dir) / "xvla_family_e2e_perf.csv", index=False
        )

    return all_results


# ==============================================================================
# Experiment 2: Model Size Scaling (Policy Transformer)
# ==============================================================================

def get_model_size_scaling_perf(
    system_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    denoising_steps: int = XVLA_DENOISING_STEPS,
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> tuple[dict, dict]:
    """
    Exp 2: Evaluate X-VLA scaling with different policy transformer sizes.

    Variants (Florence-2 encoder is fixed, policy transformer scales):
      xvla-1b:   florence2-large + xvla-policy (actual, 24L/1024H)
      xvla-7b:   florence2-large + llama2_7b proxy policy
      xvla-13b:  florence2-large + llama2_13b proxy policy
      xvla-70b:  florence2-large + llama2_70b proxy policy
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_device_list is None:
        num_device_list = [1]

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}X-VLA POLICY TRANSFORMER SIZE SCALING EVALUATION")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model_variants = []

    # Baseline: actual X-VLA policy
    policy_1b = copy.deepcopy(MODEL_DICT.get_model("xvla-policy"))
    policy_1b.vocab_size = 0
    policy_1b.model = "xvla-policy-no-vocab"
    MODEL_DICT.add_model(policy_1b)
    model_variants.append(XVLAConfig(
        name="xvla-1b",
        encoder_model="florence2-large-encoder",
        policy_model=policy_1b.model,
        denoising_steps=denoising_steps,
    ))

    # 7B variant
    policy_7b = copy.deepcopy(MODEL_DICT.get_model("llama2_7b"))
    policy_7b.vocab_size = 0
    policy_7b.model = "xvla-policy-7b-proxy"
    MODEL_DICT.add_model(policy_7b)
    model_variants.append(XVLAConfig(
        name="xvla-7b",
        encoder_model="florence2-large-encoder",
        policy_model=policy_7b.model,
        denoising_steps=denoising_steps,
    ))

    # 13B variant
    policy_13b = copy.deepcopy(MODEL_DICT.get_model("llama2_13b"))
    policy_13b.vocab_size = 0
    policy_13b.model = "xvla-policy-13b-proxy"
    MODEL_DICT.add_model(policy_13b)
    model_variants.append(XVLAConfig(
        name="xvla-13b",
        encoder_model="florence2-large-encoder",
        policy_model=policy_13b.model,
        denoising_steps=denoising_steps,
    ))

    # 70B variant
    policy_70b = copy.deepcopy(MODEL_DICT.get_model("llama2_70b"))
    policy_70b.vocab_size = 0
    policy_70b.model = "xvla-policy-70b-proxy"
    MODEL_DICT.add_model(policy_70b)
    model_variants.append(XVLAConfig(
        name="xvla-70b",
        encoder_model="florence2-large-encoder",
        policy_model=policy_70b.model,
        denoising_steps=denoising_steps,
    ))

    logger.info("\n--- Model Parameter Counts ---")
    component_params_data = []
    for cfg in model_variants:
        enc_cfg = MODEL_DICT.get_model(cfg.encoder_model)
        pol_cfg = MODEL_DICT.get_model(cfg.policy_model)
        ep = calculate_transformer_params(enc_cfg)
        pp = calculate_transformer_params(pol_cfg)
        total = ep + pp
        logger.info(f"  {cfg.name}: encoder={format_param_count(ep)}, policy={format_param_count(pp)}, "
                    f"total={format_param_count(total)}")
        component_params_data.append({
            "model": cfg.name, "encoder_params_M": ep / 1e6,
            "policy_params_M": pp / 1e6, "total_params_M": total / 1e6,
        })
    pd.DataFrame(component_params_data).to_csv(
        output_path / "xvla_model_size_scaling_params.csv", index=False
    )

    all_results = {}
    all_e2e = []
    for cfg in model_variants:
        logger.info(f"\n  Evaluating {cfg.name}...")
        results = get_xvla_e2e_perf(cfg, system_list, num_device_list, denoising_steps, bits, output_dir, logger)
        all_results[cfg.name] = results
        if not results["e2e"].empty:
            all_e2e.append(results["e2e"])

    if all_e2e:
        pd.concat(all_e2e, ignore_index=True).to_csv(
            output_path / "xvla_model_size_scaling.csv", index=False
        )

    return all_results, {cfg.name: cfg for cfg in model_variants}


def print_model_size_scaling_summary(all_results: dict, model_configs: dict, logger=None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("\n--- X-VLA Policy Size Scaling Summary (Batch=1, Chips=1) ---")
    logger.info(f"{'Model':<18} {'Hardware':<20} {'F2 Enc (ms)':>12} {'Policy (ms)':>12} {'Action (ms)':>12} {'E2E (ms)':>12} {'Hz':>8}")
    logger.info("-" * 100)
    for model_name, results in all_results.items():
        if "e2e" not in results or results["e2e"].empty:
            continue
        df = results["e2e"]
        sub = df[(df["hardware.num_chips"] == 1) & (df["batch_size"] == 1)]
        for _, row in sub.iterrows():
            e2e = row["e2e_time_ms"]
            logger.info(f"{model_name:<18} {row['hardware.name']:<20} "
                        f"{row['florence2_time_ms']:>12.2f} {row['policy_time_ms']:>12.2f} "
                        f"{row['action_time_ms']:>12.2f} {e2e:>12.2f} {1000/e2e:>8.1f}")


# ==============================================================================
# Experiment 3: Long Context (multi-camera / multi-frame)
# ==============================================================================

def run_long_context_experiment(
    config: XVLAConfig = XVLA_CONFIG,
    system_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_frames_list: list[int] = [1, 2, 3, 4, 5],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 3: Evaluate X-VLA latency as number of Florence-2 frames scales.

    Each additional frame adds 576 Florence-2 tokens to the encoder prefill
    and expands the policy context by 576 tokens accordingly.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}X-VLA LONG CONTEXT (MULTI-CAMERA) EXPERIMENT")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = []
    for num_frames in num_frames_list:
        cfg = XVLAConfig(
            name=config.name,
            encoder_model=config.encoder_model,
            policy_model=config.policy_model,
            num_frames=num_frames,
            denoising_steps=config.denoising_steps,
        )
        total_f2 = cfg.total_florence2_tokens
        total_pol = cfg.total_policy_context_tokens
        kv_mb = calculate_kv_cache_size_mb(config.policy_model, total_pol, bits)
        logger.info(f"\n  Frames={num_frames}: florence2_tokens={total_f2}, "
                    f"policy_context={total_pol}, KV cache={kv_mb:.1f} MB")

        for system in system_list:
            df_f2 = get_xvla_florence2_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)
            df_pol = get_xvla_policy_prefill_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)
            df_act = get_xvla_action_perf(cfg, [system], [1], bits, max_batch_size=1, logger=logger)

            if df_f2.empty or df_pol.empty or df_act.empty:
                continue

            f2_ms = df_f2["time_ms"].values[0]
            pol_ms = df_pol["time_ms"].values[0]
            act_ms = df_act["time_ms"].values[0]
            e2e_ms = f2_ms + pol_ms + act_ms

            results.append({
                "model": config.name, "system": system, "num_frames": num_frames,
                "total_florence2_tokens": total_f2, "total_policy_context": total_pol,
                "kv_cache_mb": kv_mb,
                "florence2_ms": f2_ms, "policy_ms": pol_ms, "action_ms": act_ms,
                "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        out = output_path / "xvla_long_context.csv"
        df.to_csv(out, index=False)
        logger.info(f"\nLong context results saved to {out}")
        logger.info("\n--- Long Context Summary ---")
        logger.info(f"{'Frames':>7} {'F2 Tok':>8} {'Pol Tok':>8} {'KV MB':>7} "
                    f"{'System':<20} {'F2 Enc':>10} {'Policy':>10} {'Action':>10} {'E2E':>10} {'Hz':>8}")
        logger.info("-" * 110)
        for _, row in df.iterrows():
            logger.info(f"{int(row['num_frames']):>7} {int(row['total_florence2_tokens']):>8} "
                        f"{int(row['total_policy_context']):>8} {row['kv_cache_mb']:>7.1f} "
                        f"{row['system']:<20} {row['florence2_ms']:>10.2f} {row['policy_ms']:>10.2f} "
                        f"{row['action_ms']:>10.2f} {row['e2e_ms']:>10.2f} {row['frequency_hz']:>8.1f}")

    return df


# ==============================================================================
# Experiment 4: Denoising Steps x Action Dimension sweep
# ==============================================================================

def compare_denoising_steps_action_dimensions(
    config: XVLAConfig = XVLA_CONFIG,
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    step_range: list[int] = [1, 5, 10, 20, 50],
    action_dim_range: list[int] = [5, 10, 20, 40, 80],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 4: 2D sweep over denoising steps and action dimensionality.

    For X-VLA, action_tokens = action_dim (one token per DoF). Varying action_dim
    shows the effect of different robot embodiments (6-DoF arm vs 20-DoF humanoid).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}X-VLA DENOISING STEPS x ACTION DIMENSION SWEEP")
    logger.info("=" * 130)
    logger.info(f"Systems: {systems}, Steps: {step_range}, Action Dims: {action_dim_range}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = []
    for system in systems:
        logger.info(f"\n  System: {system}")
        df_f2 = get_xvla_florence2_perf(config, [system], [1], bits, max_batch_size=1, logger=logger)
        df_pol = get_xvla_policy_prefill_perf(config, [system], [1], bits, max_batch_size=1, logger=logger)

        if df_f2.empty or df_pol.empty:
            logger.warning(f"    Skipped {system}")
            continue

        f2_ms = df_f2[df_f2["batch_size"] == 1]["time_ms"].values[0]
        pol_ms = df_pol[df_pol["batch_size"] == 1]["time_ms"].values[0]

        for steps in step_range:
            for action_dim in action_dim_range:
                df_act = get_xvla_action_perf(
                    config, [system], [1], bits,
                    denoising_steps=steps, action_tokens=action_dim,
                    max_batch_size=1, logger=logger,
                )
                if df_act.empty:
                    continue
                act_ms = df_act[df_act["batch_size"] == 1]["time_ms"].values[0]
                e2e_ms = f2_ms + pol_ms + act_ms
                results.append({
                    "model": config.name, "system": system,
                    "denoising_steps": steps, "action_dim": action_dim,
                    "florence2_ms": f2_ms, "policy_ms": pol_ms, "action_ms": act_ms,
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    out = output_path / "xvla_denoising_steps_action_dims.csv"
    df.to_csv(out, index=False)
    logger.info(f"\nDenoising steps x action dim results saved to {out}")

    default_steps = config.denoising_steps
    default_dim = config.action_tokens
    for system in df["system"].unique():
        sys_df = df[df["system"] == system]
        pivot = sys_df.pivot(index="denoising_steps", columns="action_dim", values="e2e_ms")
        logger.info(f"\n  System: {system} — E2E latency (ms) grid")
        col_label = "Steps \\ Dim"
        header = f"{col_label:<14}" + "".join(f"{d:>10}" for d in action_dim_range)
        logger.info(header)
        logger.info("-" * (14 + 10 * len(action_dim_range)))
        for steps in step_range:
            if steps in pivot.index:
                row_str = f"{steps:<14}" + "".join(
                    f"{pivot.loc[steps, d]:>10.2f}" if d in pivot.columns else f"{'N/A':>10}"
                    for d in action_dim_range
                )
                logger.info(row_str)

        baseline_row = sys_df[
            (sys_df["denoising_steps"] == default_steps) & (sys_df["action_dim"] == default_dim)
        ]
        if not baseline_row.empty:
            baseline_ms = baseline_row["e2e_ms"].values[0]
            logger.info(f"\n  Speedup vs baseline (steps={default_steps}, dim={default_dim})")
            logger.info(header)
            logger.info("-" * (14 + 10 * len(action_dim_range)))
            for steps in step_range:
                if steps in pivot.index:
                    row_str = f"{steps:<14}" + "".join(
                        f"{baseline_ms / pivot.loc[steps, d]:>10.2f}x" if d in pivot.columns else f"{'N/A':>10}"
                        for d in action_dim_range
                    )
                    logger.info(row_str)

    return df


# ==============================================================================
# Experiment 5: Autoregressive vs Diffusion
# ==============================================================================

def compare_autoregressive_vs_diffusion(
    config: XVLAConfig = XVLA_CONFIG,
    systems: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    num_devices: int = 1,
    bits: str = "bf16",
    denoising_steps_range: list[int] = [30],
    action_chunk_sizes: list[int] = [1, 5, 10, 20, 40],
    dof_values: list[int] = [7, 14, 20, 28, 35, 42],
    output_dir: str = "perf_results",
    experiment_num: int = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 5: Compare action generation strategies for X-VLA.

    A) Autoregressive (policy transformer as AR action predictor)
    B) Small Diffusion (actual X-VLA flow matching, policy weights shared)
    C) Large Diffusion (VLM-sized DiT proxy)
    D) Autoregressive with Parallel Decoding
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}X-VLA AUTOREGRESSIVE VS DIFFUSION")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    action_predictor_model = "xvla-policy-action-predictor"
    policy_ap = copy.deepcopy(MODEL_DICT.get_model(config.policy_model))
    policy_ap.model = action_predictor_model
    policy_ap.vocab_size = 0
    MODEL_DICT.add_model(policy_ap)

    results = []

    for system in systems:
        logger.info(f"\n  System: {system}")

        df_f2 = get_xvla_florence2_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_pol = get_xvla_policy_prefill_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)

        if df_f2.empty or df_pol.empty:
            logger.warning(f"    Skipped {system}")
            continue

        f2_ms = df_f2[df_f2["batch_size"] == 1]["time_ms"].values[0]
        pol_ms = df_pol[df_pol["batch_size"] == 1]["time_ms"].values[0]

        # Setup A: Autoregressive
        logger.info("    Setup A: Autoregressive")
        for chunk_size in action_chunk_sizes:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system, num_devices=num_devices,
                input_tokens=config.total_policy_context_tokens,
                output_tokens=config.action_dof * chunk_size,
                bits=bits, max_batch_size=1,
            )
            if decode_results:
                row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_ms = row["time_ms"] * config.action_dof * chunk_size
                e2e_ms = f2_ms + pol_ms + action_ms
                results.append({
                    "system": system, "setup": "A: Autoregressive",
                    "denoising_steps": "N/A", "action_chunk_size": chunk_size,
                    "dof": config.action_dof, "f2_ms": f2_ms, "policy_ms": pol_ms,
                    "action_ms": action_ms, "action_oi": row.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        for dof in dof_values:
            decode_results = collect_decode_perf(
                model=action_predictor_model,
                system=system, num_devices=num_devices,
                input_tokens=config.total_policy_context_tokens,
                output_tokens=dof, bits=bits, max_batch_size=1,
            )
            if decode_results:
                row = [r for r in decode_results if r["batch_size"] == 1][0]
                action_ms = row["time_ms"] * dof
                e2e_ms = f2_ms + pol_ms + action_ms
                results.append({
                    "system": system, "setup": "A: Autoregressive (DoF Comparison)",
                    "denoising_steps": "N/A", "action_chunk_size": 1, "dof": dof,
                    "f2_ms": f2_ms, "policy_ms": pol_ms,
                    "action_ms": action_ms, "action_oi": row.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        # Setup B: Small Diffusion (actual X-VLA flow matching)
        logger.info("    Setup B: Small Diffusion (actual X-VLA flow matching)")
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                df_act = get_xvla_action_perf(
                    config, [system], [num_devices], bits,
                    denoising_steps=steps, action_tokens=config.action_dof * chunk_size,
                    max_batch_size=1, logger=logger,
                )
                if not df_act.empty:
                    act_row = df_act[df_act["batch_size"] == 1].iloc[0]
                    action_ms = act_row["time_ms"]
                    e2e_ms = f2_ms + pol_ms + action_ms
                    results.append({
                        "system": system, "setup": "B: Small Diffusion",
                        "denoising_steps": steps, "action_chunk_size": chunk_size,
                        "dof": config.action_dof, "f2_ms": f2_ms, "policy_ms": pol_ms,
                        "action_ms": action_ms, "action_oi": act_row.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        for dof in dof_values:
            for steps in denoising_steps_range:
                df_act = get_xvla_action_perf(
                    config, [system], [num_devices], bits,
                    denoising_steps=steps, action_tokens=dof,
                    max_batch_size=1, logger=logger,
                )
                if not df_act.empty:
                    act_row = df_act[df_act["batch_size"] == 1].iloc[0]
                    action_ms = act_row["time_ms"]
                    e2e_ms = f2_ms + pol_ms + action_ms
                    results.append({
                        "system": system, "setup": "B: Small Diffusion (DoF Comparison)",
                        "denoising_steps": steps, "action_chunk_size": 1, "dof": dof,
                        "f2_ms": f2_ms, "policy_ms": pol_ms,
                        "action_ms": action_ms, "action_oi": act_row.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        # Setup C: Large Diffusion (VLM-sized DiT)
        logger.info("    Setup C: Large Diffusion (policy-sized DiT)")
        for steps in denoising_steps_range:
            for chunk_size in action_chunk_sizes:
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model, system=system, num_devices=num_devices,
                    input_tokens=config.total_policy_context_tokens,
                    output_tokens_parallel=config.action_dof * chunk_size,
                    self_attention=True, bits=bits, max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = f2_ms + pol_ms + action_ms
                    results.append({
                        "system": system, "setup": "C: Large Diffusion",
                        "denoising_steps": steps, "action_chunk_size": chunk_size,
                        "dof": config.action_dof, "f2_ms": f2_ms, "policy_ms": pol_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        for dof in dof_values:
            for steps in denoising_steps_range:
                action_results = collect_parallel_decode_perf(
                    model=action_predictor_model, system=system, num_devices=num_devices,
                    input_tokens=config.total_policy_context_tokens,
                    output_tokens_parallel=dof,
                    self_attention=True, bits=bits, max_batch_size=1,
                )
                if action_results:
                    ar = [r for r in action_results if r["batch_size"] == 1][0]
                    action_ms = ar["time_ms"] * steps
                    e2e_ms = f2_ms + pol_ms + action_ms
                    results.append({
                        "system": system, "setup": "C: Large Diffusion (DoF Comparison)",
                        "denoising_steps": steps, "action_chunk_size": 1, "dof": dof,
                        "f2_ms": f2_ms, "policy_ms": pol_ms,
                        "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                        "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                    })

        # Setup D: Autoregressive with Parallel Decoding
        logger.info("    Setup D: AR Parallel Decode")
        for chunk_size in action_chunk_sizes:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model, system=system, num_devices=num_devices,
                input_tokens=config.total_policy_context_tokens,
                output_tokens_parallel=config.action_dof * chunk_size,
                self_attention=True, bits=bits, max_batch_size=1,
            )
            if action_results:
                ar = [r for r in action_results if r["batch_size"] == 1][0]
                action_ms = ar["time_ms"]
                e2e_ms = f2_ms + pol_ms + action_ms
                results.append({
                    "system": system, "setup": "D: Autoregressive Parallel",
                    "denoising_steps": "N/A", "action_chunk_size": chunk_size,
                    "dof": config.action_dof, "f2_ms": f2_ms, "policy_ms": pol_ms,
                    "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

        for dof in dof_values:
            action_results = collect_parallel_decode_perf(
                model=action_predictor_model, system=system, num_devices=num_devices,
                input_tokens=config.total_policy_context_tokens,
                output_tokens_parallel=dof,
                self_attention=True, bits=bits, max_batch_size=1,
            )
            if action_results:
                ar = [r for r in action_results if r["batch_size"] == 1][0]
                action_ms = ar["time_ms"]
                e2e_ms = f2_ms + pol_ms + action_ms
                results.append({
                    "system": system, "setup": "D: Autoregressive Parallel (DoF Comparison)",
                    "denoising_steps": "N/A", "action_chunk_size": 1, "dof": dof,
                    "f2_ms": f2_ms, "policy_ms": pol_ms,
                    "action_ms": action_ms, "action_oi": ar.get("op_intensity", 0),
                    "e2e_ms": e2e_ms, "frequency_hz": 1000 / e2e_ms,
                })

    df = pd.DataFrame(results)
    if not df.empty:
        out = output_path / "xvla_autoregressive_vs_diffusion.csv"
        df.to_csv(out, index=False)
        logger.info(f"\nAR vs Diffusion results saved to {out}")

        for system in df["system"].unique():
            sys_df = df[df["system"] == system]
            logger.info(f"\n  System: {system} — E2E latency (ms) vs action chunk (DoF={config.action_dof})")
            logger.info(f"  {'Solution':<30}" + "".join(f"Chunk={c:>3}" for c in action_chunk_sizes))
            logger.info("  " + "-" * (30 + 9 * len(action_chunk_sizes)))
            for setup_label, setup_name, steps in [
                ("Autoregressive", "A: Autoregressive", "N/A"),
                ("Diffusion (actual)", "B: Small Diffusion", denoising_steps_range[0]),
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
    config: XVLAConfig = XVLA_CONFIG,
    denoising_steps: int = XVLA_DENOISING_STEPS,
    bits: str = "bf16",
    image_resolution: int = 768,
    image_compression_ratio: float = 0.1,
    action_dof: int = XVLA_ACTION_DIM,
    action_chunk_size: int = 1,
    logger=None,
) -> pd.DataFrame:
    """Exp 6: Compare on-device / edge-server / cloud inference for X-VLA."""
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
        df_f = get_xvla_florence2_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_p = get_xvla_policy_prefill_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_a = get_xvla_action_perf(config, [system], [num_devices], bits,
                                    denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
        if df_f.empty or df_p.empty or df_a.empty:
            return None, None, None, None
        f = df_f["time_ms"].values[0]
        p = df_p["time_ms"].values[0]
        a = df_a["time_ms"].values[0]
        return f, p, a, f + p + a

    for system in ["Jetson_AGX_Thor", "RTX_4090", "B100"]:
        try:
            f_ms, p_ms, a_ms, e2e = get_e2e(system)
            if e2e is None:
                continue
            results.append({
                "model": config.name, "category": "On-device", "system": system,
                "network": "N/A (Local)", "precision": bits,
                "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                "network_image_ms": 0.0, "network_action_ms": 0.0,
                "e2e_compute_ms": e2e, "e2e_total_ms": e2e,
                "frequency_hz": 1000 / e2e, "freq_async_hz": 1000 / e2e,
                "denoising_steps": denoising_steps,
            })
        except Exception as exc:
            logger.warning(f"    {system} on-device: {str(exc)[:50]}")

    edge_networks = [ETHERNET_1G_CONFIG, ETHERNET_10G_CONFIG, WIFI_6_CONFIG, WIFI_7_CONFIG,
                     CELL_5G_SA_CONFIG, CELL_4G_LTE_CONFIG]

    for system in ["RTX_4090", "B100"]:
        try:
            f_ms, p_ms, a_ms, e2e_compute = get_e2e(system)
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
                    "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat,
                    "e2e_compute_ms": e2e_compute, "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total, "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": denoising_steps,
                })
        except Exception as exc:
            logger.warning(f"    {system} edge-server: {str(exc)[:50]}")

    try:
        system = "B100"
        f_ms, p_ms, a_ms, e2e_compute = get_e2e(system)
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
                    "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat,
                    "e2e_compute_ms": e2e_compute, "e2e_total_ms": e2e_total,
                    "frequency_hz": 1000 / e2e_total, "freq_async_hz": min(inf_hz, net_hz),
                    "denoising_steps": denoising_steps,
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
    logger.info(f"{exp_header}X-VLA DEVICE VS SERVER COMPARISON")
    logger.info("=" * 130)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_results = {}
    all_dfs = []

    for config in ALL_XVLA_CONFIGS:
        logger.info(f"\n  {config.name}:")
        df = compare_device_vs_server(config=config, logger=logger)
        all_results[config.name] = df
        if not df.empty:
            all_dfs.append(df)
        print_device_vs_server_summary(df, logger=logger)

    if all_dfs:
        out = output_path / "xvla_device_vs_server.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(out, index=False)
        logger.info(f"\nResults saved to {out}")

    return all_results


# ==============================================================================
# Experiment 7: Device-Server Collaboration
# Policy prefill on server -> KV cache download -> flow matching denoising on device
# ==============================================================================

def compare_device_server_collaboration(
    config: XVLAConfig = XVLA_CONFIG,
    denoising_steps: int = XVLA_DENOISING_STEPS,
    bits: str = "bf16",
    image_resolution: int = 768,
    image_compression_ratio: float = 0.1,
    action_dof: int = XVLA_ACTION_DIM,
    action_chunk_size: int = 1,
    server_system: str = "B100",
    device_system: str = "Jetson_AGX_Thor",
    network_configs: list = None,
    logger=None,
) -> pd.DataFrame:
    """
    Exp 7: Device-Server Collaboration for X-VLA.

    Scenarios:
      1. Device Only:              Jetson runs all 3 stages locally
      2. Server Only:              Server runs all; sends action to robot
      3. Device-Server Collab:     Server runs Florence-2 + policy prefill;
                                   KV cache (policy context) sent to device;
                                   Device runs flow matching denoising
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
        model_name=config.policy_model,
        seq_lengths=[config.total_policy_context_tokens],
        pretty_name=f"{config.name}-Policy",
    )
    if not kv_cache_configs:
        logger.warning(f"Could not create KV cache config for {config.policy_model}")
        return pd.DataFrame()
    kv_cache_config = kv_cache_configs[0]

    results = []
    logger.info(f"\n  Image: {image_config.name}, KV Cache: {kv_cache_config.name}")
    logger.info(f"  Server: {server_system}, Device: {device_system}")

    def get_all_latencies(system):
        df_f = get_xvla_florence2_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_p = get_xvla_policy_prefill_perf(config, [system], [num_devices], bits, max_batch_size=1, logger=logger)
        df_a = get_xvla_action_perf(config, [system], [num_devices], bits,
                                    denoising_steps=denoising_steps, max_batch_size=1, logger=logger)
        if df_f.empty or df_p.empty or df_a.empty:
            return None
        return df_f["time_ms"].values[0], df_p["time_ms"].values[0], df_a["time_ms"].values[0]

    # Scenario 1: Device Only
    try:
        lat = get_all_latencies(device_system)
        if lat:
            f_ms, p_ms, a_ms = lat
            e2e = f_ms + p_ms + a_ms
            results.append({
                "model": config.name, "category": "Device Only",
                "server_system": "N/A", "device_system": device_system,
                "network": "N/A (Local)", "precision": bits,
                "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                "network_image_ms": 0.0, "network_action_ms": 0.0, "network_kv_cache_ms": 0.0,
                "e2e_total_ms": e2e, "frequency_hz": 1000 / e2e, "freq_async_hz": 1000 / e2e,
                "denoising_steps": denoising_steps,
            })
    except Exception as e:
        logger.warning(f"    Device Only: {str(e)[:50]}")

    # Scenario 2: Server Only
    try:
        lat = get_all_latencies(server_system)
        if lat:
            f_ms, p_ms, a_ms = lat
            e2e_compute = f_ms + p_ms + a_ms
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
                    "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat, "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz), "denoising_steps": denoising_steps,
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
                    "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": act_lat, "network_kv_cache_ms": 0.0,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
                    "freq_async_hz": min(inf_hz, net_hz), "denoising_steps": denoising_steps,
                })
    except Exception as e:
        logger.warning(f"    Server Only: {str(e)[:50]}")

    # Scenario 3: Device-Server Collaboration
    try:
        # Server: Florence-2 + policy prefill
        df_f_server = get_xvla_florence2_perf(
            config, [server_system], [num_devices], bits, max_batch_size=1, logger=logger
        )
        df_p_server = get_xvla_policy_prefill_perf(
            config, [server_system], [num_devices], bits, max_batch_size=1, logger=logger
        )
        # Device: flow matching denoising (using policy model on device)
        df_a_device = get_xvla_action_perf(
            config, [device_system], [num_devices], bits,
            denoising_steps=denoising_steps, max_batch_size=1, logger=logger
        )
        if not (df_f_server.empty or df_p_server.empty or df_a_device.empty):
            f_ms = df_f_server["time_ms"].values[0]
            p_ms = df_p_server["time_ms"].values[0]
            a_ms = df_a_device["time_ms"].values[0]
            server_compute_ms = f_ms + p_ms
            server_hz = 1000.0 / server_compute_ms
            device_hz = 1000.0 / a_ms if a_ms > 0 else float("inf")
            for net in network_configs:
                img_lat = estimate_image_latency(net, image_config)["total_latency_ms"]
                kv_lat = estimate_kvcache_latency(net, kv_cache_config)["total_latency_ms"]
                e2e_total = img_lat + server_compute_ms + kv_lat + a_ms
                net_hz = compute_network_throughput_hz(net, image_config, kvcache_config=kv_cache_config)
                results.append({
                    "model": config.name, "category": "Device-Server Collaboration",
                    "server_system": server_system, "device_system": device_system,
                    "network": net.name, "precision": bits,
                    "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": 0.0, "network_kv_cache_ms": kv_lat,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
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
                e2e_total = img_lat + server_compute_ms + kv_lat + a_ms
                net_hz = min(
                    compute_network_throughput_hz(local_net, image_config, kvcache_config=kv_cache_config),
                    compute_network_throughput_hz(cloud_net, image_config, kvcache_config=kv_cache_config),
                )
                results.append({
                    "model": config.name, "category": "Device-Server Collaboration",
                    "server_system": server_system, "device_system": device_system,
                    "network": net_name, "precision": bits,
                    "florence2_ms": f_ms, "policy_ms": p_ms, "action_ms": a_ms,
                    "network_image_ms": img_lat, "network_action_ms": 0.0, "network_kv_cache_ms": kv_lat,
                    "e2e_total_ms": e2e_total, "frequency_hz": 1000 / e2e_total,
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
    if logger is None:
        logger = logging.getLogger(__name__)

    exp_header = f"EXPERIMENT {experiment_num}: " if experiment_num is not None else ""
    logger.info("\n" + "=" * 130)
    logger.info(f"{exp_header}X-VLA DEVICE-SERVER COLLABORATION")
    logger.info("=" * 130)
    logger.info(f"Server: {server_system} (Florence-2 + policy prefill), "
                f"Device: {device_system} (flow matching denoising)")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_results = {}
    all_dfs = []

    for config in ALL_XVLA_CONFIGS:
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
        out = output_path / "xvla_device_server_collaboration.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(out, index=False)
        logger.info(f"\nResults saved to {out}")

    return all_results


# ==============================================================================
# Summary
# ==============================================================================

def print_summary(results: dict[str, pd.DataFrame], logger=None) -> None:
    """Print a summary table of X-VLA E2E performance."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\nX-VLA model characteristics:")
    logger.info(f"  VLM:     Florence-2-Large ({XVLA_FLORENCE2_TOKENS} fused vision+text tokens)")
    logger.info(f"  Policy:  SoftPromptedTransformer 24L/1024H ({XVLA_SOFT_PROMPT_TOKENS} soft prompt tokens)")
    logger.info(f"  Action:  Flow Matching, {XVLA_DENOISING_STEPS} steps, {XVLA_ACTION_TOKENS}D action space")

    logger.info("\n" + "=" * 120)
    logger.info("X-VLA Performance Summary")
    logger.info("=" * 120)

    if "e2e" in results and not results["e2e"].empty:
        df = results["e2e"]
        logger.info("-" * 120)
        logger.info(
            f"{'Hardware':<18} {'Chips':<6} {'Batch':<6} "
            f"{'Florence2 (ms)':>16} {'Policy (ms)':>14} {'Action (ms)':>14} "
            f"{'E2E (ms)':>12} {'Hz':>10}"
        )
        logger.info("-" * 120)
        for _, row in df.iterrows():
            e2e = row["e2e_time_ms"]
            hz = 1000 / e2e if e2e > 0 else 0
            logger.info(
                f"{row['hardware.name']:<18} {int(row['hardware.num_chips']):<6} "
                f"{int(row['batch_size']):<6} "
                f"{row['florence2_time_ms']:>16.2f} {row['policy_time_ms']:>14.2f} "
                f"{row['action_time_ms']:>14.2f} {e2e:>12.2f} {hz:>10.1f}"
            )
        logger.info("-" * 120)
    else:
        logger.warning("No E2E results available.")


if __name__ == "__main__":
    logger = setup_logging("perf_results/xvla_perf.log")

    system_list = ["A100_80GB", "H100", "B100", "RTX_3090", "RTX_4090", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"
    denoising_steps = XVLA_DENOISING_STEPS

    runall = True
    run_exp_1 = True
    run_exp_2 = True
    run_exp_3 = True
    run_exp_4 = True
    run_exp_5 = True
    run_exp_6 = True
    run_exp_7 = True

    logger.info("=" * 100)
    logger.info("Starting X-VLA Performance Evaluation (7 Experiments)")
    logger.info("=" * 100)
    logger.info(f"Systems: {system_list}")
    logger.info(f"Devices: {num_device_list}")
    logger.info(f"Precision: {bits}")

    exp_counter = 0

    if runall or run_exp_1:
        exp_counter += 1
        all_results = get_all_xvla_perf(
            system_list=system_list,
            num_device_list=num_device_list,
            bits=bits,
            denoising_steps=denoising_steps,
            experiment_num=exp_counter,
            logger=logger,
        )
        print_summary(all_results.get("xvla-0.9b", {}), logger=logger)

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
        compare_denoising_steps_action_dimensions(
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
