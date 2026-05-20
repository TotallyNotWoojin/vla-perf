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

Reference:
    arxiv.org/abs/2510.10274
    github.com/2toinf/X-VLA
    genz/GenZ/Models/Model_sets/vla_models.py -> X-VLA model configs
"""

import pandas as pd
import logging
from pathlib import Path

from perf_utils import (
    get_powers_of_two_up_to,
    get_optimal_df,
    collect_prefill_perf,
    collect_parallel_decode_perf,
    RESULT_COLUMNS,
    setup_logging,
)


# X-VLA Architecture Constants
XVLA_FLORENCE2_TOKENS = 576        # fused vision + text tokens from Florence-2-Large (768px input)
XVLA_SOFT_PROMPT_TOKENS = 16       # learnable embodiment soft prompt tokens
XVLA_POLICY_CONTEXT_TOKENS = XVLA_FLORENCE2_TOKENS + XVLA_SOFT_PROMPT_TOKENS
XVLA_ACTION_DIM = 20               # ee6d action space: 20D
XVLA_ACTION_TOKENS = XVLA_ACTION_DIM  # one token per action dimension
XVLA_DENOISING_STEPS = 30         # default flow matching steps (paper uses 30)


def get_xvla_florence2_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """Evaluate Florence-2-Large joint vision + text encoding."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  Florence-2 encoding — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model="florence2-large-encoder",
                system=system,
                num_devices=num_devices,
                input_tokens=XVLA_FLORENCE2_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No Florence-2 results collected for X-VLA")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_xvla_policy_prefill_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """Evaluate X-VLA SoftPromptedTransformer policy prefill."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  Policy prefill — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model="xvla-policy",
                system=system,
                num_devices=num_devices,
                input_tokens=XVLA_POLICY_CONTEXT_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No policy prefill results collected for X-VLA")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_xvla_action_perf(
    system_list: list[str],
    num_device_list: list[int],
    denoising_steps: int = XVLA_DENOISING_STEPS,
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate X-VLA flow matching action denoiser.

    Each denoising step is a parallel decode pass over XVLA_ACTION_TOKENS tokens,
    conditioned on the policy transformer's hidden states.
    Total action latency = denoising_steps * single-step latency.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  Action denoiser — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_parallel_decode_perf(
                model="xvla-policy",
                system=system,
                num_devices=num_devices,
                input_tokens=XVLA_POLICY_CONTEXT_TOKENS,
                output_tokens_parallel=XVLA_ACTION_TOKENS,
                self_attention=True,
                bits=bits,
            )
            if model_results:
                for r in model_results:
                    r["time_ms"] *= denoising_steps
                    r["model.dec_steps"] = denoising_steps
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No action denoiser results collected for X-VLA")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_xvla_e2e_perf(
    system_list: list[str] = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    denoising_steps: int = XVLA_DENOISING_STEPS,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate end-to-end X-VLA performance across all components.

    Returns a dict of DataFrames: florence2, policy_prefill, action, e2e.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    logger.info("Evaluating X-VLA Florence-2 encoding...")
    df_florence2 = get_xvla_florence2_perf(system_list, num_device_list, bits, logger)
    results["florence2"] = df_florence2
    df_florence2.to_csv(output_path / "xvla_florence2_perf.csv", index=False)

    logger.info("Evaluating X-VLA policy transformer prefill...")
    df_policy = get_xvla_policy_prefill_perf(system_list, num_device_list, bits, logger)
    results["policy_prefill"] = df_policy
    df_policy.to_csv(output_path / "xvla_policy_prefill_perf.csv", index=False)

    logger.info(f"Evaluating X-VLA action denoiser ({denoising_steps} steps)...")
    df_action = get_xvla_action_perf(
        system_list, num_device_list, denoising_steps, bits, logger
    )
    results["action"] = df_action
    df_action.to_csv(output_path / "xvla_action_perf.csv", index=False)

    logger.info("Computing end-to-end latency...")
    if df_florence2.empty or df_policy.empty or df_action.empty:
        logger.warning("One or more component DataFrames empty — skipping E2E.")
        results["e2e"] = pd.DataFrame()
    else:
        group_cols = ["hardware.name", "hardware.num_chips", "batch_size"]

        f2_times = (
            df_florence2[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "florence2_time_ms"})
        )
        policy_times = (
            df_policy[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "policy_time_ms"})
        )
        action_times = (
            df_action[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "action_time_ms"})
        )

        df_merged = f2_times.merge(policy_times, on=group_cols, how="inner")
        df_merged = df_merged.merge(action_times, on=group_cols, how="inner")
        df_merged["e2e_time_ms"] = (
            df_merged["florence2_time_ms"]
            + df_merged["policy_time_ms"]
            + df_merged["action_time_ms"]
        )

        df_merged["model.name"] = "xvla-0.9b"
        df_merged["model.stage"] = "e2e"
        df_merged["model.dec_steps"] = denoising_steps
        df_merged["model.seq_len_inference_prefill"] = XVLA_POLICY_CONTEXT_TOKENS

        df_e2e = df_merged[[
            "model.name", "model.stage", "model.dec_steps",
            "model.seq_len_inference_prefill",
            "hardware.name", "hardware.num_chips", "batch_size",
            "florence2_time_ms", "policy_time_ms", "action_time_ms", "e2e_time_ms",
        ]]
        results["e2e"] = df_e2e
        df_e2e.to_csv(output_path / "xvla_e2e_perf.csv", index=False)
        logger.info(f"  -> Saved to {output_path / 'xvla_e2e_perf.csv'}")

    return results


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
    logger.info("=" * 80)
    logger.info("Starting X-VLA Performance Evaluation")
    logger.info("=" * 80)

    system_list = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"

    logger.info(f"Systems: {system_list}")
    logger.info(f"Devices: {num_device_list}")
    logger.info(f"Precision: {bits}")

    results = get_xvla_e2e_perf(
        system_list=system_list,
        num_device_list=num_device_list,
        bits=bits,
        logger=logger,
    )

    print_summary(results, logger=logger)

    logger.info("=" * 80)
    logger.info("X-VLA evaluation complete")
    logger.info("=" * 80)
