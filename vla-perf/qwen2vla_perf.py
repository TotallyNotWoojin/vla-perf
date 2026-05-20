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
           224px input → 16×16=256 raw patches → 64 merged tokens (2×2 merge)
- VLM:     Qwen2-7B LLM (28 layers, 3584 hidden, GQA 28/4 heads)
           Processes merged visual tokens + language instruction tokens
- Action:  Autoregressive decode of 7 continuous action tokens
           (can swap for a diffusion head as in CogACT; see collect_parallel_decode_perf)

Performance modeling breakdown:
1. Vision encoding (Qwen2-VL ViT prefill)
2. VLM prefill     (Qwen2-7B processes merged visual + text tokens)
3. Action decode   (7 action tokens generated autoregressively)

Reference:
    Qwen2-VL: arxiv.org/abs/2409.12191
    CogACT:   arxiv.org/abs/2411.19650
    genz/GenZ/Models/Model_sets/vla_models.py -> Qwen2-VL model configs
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


# Qwen2-VL-7B VLA Architecture Constants
QWEN2VL_RAW_VISION_TOKENS = 256    # 224px / 14px patch = 16×16 = 256 raw tokens
QWEN2VL_MERGED_VISION_TOKENS = 64  # after 2×2 spatial merge in the VLM projector
QWEN2VL_LANGUAGE_TOKENS = 32       # typical instruction length
QWEN2VL_TOTAL_VLM_TOKENS = QWEN2VL_MERGED_VISION_TOKENS + QWEN2VL_LANGUAGE_TOKENS
QWEN2VL_ACTION_TOKENS = 7          # 7-DoF action output (same convention as OpenVLA)


def get_qwen2vla_vision_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """Evaluate Qwen2-VL ViT vision encoding."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  Vision — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model="qwen2-vl-7b-vision",
                system=system,
                num_devices=num_devices,
                input_tokens=QWEN2VL_RAW_VISION_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No vision results collected for Qwen2-VL")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_qwen2vla_vlm_prefill_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """Evaluate Qwen2-7B VLM prefill (merged visual tokens + language tokens)."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  VLM prefill — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model="qwen2-vl-7b-llm",
                system=system,
                num_devices=num_devices,
                input_tokens=QWEN2VL_TOTAL_VLM_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No VLM prefill results collected for Qwen2-VL")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_qwen2vla_action_decode_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """Evaluate Qwen2-7B autoregressive action token decode (7 tokens)."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  Action decode — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_decode_perf(
                model="qwen2-vl-7b-llm",
                system=system,
                num_devices=num_devices,
                input_tokens=QWEN2VL_TOTAL_VLM_TOKENS,
                output_tokens=QWEN2VL_ACTION_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No action decode results collected for Qwen2-VL")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_qwen2vla_e2e_perf(
    system_list: list[str] = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate end-to-end Qwen2-VL VLA performance across all components.

    Returns a dict of DataFrames: vision, vlm_prefill, action_decode, e2e.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    logger.info("Evaluating Qwen2-VL vision encoding...")
    df_vision = get_qwen2vla_vision_perf(system_list, num_device_list, bits, logger)
    results["vision"] = df_vision
    df_vision.to_csv(output_path / "qwen2vla_vision_perf.csv", index=False)

    logger.info("Evaluating Qwen2-VL VLM prefill...")
    df_prefill = get_qwen2vla_vlm_prefill_perf(system_list, num_device_list, bits, logger)
    results["vlm_prefill"] = df_prefill
    df_prefill.to_csv(output_path / "qwen2vla_vlm_prefill_perf.csv", index=False)

    logger.info("Evaluating Qwen2-VL action decode...")
    df_decode = get_qwen2vla_action_decode_perf(system_list, num_device_list, bits, logger)
    results["action_decode"] = df_decode
    df_decode.to_csv(output_path / "qwen2vla_action_decode_perf.csv", index=False)

    logger.info("Computing end-to-end latency...")
    if df_vision.empty or df_prefill.empty or df_decode.empty:
        logger.warning("One or more component DataFrames empty — skipping E2E.")
        results["e2e"] = pd.DataFrame()
    else:
        group_cols = ["hardware.name", "hardware.num_chips", "batch_size"]

        vision_times = (
            df_vision[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "vision_time_ms"})
        )
        prefill_times = (
            df_prefill[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "prefill_time_ms"})
        )
        # Autoregressive decode: latency returned is per-token, multiply by action tokens
        decode_times = (
            df_decode[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "decode_time_ms"})
        )
        decode_times["decode_time_ms"] *= QWEN2VL_ACTION_TOKENS

        df_merged = vision_times.merge(prefill_times, on=group_cols, how="inner")
        df_merged = df_merged.merge(decode_times, on=group_cols, how="inner")
        df_merged["e2e_time_ms"] = (
            df_merged["vision_time_ms"]
            + df_merged["prefill_time_ms"]
            + df_merged["decode_time_ms"]
        )

        df_merged["model.name"] = "qwen2-vl-7b-vla"
        df_merged["model.stage"] = "e2e"
        df_merged["model.dec_steps"] = QWEN2VL_ACTION_TOKENS
        df_merged["model.seq_len_inference_prefill"] = QWEN2VL_TOTAL_VLM_TOKENS

        df_e2e = df_merged[[
            "model.name", "model.stage", "model.dec_steps",
            "model.seq_len_inference_prefill",
            "hardware.name", "hardware.num_chips", "batch_size",
            "vision_time_ms", "prefill_time_ms", "decode_time_ms", "e2e_time_ms",
        ]]
        results["e2e"] = df_e2e
        df_e2e.to_csv(output_path / "qwen2vla_e2e_perf.csv", index=False)
        logger.info(f"  -> Saved to {output_path / 'qwen2vla_e2e_perf.csv'}")

    return results


def print_summary(results: dict[str, pd.DataFrame], logger=None) -> None:
    """Print a summary table of Qwen2-VL VLA E2E performance."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\nQwen2-VL-7B VLA model characteristics:")
    logger.info(f"  Vision:  Qwen2-VL ViT ({QWEN2VL_RAW_VISION_TOKENS} raw → {QWEN2VL_MERGED_VISION_TOKENS} merged tokens)")
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
    logger.info("=" * 80)
    logger.info("Starting Qwen2-VL-7B VLA Performance Evaluation")
    logger.info("=" * 80)

    system_list = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"

    logger.info(f"Systems: {system_list}")
    logger.info(f"Devices: {num_device_list}")
    logger.info(f"Precision: {bits}")

    results = get_qwen2vla_e2e_perf(
        system_list=system_list,
        num_device_list=num_device_list,
        bits=bits,
        logger=logger,
    )

    print_summary(results, logger=logger)

    logger.info("=" * 80)
    logger.info("Qwen2-VL-7B VLA evaluation complete")
    logger.info("=" * 80)
