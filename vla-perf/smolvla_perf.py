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
- Vision:  SigLIP-SO400M/14 (64 visual tokens per frame via layer skipping)
- VLM:     SmolLM2-1.7B (24 layers, 2048 hidden) — processes visual + text tokens
- Action:  Flow Matching Transformer (~100M, 10 layers, 768 hidden)
           Cross-attends to VLM hidden states; N denoising steps.

Performance modeling breakdown:
1. Vision encoding  (SigLIP prefill)
2. VLM prefill      (SmolLM2-1.7B processes visual + language tokens)
3. Action Expert    (flow matching parallel decode, N denoising steps)

Reference:
    arxiv.org/abs/2506.01844
    huggingface.co/blog/smolvla
    genz/GenZ/Models/Model_sets/vla_models.py -> SmolVLA model configs
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


# SmolVLA Architecture Constants
SMOLVLA_VISION_TOKENS = 64         # 64 visual tokens per frame (layer-skipped SigLIP)
SMOLVLA_LANGUAGE_TOKENS = 32       # Typical short instruction token count
SMOLVLA_TOTAL_VLM_TOKENS = SMOLVLA_VISION_TOKENS + SMOLVLA_LANGUAGE_TOKENS
SMOLVLA_ACTION_CHUNK = 16          # Action tokens per flow matching pass
SMOLVLA_DENOISING_STEPS = 10       # Default denoising iterations


def get_smolvla_vision_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """Evaluate SmolVLA SigLIP vision encoding."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  Vision — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model="smolvla-vision",
                system=system,
                num_devices=num_devices,
                input_tokens=SMOLVLA_VISION_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No vision results collected for SmolVLA")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_smolvla_vlm_perf(
    system_list: list[str],
    num_device_list: list[int],
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """Evaluate SmolLM2-1.7B VLM prefill for SmolVLA."""
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  VLM prefill — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_prefill_perf(
                model="smollm2-1.7b",
                system=system,
                num_devices=num_devices,
                input_tokens=SMOLVLA_TOTAL_VLM_TOKENS,
                bits=bits,
            )
            if model_results:
                results.extend(model_results)

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("No VLM prefill results collected for SmolVLA")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_smolvla_action_perf(
    system_list: list[str],
    num_device_list: list[int],
    denoising_steps: int = SMOLVLA_DENOISING_STEPS,
    action_chunk: int = SMOLVLA_ACTION_CHUNK,
    bits: str = "bf16",
    logger=None,
) -> pd.DataFrame:
    """
    Evaluate SmolVLA action expert (flow matching transformer).

    Each denoising step is a parallel decode pass over `action_chunk` tokens,
    cross-attending to the VLM's hidden states. Total action latency =
    denoising_steps * single-step latency.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for system in system_list:
        logger.info(f"  Action Expert — system: {system}")
        for num_devices in num_device_list:
            model_results = collect_parallel_decode_perf(
                model="smolvla-action-expert",
                system=system,
                num_devices=num_devices,
                input_tokens=SMOLVLA_TOTAL_VLM_TOKENS,
                output_tokens_parallel=action_chunk,
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
        logger.warning("No action expert results collected for SmolVLA")
        return df
    return get_optimal_df(df, apply_pareto=True)


def get_smolvla_e2e_perf(
    system_list: list[str] = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"],
    num_device_list: list[int] = None,
    denoising_steps: int = SMOLVLA_DENOISING_STEPS,
    action_chunk: int = SMOLVLA_ACTION_CHUNK,
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate end-to-end SmolVLA performance across all components.

    Returns a dict of DataFrames: vision, vlm, action, e2e.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(4)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    logger.info("Evaluating SmolVLA vision encoding...")
    df_vision = get_smolvla_vision_perf(system_list, num_device_list, bits, logger)
    results["vision"] = df_vision
    df_vision.to_csv(output_path / "smolvla_vision_perf.csv", index=False)

    logger.info("Evaluating SmolVLA VLM prefill (SmolLM2-1.7B)...")
    df_vlm = get_smolvla_vlm_perf(system_list, num_device_list, bits, logger)
    results["vlm"] = df_vlm
    df_vlm.to_csv(output_path / "smolvla_vlm_perf.csv", index=False)

    logger.info(f"Evaluating SmolVLA action expert ({denoising_steps} denoising steps)...")
    df_action = get_smolvla_action_perf(
        system_list, num_device_list, denoising_steps, action_chunk, bits, logger
    )
    results["action"] = df_action
    df_action.to_csv(output_path / "smolvla_action_perf.csv", index=False)

    logger.info("Computing end-to-end latency...")
    if df_vision.empty or df_vlm.empty or df_action.empty:
        logger.warning("One or more component DataFrames empty — skipping E2E.")
        results["e2e"] = pd.DataFrame()
    else:
        group_cols = ["hardware.name", "hardware.num_chips", "batch_size"]

        vision_times = (
            df_vision[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "vision_time_ms"})
        )
        vlm_times = (
            df_vlm[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "vlm_time_ms"})
        )
        action_times = (
            df_action[group_cols + ["time_ms"]].copy()
            .rename(columns={"time_ms": "action_time_ms"})
        )

        df_merged = vision_times.merge(vlm_times, on=group_cols, how="inner")
        df_merged = df_merged.merge(action_times, on=group_cols, how="inner")
        df_merged["e2e_time_ms"] = (
            df_merged["vision_time_ms"]
            + df_merged["vlm_time_ms"]
            + df_merged["action_time_ms"]
        )

        df_merged["model.name"] = "smolvla"
        df_merged["model.stage"] = "e2e"
        df_merged["model.dec_steps"] = denoising_steps
        df_merged["model.seq_len_inference_prefill"] = SMOLVLA_TOTAL_VLM_TOKENS

        df_e2e = df_merged[[
            "model.name", "model.stage", "model.dec_steps",
            "model.seq_len_inference_prefill",
            "hardware.name", "hardware.num_chips", "batch_size",
            "vision_time_ms", "vlm_time_ms", "action_time_ms", "e2e_time_ms",
        ]]
        results["e2e"] = df_e2e
        df_e2e.to_csv(output_path / "smolvla_e2e_perf.csv", index=False)
        logger.info(f"  -> Saved to {output_path / 'smolvla_e2e_perf.csv'}")

    return results


def print_summary(results: dict[str, pd.DataFrame], logger=None) -> None:
    """Print a summary table of SmolVLA E2E performance."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\nSmolVLA model characteristics:")
    logger.info(f"  Vision:  SigLIP-SO400M/14 ({SMOLVLA_VISION_TOKENS} tokens via layer skipping)")
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
    logger.info("=" * 80)
    logger.info("Starting SmolVLA Performance Evaluation")
    logger.info("=" * 80)

    system_list = ["A100_80GB", "H100", "B100", "Jetson_AGX_Thor"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"

    logger.info(f"Systems: {system_list}")
    logger.info(f"Devices: {num_device_list}")
    logger.info(f"Precision: {bits}")

    results = get_smolvla_e2e_perf(
        system_list=system_list,
        num_device_list=num_device_list,
        bits=bits,
        logger=logger,
    )

    print_summary(results, logger=logger)

    logger.info("=" * 80)
    logger.info("SmolVLA evaluation complete")
    logger.info("=" * 80)
