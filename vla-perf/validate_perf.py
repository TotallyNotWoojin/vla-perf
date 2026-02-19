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
Performance Validation: Roofline Model vs Real Measurements

Compares our analytical roofline performance model predictions against real measured
latencies from the realtime-vla project (https://github.com/Dexmal/realtime-vla).

realtime-vla uses hand-optimized Triton kernels for Pi0/Pi0.5 inference on
consumer GPUs (RTX 4090, RTX 5090). Their benchmarks measure end-to-end
latency for one set of observations with:
    - 10 flow matching denoising steps
    - Chunk size 63 (action tokens)
    - Empty prompt (0 language tokens)

Reference benchmark matrix (from realtime-vla README):

    Model / Backend   RTX 4090 (1v)  RTX 4090 (2v)  RTX 4090 (3v)  RTX 5090 (1v)  RTX 5090 (2v)  RTX 5090 (3v)
    Pi0 Triton        20.0ms         27.3ms         36.8ms         17.6ms         24.0ms         31.9ms
    Pi05 Triton       22.1ms         29.2ms         38.9ms         20.1ms         26.6ms         34.2ms

The "% of roofline achieved" metric shows how close the real system gets to the
analytical roofline prediction:
    roofline_pct = roofline_time / real_time * 100
    (e.g., roofline=15ms, real=20ms -> 15/20 = 75% of roofline achieved)

Note: Pi0.5 has the same transformer architecture as Pi0 (same params), only
differing in normalization (adaRMSNorm). We only validate against Pi0 here.

Reference files:
    pi0_perf.py -> Pi0 performance evaluation functions
    genz/GenZ/Models/Model_sets/vla_models.py -> Pi0 model configs
    genz/Systems/system_configs.py -> Hardware system configs (RTX 4090, RTX 5090)
"""

import pandas as pd
import logging
from pathlib import Path

from Systems.system_configs import system_configs

from perf_utils import setup_logging

from pi0_perf import (
    Pi0Config,
    PI0_VISION_TOKENS,
    get_pi0_vision_perf,
    get_pi0_vlm_perf,
    get_pi0_action_expert_perf,
)


# ==============================================================================
# Real benchmark data from realtime-vla
# https://github.com/Dexmal/realtime-vla
#
# Conditions: 10 flow steps, chunk size 63, empty prompt
# ==============================================================================

REALTIME_VLA_BENCHMARKS = {
    # (system, num_views): measured_latency_ms
    ("RTX_4090", 1): 20.0,
    ("RTX_4090", 2): 27.3,
    ("RTX_4090", 3): 36.8,
    ("RTX_5090", 1): 17.6,
    ("RTX_5090", 2): 24.0,
    ("RTX_5090", 3): 31.9,
}


# ==============================================================================
# Benchmark parameters
# ==============================================================================
BENCHMARK_DENOISING_STEPS = 10
BENCHMARK_CHUNK_SIZE = 63
BENCHMARK_LANGUAGE_TOKENS = 0  # Empty prompt


def create_validation_config(num_views: int) -> Pi0Config:
    """
    Create a Pi0Config matching the realtime-vla benchmark conditions.

    Pi0 benchmark: SigLIP + Gemma 2B + 300M DiT, 10 flow steps,
    chunk size 63, empty prompt.

    Args:
        num_views: Number of camera views (1, 2, or 3)

    Returns:
        Pi0Config for validation
    """
    vlm_seq_len = PI0_VISION_TOKENS * num_views + BENCHMARK_LANGUAGE_TOKENS

    return Pi0Config(
        name=f"pi0-{num_views}v",
        vision_model="pi0-vision",
        vlm_model="pi0-vlm",
        action_expert_model="pi0-action-expert",
        vision_tokens=PI0_VISION_TOKENS,
        vision_frames=num_views,
        language_tokens=BENCHMARK_LANGUAGE_TOKENS,
        vlm_sequence_length=vlm_seq_len,
        action_chunk_size=BENCHMARK_CHUNK_SIZE,
        denoising_steps=BENCHMARK_DENOISING_STEPS,
    )


def run_validation(
    systems: list[str] = ["RTX_4090", "RTX_5090"],
    views_list: list[int] = [1, 2, 3],
    bits: str = "bf16",
    output_dir: str = "perf_results",
    logger=None,
) -> pd.DataFrame:
    """
    Run roofline model for configurations matching realtime-vla Pi0 benchmarks
    and compare against real measurements.

    Args:
        systems: Hardware systems to evaluate
        views_list: Number of camera views to test
        bits: Precision (bf16 for consistency with realtime-vla bfloat16)
        output_dir: Directory to save results
        logger: Logger instance

    Returns:
        DataFrame with roofline predictions, real measurements, and roofline efficiency
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Filter systems to only those available in system_configs
    available_systems = [s for s in systems if s in system_configs]
    missing_systems = set(systems) - set(available_systems)
    if missing_systems:
        logger.warning(f"Systems not found in system_configs: {missing_systems}")

    if not available_systems:
        logger.error("No valid systems to evaluate.")
        return pd.DataFrame()

    logger.info("=" * 120)
    logger.info("PERFORMANCE VALIDATION: Roofline Model vs realtime-vla Measurements")
    logger.info("=" * 120)
    logger.info(f"Reference: https://github.com/Dexmal/realtime-vla")
    logger.info(f"Benchmark conditions: {BENCHMARK_DENOISING_STEPS} flow steps, "
                f"chunk size {BENCHMARK_CHUNK_SIZE}, empty prompt, batch=1, {bits}")
    logger.info(f"Systems: {available_systems}")
    logger.info(f"Views: {views_list}")
    logger.info("")

    results = []

    for system in available_systems:
        for num_views in views_list:
            logger.info(f"Evaluating Pi0 on {system} with {num_views} view(s)...")

            config = create_validation_config(num_views)

            # --- Vision encoder ---
            df_vision = get_pi0_vision_perf(config, [system], [1], bits, max_batch_size=1)
            if df_vision.empty:
                logger.warning(f"  Vision perf empty for {system}, skipping")
                continue
            vision_row = df_vision[df_vision["batch_size"] == 1].iloc[0]
            vision_ms = vision_row["time_ms"]
            vision_bound = vision_row.get("boundness", "N/A")

            # --- VLM backbone ---
            df_vlm = get_pi0_vlm_perf(config, [system], [1], bits, max_batch_size=1)
            if df_vlm.empty:
                logger.warning(f"  VLM perf empty for {system}, skipping")
                continue
            vlm_row = df_vlm[df_vlm["batch_size"] == 1].iloc[0]
            vlm_ms = vlm_row["time_ms"]
            vlm_bound = vlm_row.get("boundness", "N/A")

            # --- Action expert (DiT x denoising steps) ---
            df_action = get_pi0_action_expert_perf(
                config, [system], [1], bits,
                denoising_steps=BENCHMARK_DENOISING_STEPS,
                vlm_sequence_length=config.vlm_sequence_length,
                action_chunk_size=BENCHMARK_CHUNK_SIZE,
                max_batch_size=1,
            )
            if df_action.empty:
                logger.warning(f"  Action perf empty for {system}, skipping")
                continue
            action_row = df_action[df_action["batch_size"] == 1].iloc[0]
            action_ms = action_row["time_ms"]
            action_bound = action_row.get("boundness", "N/A")

            # --- E2E total ---
            e2e_ms = vision_ms + vlm_ms + action_ms
            real_ms = REALTIME_VLA_BENCHMARKS.get((system, num_views))

            result = {
                "system": system,
                "num_views": num_views,
                "precision": bits,
                "vlm_seq_len": config.vlm_sequence_length,
                "vision_ms": round(vision_ms, 2),
                "vlm_ms": round(vlm_ms, 2),
                "action_ms": round(action_ms, 2),
                "roofline_ms": round(e2e_ms, 2),
                "vision_bound": vision_bound,
                "vlm_bound": vlm_bound,
                "action_bound": action_bound,
                "real_ms": real_ms,
            }
            if real_ms is not None:
                result["roofline_pct"] = round(e2e_ms / real_ms * 100, 1)

            results.append(result)

            logger.info(f"  Vision: {vision_ms:.2f}ms ({vision_bound}), "
                        f"VLM: {vlm_ms:.2f}ms ({vlm_bound}), "
                        f"Action: {action_ms:.2f}ms ({action_bound})")
            logger.info(f"  Roofline: {e2e_ms:.2f}ms | Real Pi0: {real_ms}ms")

    df = pd.DataFrame(results)

    # Save to CSV
    if not df.empty:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / "perf_validation_vs_realtime_vla.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")

    return df


def print_validation_summary(df: pd.DataFrame, logger=None) -> None:
    """
    Print formatted comparison tables of roofline predictions vs real measurements.

    The "Roofline %" metric shows what fraction of the roofline (analytical best-case)
    the real system achieves: roofline_time / real_time * 100.
    A value of 75% means the real system is 1.33x slower than roofline.

    Args:
        df: DataFrame from run_validation()
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if df.empty:
        logger.info("No results to display.")
        return

    # ---- Table 1: Detailed comparison ----
    logger.info("")
    logger.info("=" * 120)
    logger.info("DETAILED COMPARISON: Pi0 Roofline vs Real (realtime-vla)")
    logger.info("=" * 120)
    logger.info(f"Benchmark: {BENCHMARK_DENOISING_STEPS} flow steps, "
                f"chunk size {BENCHMARK_CHUNK_SIZE}, empty prompt, batch=1, bf16")
    logger.info("-" * 120)

    header = (f"{'System':<12} {'Views':>5} {'VLM Seq':>8} "
              f"{'Vision':>8} {'VLM':>8} {'Action':>8} "
              f"{'Roofline':>9} {'Real':>8} {'Roofline %':>11}")
    logger.info(header)

    units = (f"{'':12} {'':>5} {'(tok)':>8} "
             f"{'(ms)':>8} {'(ms)':>8} {'(ms)':>8} "
             f"{'(ms)':>9} {'(ms)':>8} {'':>11}")
    logger.info(units)
    logger.info("-" * 120)

    for _, row in df.iterrows():
        system = row["system"]
        views = int(row["num_views"])
        vlm_seq = int(row["vlm_seq_len"])
        vision = row["vision_ms"]
        vlm = row["vlm_ms"]
        action = row["action_ms"]
        roofline = row["roofline_ms"]
        real = row.get("real_ms")

        real_str = f"{real:.1f}" if pd.notna(real) else "N/A"
        pct_str = "N/A"
        if "roofline_pct" in row and pd.notna(row.get("roofline_pct")):
            pct_str = f"{row['roofline_pct']:.1f}%"

        line = (f"{system:<12} {views:>5} {vlm_seq:>8} "
                f"{vision:>8.2f} {vlm:>8.2f} {action:>8.2f} "
                f"{roofline:>9.2f} {real_str:>8} {pct_str:>11}")
        logger.info(line)

    logger.info("-" * 120)

    # ---- Table 2: Side-by-side benchmark matrix ----
    logger.info("")
    logger.info("=" * 110)
    logger.info("BENCHMARK MATRIX: Roofline vs Real (E2E latency in ms)")
    logger.info("=" * 110)

    systems = df["system"].unique()
    views_list = sorted(df["num_views"].unique())

    # Build header
    col_headers = [""]
    for system in systems:
        for v in views_list:
            col_headers.append(f"{system} ({v}v)")

    col_width = 16
    header_line = "".join(h.center(col_width) for h in col_headers)
    logger.info(header_line)
    logger.info("-" * len(header_line))

    # Row: Roofline
    row_str = "Roofline".ljust(col_width)
    for system in systems:
        for v in views_list:
            match = df[(df["system"] == system) & (df["num_views"] == v)]
            if not match.empty:
                row_str += f"{match.iloc[0]['roofline_ms']:.1f}".center(col_width)
            else:
                row_str += "N/A".center(col_width)
    logger.info(row_str)

    # Row: Pi0 Real
    row_str = "Pi0 Triton".ljust(col_width)
    for system in systems:
        for v in views_list:
            real = REALTIME_VLA_BENCHMARKS.get((system, v))
            if real is not None:
                row_str += f"{real:.1f}".center(col_width)
            else:
                row_str += "N/A".center(col_width)
    logger.info(row_str)

    # Row: Roofline %
    row_str = "Roofline %".ljust(col_width)
    for system in systems:
        for v in views_list:
            match = df[(df["system"] == system) & (df["num_views"] == v)]
            if not match.empty and "roofline_pct" in match.columns:
                pct = match.iloc[0].get("roofline_pct")
                if pd.notna(pct):
                    row_str += f"{pct:.1f}%".center(col_width)
                else:
                    row_str += "N/A".center(col_width)
            else:
                row_str += "N/A".center(col_width)
    logger.info(row_str)

    logger.info("-" * len(header_line))

    # ---- Summary statistics ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY STATISTICS: % of Roofline Achieved")
    logger.info("=" * 60)

    if "roofline_pct" in df.columns:
        pcts = df["roofline_pct"].dropna()
        if not pcts.empty:
            logger.info(f"  Mean roofline achieved:  {pcts.mean():.1f}%")
            logger.info(f"  Min roofline achieved:   {pcts.min():.1f}%")
            logger.info(f"  Max roofline achieved:   {pcts.max():.1f}%")

    logger.info("")
    logger.info("Interpretation:")
    logger.info("  Roofline % = roofline_time / real_time * 100")
    logger.info("  100% = real matches roofline (perfect HW utilization)")
    logger.info("   75% = real is 1.33x slower than roofline")
    logger.info("   50% = real is 2x slower than roofline")
    logger.info("  >100% = roofline overestimates latency (too conservative)")


def generate_validation_tex(
    df: pd.DataFrame,
    output_dir: str = "paper_tables",
    logger=None,
) -> None:
    """
    Generate a LaTeX table comparing roofline predictions vs real measurements.

    Args:
        df: DataFrame from run_validation()
        output_dir: Directory to save the .tex file
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if df.empty:
        logger.warning("No data to generate LaTeX table.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / "validate_perf.tex"

    # Collect data grouped by system and views
    systems = list(df["system"].unique())
    views_list = sorted(df["num_views"].unique())

    # Build system display names: RTX_4090 -> RTX 4090
    def system_display(s):
        return s.replace("_", " ")

    # Number of data columns
    n_cols = len(systems) * len(views_list)

    # Column spec: label column + data columns
    col_spec = "L{10em} " + " ".join(["R{5.5em}"] * n_cols)

    # Header rows
    # Top-level: system groups
    system_headers = []
    cmidrules = []
    col_idx = 2  # 1-indexed, first data column is 2
    for system in systems:
        n = len(views_list)
        system_headers.append(
            f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{system_display(system)}}}}}"
        )
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n

    # Sub-header: camera counts
    camera_headers = []
    for _ in systems:
        for v in views_list:
            label = f"{v} camera" if v == 1 else f"{v} cameras"
            camera_headers.append(f"\\multicolumn{{1}}{{c}}{{\\textbf{{{label}}}}}")

    # Data rows
    def get_val(system, views, column):
        match = df[(df["system"] == system) & (df["num_views"] == views)]
        if not match.empty and column in match.columns:
            return match.iloc[0].get(column)
        return None

    # Roofline row
    roofline_cells = []
    for system in systems:
        for v in views_list:
            val = get_val(system, v, "roofline_ms")
            roofline_cells.append(f"{val:.1f} ms" if pd.notna(val) else "N/A")

    # Real performance row
    real_cells = []
    for system in systems:
        for v in views_list:
            real = REALTIME_VLA_BENCHMARKS.get((system, v))
            real_cells.append(f"{real:.1f} ms" if real is not None else "N/A")

    # Accuracy row
    accuracy_cells = []
    for system in systems:
        for v in views_list:
            val = get_val(system, v, "roofline_pct")
            if pd.notna(val):
                accuracy_cells.append(f"\\cellcolor{{lightgray}} {val:.1f}\\%")
            else:
                accuracy_cells.append("N/A")

    # Assemble LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \setlength\dashlinedash{0.2pt}")
    lines.append(r"    \setlength\dashlinegap{1.5pt}")
    lines.append(r"    \begin{footnotesize}")
    lines.append(
        r"    \caption{Roofline model validation against real $\pi_0$ Triton "
        r"inference latencies from~\cite{ma2025runningvlasrealtimespeed}. "
        r"Benchmark conditions: 10 flow matching steps, chunk size 63, empty prompt, "
        r"batch size 1, bf16 precision. Accuracy denotes the ratio of the roofline "
        r"prediction to the real measurement "
        r"($\text{roofline} / \text{real} \times 100\%$).}"
    )
    lines.append(r"    \label{tab:validate_perf}")
    lines.append(r"    \scalebox{1.0}{")
    lines.append(f"        \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # System header
    lines.append("& " + " & ".join(system_headers) + r" \\")
    lines.append(" ".join(cmidrules))

    # Camera header
    lines.append("& " + "\n& ".join(camera_headers) + r" \\")

    lines.append(r"\midrule")

    # Roofline row
    lines.append("Roofline (ours)")
    for cell in roofline_cells:
        lines.append(f"& {cell}")
    lines.append(r"\\")

    # Dashed line
    lines.append(r"\hdashline")

    # Real performance row
    lines.append(r"Real perf.\ (Triton)")
    for cell in real_cells:
        lines.append(f"& {cell}")
    lines.append(r"\\")

    # Dashed line
    lines.append(r"\hdashline")

    # Accuracy row
    lines.append("Accuracy")
    for cell in accuracy_cells:
        lines.append(f"& {cell}")
    lines.append(r"\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"    }")
    lines.append(r"    \end{footnotesize}")
    lines.append(r"\end{table*}")

    tex_content = "\n".join(lines) + "\n"

    output_file.write_text(tex_content)
    logger.info(f"LaTeX table saved to {output_file}")


# ==============================================================================
# Main entry point
# ==============================================================================

if __name__ == "__main__":
    logger = setup_logging("perf_results/perf_validation.log")

    logger.info("Performance Validation Script")
    logger.info("Comparing roofline model vs realtime-vla measured latencies")
    logger.info("")

    # Run validation for RTX 4090 and RTX 5090
    df = run_validation(
        systems=["RTX_4090", "RTX_5090"],
        views_list=[1, 2, 3],
        bits="bf16",
        logger=logger,
    )

    # Print formatted comparison
    print_validation_summary(df, logger=logger)

    # Generate LaTeX table
    generate_validation_tex(df, logger=logger)
