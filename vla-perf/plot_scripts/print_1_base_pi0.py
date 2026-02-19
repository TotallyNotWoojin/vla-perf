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
Generate LaTeX tables for Pi0 base performance results.

This script loads results from the first experiment of pi0_perf.py 
(pi0_family_e2e_perf.csv) and generates two LaTeX tables:
1. Hardware performance table: ms and % of each component / E2E
2. Workload characteristics table: Memory/Compute bound for each component
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path to import system_configs
script_dir = Path(__file__).parent.resolve()
genz_systems_path = script_dir.parent.parent / "genz" / "Systems"
genz_systems_path = genz_systems_path.resolve()

if str(genz_systems_path) not in sys.path:
    sys.path.insert(0, str(genz_systems_path))

from system_configs import system_configs

# Import plot utilities
from plot_util import rename_hardware


def load_base_pi0_results(csv_path: str = "../perf_results/pi0_family_e2e_perf.csv") -> pd.DataFrame:
    """Load the base Pi0 results from CSV."""
    script_dir = Path(__file__).parent
    full_path = script_dir / csv_path
    df = pd.read_csv(full_path)
    
    # Filter for Batch=1, Chips=1, and pi0 model
    df_filtered = df[(df['batch_size'] == 1) & 
                     (df['hardware.num_chips'] == 1) & 
                     (df['model.name'] == 'pi0')]
    
    return df_filtered


def generate_performance_table(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate Table 1: Hardware performance with ms and % of E2E merged.
    
    Columns: Hardware | Vision | VLM | Action | E2E (ms/Hz)
    Format: xx ms (yy%)
    """
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("    \\begin{footnotesize}")
    latex_lines.append("    \\caption{Inference performance of $\\pi_0$ on various GPUs without considering network latency.}")
    latex_lines.append("    % \\vspace{-1em}")
    latex_lines.append("    \\label{tab:pi0_base_performance}")
    latex_lines.append("    \\scalebox{1.0}{")
    latex_lines.append("        \\begin{tabular}{L{5em} R{5em} R{5em} R{5em} R{5em} R{5em}}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Hardware}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Vision Lat.}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{VLM Lat.}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Action Lat.}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{E2E Lat.}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{E2E Freq.}} \\\\")
    latex_lines.append("\\midrule")
    
    # Sort by hardware name for consistent ordering
    hardware_order = ["Jetson_AGX_Thor", "RTX_4090", "A100_80GB", "H100", "B100"]
    
    for i, hw in enumerate(hardware_order):
        hw_data = df[df['hardware.name'] == hw]
        if hw_data.empty:
            continue
        
        row = hw_data.iloc[0]
        vision_ms = row['vision_time_ms']
        vlm_ms = row['vlm_time_ms']
        action_ms = row['action_time_ms']
        e2e_ms = row['e2e_time_ms']
        
        # Frequency
        freq_hz = 1000 / e2e_ms
        
        # Format hardware name using rename utility
        hw_display = rename_hardware(hw)
        
        # Use hdashline between rows, bottomrule at the end
        line_ending = " \\\\\n\\hdashline" if i < len(hardware_order) - 1 else " \\\\"
        
        latex_lines.append(
            f"{hw_display}\n"
            f"& {vision_ms:.2f} ms\n"
            f"& {vlm_ms:.2f} ms\n"
            f"& {action_ms:.2f} ms\n"
            f"& {e2e_ms:.2f} ms\n"
            "& \cellcolor{lightgray} " + f"{freq_hz:.1f} Hz{line_ending}"
        )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("    }")
    latex_lines.append("    \\end{footnotesize}")
    latex_lines.append("\\end{table*}")
    
    # Save to file
    output_file = output_dir / "1_pi0_perf.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Table 1 saved to: {output_file}")
    return '\n'.join(latex_lines)


def get_hardware_oi(hw_name: str) -> float:
    """
    Calculate hardware operational intensity (OI).
    
    OI = Peak FLOPS / Memory Bandwidth
       = (TFLOPS * 10^12) / (Memory_BW GB/s * 10^9)
       = TFLOPS * 1000 / Memory_BW
    
    Uses bf16 precision, falling back to fp16 if bf16 not available.
    
    Args:
        hw_name: Hardware configuration name
    
    Returns:
        Hardware OI in FLOP/Byte
    """
    if hw_name not in system_configs:
        raise ValueError(f"Hardware {hw_name} not found in system_configs")
    
    config = system_configs[hw_name]
    flops_dict = config['Flops']
    memory_bw = config['Memory_BW']
    
    # Get TFLOPS, prefer bf16, fallback to fp16
    if isinstance(flops_dict, dict):
        tflops = flops_dict.get('bf16', flops_dict.get('fp16', None))
        if tflops is None:
            raise ValueError(f"Hardware {hw_name} has no bf16 or fp16 FLOPS data")
    else:
        # Legacy format: single number assumed to be bf16
        tflops = flops_dict
    
    # Calculate OI: FLOP/Byte
    hardware_oi = tflops * 1000 / memory_bw
    return hardware_oi


def determine_boundness(model_oi: float, hw_oi: float) -> str:
    """
    Determine if workload is memory or compute bound.
    
    Args:
        model_oi: Model operational intensity (FLOP/Byte)
        hw_oi: Hardware operational intensity (FLOP/Byte)
    
    Returns:
        "Memory" if model_oi < hw_oi, "Compute" otherwise
    """
    return "Memory" if model_oi < hw_oi else "Compute"


def generate_workload_characteristics_table(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate Table 2: Workload characteristics showing Memory/Compute bound.
    
    Columns: Hardware | Balance OI | Vision | VLM | Action
    Cells: Memory/Compute bound based on Model OI vs HW OI comparison
    """
    # Model OIs from the original table
    vision_model_oi = 321.4
    vlm_model_oi = 542.8
    action_model_oi = 54.0
    
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("    \\begin{footnotesize}")
    latex_lines.append("    ")
    latex_lines.append("    \\caption{Compute- vs. memory-bound analysis of $\\pi_0$ across different hardware. Operator intensity (OI) denotes the ratio between compute operations and memory accesses (FLOPs/Bytes). The balance OI denotes the hardware balance point at which compute throughput and memory bandwidth are equally limiting.}")
    latex_lines.append("")
    latex_lines.append("    \\vspace{.5em}")
    latex_lines.append("    \\label{tab:pi0_base_workload}")
    latex_lines.append("    \\scalebox{1.0}{")
    latex_lines.append("        \\begin{tabular}{L{5em} M{5em} M{6em} M{6em} M{6em}}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Hardware}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Balance OI}} ")
    latex_lines.append(f"& \\multicolumn{{1}}{{c}}{{\\textbf{{Vision}} (OI={vision_model_oi})}}")
    latex_lines.append(f"& \\multicolumn{{1}}{{c}}{{\\textbf{{VLM}} (OI={vlm_model_oi})}}")
    latex_lines.append(f"& \\multicolumn{{1}}{{c}}{{\\textbf{{Action}} (OI={action_model_oi})}} \\\\")
    latex_lines.append("\\midrule")
    # Sort by hardware name for consistent ordering
    hardware_order = ["Jetson_AGX_Thor", "RTX_4090", "A100_80GB", "H100", "B100"]
    
    for i, hw in enumerate(hardware_order):
        hw_data = df[df['hardware.name'] == hw]
        if hw_data.empty:
            continue
        
        # Get hardware OI
        hw_oi = get_hardware_oi(hw)
        
        # Determine boundness based on model OI vs hardware OI
        vision_bound = determine_boundness(vision_model_oi, hw_oi)
        vlm_bound = determine_boundness(vlm_model_oi, hw_oi)
        action_bound = determine_boundness(action_model_oi, hw_oi)
        
        # Format hardware name using rename utility
        hw_display = rename_hardware(hw)
        
        # Use hdashline between rows, bottomrule at the end
        line_ending = " \\\\\n\\hdashline" if i < len(hardware_order) - 1 else " \\\\"
        
        latex_lines.append(
            f"{hw_display} \n"
            f"& {hw_oi:.1f} \n"
            f"& {vision_bound}\n"
            f"& {vlm_bound}\n"
            f"& {action_bound}{line_ending}"
        )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("    }")
    latex_lines.append("    \\end{footnotesize}")
    latex_lines.append("\\end{table}")
    
    # Save to file
    output_file = output_dir / "1_pi0_bound.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Table 2 saved to: {output_file}")
    return '\n'.join(latex_lines)


def main():
    """Main function to generate both tables."""
    # Load data
    print("Loading Pi0 base performance results...")
    df = load_base_pi0_results()
    
    print(f"Found {len(df)} hardware configurations (Batch=1, Chips=1)")
    print(f"Hardware: {df['hardware.name'].unique().tolist()}")
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "../paper_tables"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate Table 1: Performance
    print("\n" + "="*80)
    print("Generating Table 1: Performance Table")
    print("="*80)
    table1 = generate_performance_table(df, output_dir)
    print("\n" + table1)
    
    # Generate Table 2: Workload Characteristics
    print("\n" + "="*80)
    print("Generating Table 2: Workload Characteristics Table")
    print("="*80)
    table2 = generate_workload_characteristics_table(df, output_dir)
    print("\n" + table2)
    
    print("\n" + "="*80)
    print("Done! Tables saved to:", output_dir)
    print("="*80)


if __name__ == "__main__":
    main()
