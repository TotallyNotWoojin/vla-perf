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
Generate plots and LaTeX tables for Pi0 long context experiment results.

This script loads results from:
- pi0_long_context.csv: Performance data across different context lengths

Generates:
1. Log-log plot showing latency vs timesteps for different hardware
2. LaTeX table with E2E latency and frequency for each timestep

x-axis: timesteps (each timestep = 3 frames)
y-axis: E2E latency (ms)
Each curve represents a different hardware platform
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import numpy as np
from matplotlib.ticker import FuncFormatter

# Add parent directory to path
script_dir = Path(__file__).parent.resolve()
genz_systems_path = script_dir.parent.parent / "genz" / "Systems"
genz_systems_path = genz_systems_path.resolve()

if str(genz_systems_path) not in sys.path:
    sys.path.insert(0, str(genz_systems_path))

# Import plot utilities
from plot_util import rename_hardware

# Plot styling constants
LABEL_FONT = 16
TICK_FONT = 14
TEXT_FONT = 16
MARKERSIZE = 8
LEGEND_FONT = 14

# Style options
STYLE_OPTIONS = {
    'colorblind': 'seaborn-v0_8-colorblind',
    'deep': 'seaborn-v0_8-deep',
    'pastel': 'seaborn-v0_8-pastel',
    'ggplot': 'ggplot'
}


def load_long_context_results(
    csv_path: str = "../perf_results/pi0_long_context.csv"
) -> pd.DataFrame:
    """Load the long context performance results."""
    script_dir = Path(__file__).parent
    
    # Load performance data
    full_path = script_dir / csv_path
    df = pd.read_csv(full_path)
    
    return df


def format_axis_real_numbers(value, pos):
    """Format axis labels as real numbers (1, 10, 100) instead of scientific notation."""
    if value >= 1:
        return f'{int(value)}'
    else:
        return f'{value:.1f}'


def plot_long_context_latency(
    df: pd.DataFrame,
    output_dir: Path,
    hardware_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    style: str = 'colorblind'
) -> None:
    """
    Create a log-log plot of timesteps vs E2E latency.
    
    Args:
        df: Performance data
        output_dir: Directory to save the plot (should be paper_figures)
        hardware_list: List of hardware to plot
        style: Plot style ('colorblind', 'deep', 'pastel', 'ggplot')
    """
    # Set style
    if style in STYLE_OPTIONS:
        try:
            plt.style.use(STYLE_OPTIONS[style])
        except:
            print(f"Warning: Style {style} not available, using default")
    
    # Set up the plot - more square aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=((5.5, 3.5)))
    
    # Marker mapping (colors will come from style)
    markers = {
        "B100": "o",
        "RTX_4090": "s",
        "Jetson_AGX_Thor": "^"
    }
    
    # Plot each hardware
    for hw in hardware_list:
        hw_data = df[df['system'] == hw].sort_values('timesteps')
        
        if hw_data.empty:
            print(f"Warning: No data for hardware {hw}")
            continue
        
        timesteps = hw_data['timesteps'].values
        latencies = hw_data['e2e_ms'].values
        
        hw_display = rename_hardware(hw)
        
        ax.loglog(
            timesteps,
            latencies,
            marker=markers.get(hw, 'o'),
            linewidth=2,
            markersize=MARKERSIZE,
            label=hw_display
        )
    
    # ax.set_title('Long Context VLA Performance', fontsize=LABEL_FONT, pad=25)
    ax.set_xlabel('Timesteps', fontsize=LABEL_FONT)
    ax.set_ylabel('Latency / step (ms)', fontsize=LABEL_FONT)
    
    # Format axes with real numbers
    ax.xaxis.set_major_formatter(FuncFormatter(format_axis_real_numbers))
    ax.yaxis.set_major_formatter(FuncFormatter(format_axis_real_numbers))
    
    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    
    # Add horizontal reference lines at 10 ms and 100 ms for clarity
    ax.axhline(y=10, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Place legend on top with three columns
    ax.legend(fontsize=LEGEND_FONT, frameon=False, loc='upper center', 
              bbox_to_anchor=(0.42, 1.25), ncol=3)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "pi0_long_context.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also save as PDF for paper
    output_pdf = output_dir / "pi0_long_context.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    plt.close()


def generate_long_context_latex_table(
    df: pd.DataFrame,
    tables_dir: Path,
    hardware_list: list[str] = ["Jetson_AGX_Thor", "RTX_4090", "B100"]
) -> str:
    """
    Generate LaTeX table showing performance for different timesteps.
    
    Format: 
    - Rows: Timesteps
    - Columns: Timesteps | Total Memory (GB) | KV Cache (GB) | Hardware (ms + Hz in one cell)
    
    Args:
        df: Performance data
        tables_dir: Directory to save the table (paper_tables)
        hardware_list: List of hardware in desired order
    
    Returns:
        LaTeX table string
    """
    # Get unique timesteps sorted
    timestep_list = sorted(df['timesteps'].unique())
    
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("    \\caption{Inference performance and memory consumption of long-context VLA models.}")
    latex_lines.append("    \\label{tab:pi0_long_context}")
    latex_lines.append("    \\vspace{0.5em}")
    latex_lines.append("    \\scalebox{0.9}{")
    
    # Column format: Timesteps | Total Memory | KV Cache Size | Jetson Thor | RTX 4090 | B100
    col_format = "@{} L{4em} R{6em} R{6em} R{7em} R{7em} R{7em} @{}"
    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\toprule")
    
    # Single header row
    header = [
        "\\textbf{Timesteps}", 
        "\\multicolumn{1}{c}{\\textbf{Total Memory}}", 
        "\\multicolumn{1}{c}{\\textbf{KV Cache Size}}"
    ]
    for hw in hardware_list:
        hw_display = rename_hardware(hw)
        header.append(f"\\multicolumn{{1}}{{c}}{{\\textbf{{{hw_display}}}}}")
    
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows - one per timestep
    for i, timesteps in enumerate(timestep_list):
        row_cells = []
        
        # Timesteps column
        row_cells.append(f"{timesteps}")
        
        # Memory columns - get from first hardware (same across all hardware)
        first_hw_data = df[df['timesteps'] == timesteps]
        if not first_hw_data.empty:
            # Total memory
            total_memory_mb = first_hw_data.iloc[0]['total_memory_mb']
            total_memory_gb = total_memory_mb / 1024.0
            row_cells.append(f"{total_memory_gb:.1f} GB")
            
            # KV cache
            vlm_kv_cache_mb = first_hw_data.iloc[0]['vlm_kv_cache_mb']
            vlm_kv_cache_gb = vlm_kv_cache_mb / 1024.0
            row_cells.append(f"{vlm_kv_cache_gb:.2f} GB" if vlm_kv_cache_gb < 1.0 else f"{vlm_kv_cache_gb:.1f} GB")
        else:
            row_cells.append("N/A")
            row_cells.append("N/A")
        
        # Hardware performance - combine ms and Hz in one cell
        for hw in hardware_list:
            hw_data = df[(df['system'] == hw) & (df['timesteps'] == timesteps)]
            
            if hw_data.empty:
                row_cells.append("N/A")
            else:
                row = hw_data.iloc[0]
                e2e_ms = row['e2e_ms']
                freq_hz = 1000 / e2e_ms
                row_cells.append(f"\\cellcolor{{lightgray}}{e2e_ms:.1f} ms ({freq_hz:.1f} Hz)")
        
        # Use hdashline between rows, bottomrule at the end
        line_ending = " \\\\\n\\hdashline" if i < len(timestep_list) - 1 else " \\\\"
        latex_lines.append(" & ".join(row_cells) + line_ending)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")
    
    # Save to file
    output_file = tables_dir / "3_long_context.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table saved to: {output_file}")
    return '\n'.join(latex_lines)


def main(style: str = 'colorblind'):
    """
    Main function to generate plot and table.
    
    Args:
        style: Plot style ('colorblind', 'deep', 'pastel', 'ggplot')
               Default: 'colorblind'
    """
    print("="*80)
    print("Pi0 Long Context Visualization")
    print("="*80)
    print(f"Using plot style: {style}")
    
    # Load data
    print("\nLoading data...")
    df = load_long_context_results()
    
    print(f"\nFound {len(df['timesteps'].unique())} timestep configurations:")
    for ts in sorted(df['timesteps'].unique()):
        print(f"  - {ts} timesteps")
    
    print(f"\nFound {len(df['system'].unique())} hardware configurations:")
    for hw in df['system'].unique():
        hw_display = rename_hardware(hw)
        print(f"  - {hw_display}")
    
    # Create output directories
    script_dir = Path(__file__).parent
    figures_dir = script_dir / "../paper_figures"
    tables_dir = script_dir / "../paper_tables"
    
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directories:")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Tables:  {tables_dir}")
    
    # Hardware to plot
    hardware_list = ["B100", "RTX_4090", "Jetson_AGX_Thor"]
    
    # Generate plot
    print("\n" + "="*80)
    print("Generating Long Context Plot")
    print("="*80)
    plot_long_context_latency(
        df=df,
        output_dir=figures_dir,
        hardware_list=hardware_list,
        style=style
    )
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("Generating LaTeX Table")
    print("="*80)
    table = generate_long_context_latex_table(df, tables_dir)
    print("\n" + table)
    
    print("\n" + "="*80)
    print("Done! Outputs saved to:")
    print(f"  - Figures: {figures_dir}")
    print(f"    * pi0_long_context.png/pdf")
    print(f"  - Tables: {tables_dir}")
    print(f"    * 3_long_context.tex")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Pi0 long context plots and tables')
    parser.add_argument(
        '--style',
        type=str,
        choices=['colorblind', 'deep', 'pastel', 'ggplot'],
        default='colorblind',
        help='Plot style (default: colorblind)'
    )
    
    args = parser.parse_args()
    main(style=args.style)
