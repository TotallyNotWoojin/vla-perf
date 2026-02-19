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
Generate plots for Pi0 denoising steps and action chunk size comparison.

This script loads results from pi0_denoising_steps_action_lengths.csv 
and generates figures for B100:
1. Heatmap of absolute E2E latency vs denoising steps and action chunk size
2. Heatmap of relative E2E latency (normalized to 10 steps, 50 chunk size)
3. Heatmap of absolute diffusion model latency vs denoising steps and action chunk size
4. Heatmap of relative diffusion model latency (normalized to 10 steps, 50 chunk size)
5. Line plot of operator intensity (OI) vs action chunk size with B100 balanced OI line

Figures are saved to ../paper_figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from matplotlib.colors import LogNorm

# Add parent directory to path to import system_configs
script_dir = Path(__file__).parent.resolve()
genz_systems_path = script_dir.parent.parent / "genz" / "Systems"
genz_systems_path = genz_systems_path.resolve()

if str(genz_systems_path) not in sys.path:
    sys.path.insert(0, str(genz_systems_path))

from system_configs import system_configs
from plot_util import rename_hardware

# Plot styling constants
LABEL_FONT = 16
TICK_FONT = 14
TEXT_FONT = 14
LEGEND_FONT = 14
TITLE_FONT = 16
HEATMAP_ANNOTATION_FONT = 11  # Font size for text inside heatmap cells

# Colormap options for heatmaps (will rotate through these)
CMAP_OPTIONS = ['summer', 'winter', 'Wistia', 'cool', 'coolwarm', 'bwr', 'seismic']
# CMAP_OPTIONS = ['RdBu_r', 'RdYlGn_r', 'coolwarm', 'bwr', 'seismic']

# Default normalization point for relative plots
DEFAULT_DENOISING_STEPS = 10
DEFAULT_ACTION_CHUNK_SIZE = 50

# Title display option
SHOW_TITLES = False  # Set to True to enable titles on plots


def load_results(csv_path: str = "../perf_results/pi0_denoising_steps_action_lengths.csv") -> pd.DataFrame:
    """Load the denoising steps and action chunk size results from CSV."""
    script_dir = Path(__file__).parent
    full_path = script_dir / csv_path
    df = pd.read_csv(full_path)
    return df


def get_text_color_for_background(colormap, norm_value, threshold=0.5):
    """
    Determine if text should be black or white based on background color luminance.
    
    Uses relative luminance formula: L = 0.299*R + 0.587*G + 0.114*B
    Returns 'white' for dark backgrounds, 'black' for light backgrounds.
    
    Args:
        colormap: Matplotlib colormap object
        norm_value: Normalized value (0 to 1) for the colormap
        threshold: Luminance threshold (default 0.5)
    
    Returns:
        'white' or 'black'
    """
    # Get RGBA color from colormap
    rgba = colormap(norm_value)
    
    # Calculate relative luminance (perceived brightness)
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    
    # Return white text for dark backgrounds, black for light
    return 'white' if luminance < threshold else 'black'


def get_hardware_balanced_oi(hw_name: str) -> float:
    """
    Calculate hardware balanced operational intensity (OI).
    
    Balanced OI = Peak FLOPS / Memory Bandwidth
                = (TFLOPS * 10^12) / (Memory_BW GB/s * 10^9)
                = TFLOPS * 1000 / Memory_BW
    
    Uses bf16 precision, falling back to fp16 if bf16 not available.
    
    Args:
        hw_name: Hardware configuration name
    
    Returns:
        Hardware balanced OI in FLOP/Byte
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
    
    # Calculate balanced OI: FLOP/Byte
    balanced_oi = tflops * 1000 / memory_bw
    return balanced_oi


def plot_heatmap(df: pd.DataFrame, system: str, metric: str, output_dir: Path, 
                 relative: bool = False, baseline_steps: int = DEFAULT_DENOISING_STEPS,
                 baseline_chunk: int = DEFAULT_ACTION_CHUNK_SIZE, cmap: str = 'RdBu_r',
                 show_title: bool = SHOW_TITLES):
    """
    Create a heatmap for a specific metric.
    
    Args:
        df: DataFrame with results
        system: System name to plot
        metric: Metric column name (e.g., 'e2e_ms' or 'action_ms')
        output_dir: Directory to save the figure
        relative: If True, show relative values normalized to baseline
        baseline_steps: Baseline denoising steps for normalization
        baseline_chunk: Baseline action chunk size for normalization
        cmap: Colormap to use
        show_title: If True, display title on plot (default: SHOW_TITLES)
    """
    # Filter data for this system
    df_system = df[df['system'] == system].copy()
    
    if df_system.empty:
        print(f"No data found for system {system}")
        return
    
    # Create pivot table for heatmap (rows=denoising_steps, cols=action_chunk_size)
    pivot_data = df_system.pivot(
        index='denoising_steps',
        columns='action_chunk_size',
        values=metric
    )
    
    # Store original values for text annotations
    original_values = pivot_data.copy()
    
    # Normalize if relative
    if relative:
        baseline_value = pivot_data.loc[baseline_steps, baseline_chunk]
        pivot_data = pivot_data / baseline_value
    
    # Create figure with smaller size (roughly half)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Create heatmap with log scale for absolute plots
    if relative:
        im = ax.imshow(pivot_data.values, aspect='auto', cmap=cmap, origin='lower')
    else:
        # Use log scale for colorbar
        im = ax.imshow(pivot_data.values, aspect='auto', cmap=cmap, origin='lower', 
                      norm=LogNorm(vmin=pivot_data.values.min(), vmax=pivot_data.values.max()))
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns, fontsize=TICK_FONT)
    ax.set_yticklabels(pivot_data.index, fontsize=TICK_FONT)
    
    # Labels
    ax.set_xlabel('Action Chunk Size', fontsize=LABEL_FONT)
    ax.set_ylabel('Denoising Steps', fontsize=LABEL_FONT)
    
    # Title (optional)
    if show_title:
        metric_name = "VLA Total Latency" if metric == "e2e_ms" else "Action Expert Latency"
        title = f"{metric_name} ({rename_hardware(system)})"
        ax.set_title(title, fontsize=TITLE_FONT, pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if relative:
        cbar.set_label('Relative Latency', fontsize=LABEL_FONT)
    else:
        cbar.set_label('Latency (ms)', fontsize=LABEL_FONT)
    cbar.ax.tick_params(labelsize=TICK_FONT)
    
    # Get colormap object for text color selection
    cmap_obj = plt.cm.get_cmap(cmap)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if relative:
                # For relative, show the normalized ratio
                value = pivot_data.values[i, j]
                # Normalize value for colormap (relative values typically range around 0.5 to 2.0)
                vmin, vmax = pivot_data.values.min(), pivot_data.values.max()
                norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = get_text_color_for_background(cmap_obj, norm_value)
                ax.text(j, i, f'{value:.2f}x',
                       ha='center', va='center', color=text_color, fontsize=HEATMAP_ANNOTATION_FONT,
                       weight='bold')
            else:
                # For absolute, show the actual latency value
                value = original_values.values[i, j]
                # For absolute with log scale, normalize in log space
                log_val = np.log(value)
                log_min = np.log(pivot_data.values.min())
                log_max = np.log(pivot_data.values.max())
                norm_value = (log_val - log_min) / (log_max - log_min) if log_max > log_min else 0.5
                text_color = get_text_color_for_background(cmap_obj, norm_value)
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center', color=text_color, fontsize=HEATMAP_ANNOTATION_FONT,
                       weight='bold')
    
    plt.tight_layout()
    
    # Save figure in both PNG and PDF
    metric_suffix = "e2e" if metric == "e2e_ms" else "diffusion"
    rel_suffix = "_relative" if relative else "_absolute"
    base_filename = f"pi0_denoise_steps_action_lengths_{metric_suffix}{rel_suffix}_{system}"
    
    # Save PNG
    output_file_png = output_dir / f"{base_filename}.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_file_png}")
    
    # Save PDF
    output_file_pdf = output_dir / f"{base_filename}.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved heatmap to {output_file_pdf}")
    
    plt.close()


def plot_oi_vs_action_chunk_size(df: pd.DataFrame, output_dir: Path, system: str = 'B100',
                                  show_title: bool = SHOW_TITLES):
    """
    Create a line plot of operator intensity vs action chunk size.
    
    Shows OI for different action chunk sizes with a dashed horizontal line
    indicating the balanced OI for B100, annotated with an arrow and text box.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save the figure
        system: System to show balanced OI for (default: 'B100')
        show_title: If True, display title on plot (default: SHOW_TITLES)
    """
    # Get unique action chunk sizes
    action_chunk_sizes = sorted(df['action_chunk_size'].unique())
    
    # Get OI values (they should be the same across all systems and denoising steps for a given action chunk size)
    # We'll take one denoising step value for simplicity (OI doesn't depend on denoising steps)
    df_oi = df[df['denoising_steps'] == df['denoising_steps'].min()].copy()
    
    # Get unique OI values per action chunk size
    oi_values = []
    for chunk_size in action_chunk_sizes:
        oi = df_oi[df_oi['action_chunk_size'] == chunk_size]['action_op_intensity'].iloc[0]
        oi_values.append(oi)
    
    # Get balanced OI for B100
    balanced_oi = get_hardware_balanced_oi(system)
    
    # Create figure with smaller size (roughly half)
    fig, ax = plt.subplots(figsize=(3, 3.5))
    # Plot horizontal dashed line for balanced OI
    ax.axhline(y=balanced_oi, color='grey', linestyle='--', linewidth=2.5, zorder=2)
    # Plot OI vs action chunk size
    ax.plot(action_chunk_sizes, oi_values, marker='o', linewidth=2.5, markersize=9,
            label='Diffusion Model OI', color='#1f77b4', zorder=3)
    
    # Add annotation with arrow pointing to the balanced OI line
    # Position the text box to the right side
    annotation_x = action_chunk_sizes[-1] * 0.4  # Position at 70% of x-axis
    annotation_y = balanced_oi * 0.75  # Position above the line
    
    ax.annotate(f'Balanced OI\n= {balanced_oi:.1f} ({rename_hardware(system)})',
                xy=(annotation_x, balanced_oi),  # Point to the line
                xytext=(annotation_x, annotation_y),  # Text position
                fontsize=TEXT_FONT,
                # color='#d62728',
                ha='center',
                va='bottom',
                # bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#d62728', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color='black', linewidth=2, shrinkA=0, shrinkB=0))
    
    # Labels and title
    ax.set_xlabel('Action Chunk Size', fontsize=LABEL_FONT)
    ax.set_ylabel('OI (FLOPs/Byte)', fontsize=LABEL_FONT)
    if show_title:
        ax.set_title('Action Expert Operator Intensity (OI)', fontsize=TITLE_FONT, pad=15)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, zorder=1)
    # ax.legend(fontsize=LEGEND_FONT, loc='upper left', framealpha=0.9)
    
    # Set x-axis to show all action chunk sizes
    ax.set_xticks(action_chunk_sizes)
    ax.set_xticklabels([5, "", 10, 50, 100], fontsize=TICK_FONT)
    # ax.set_xticklabels(action_chunk_sizes, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    
    plt.tight_layout()
    
    # Save figure in both PNG and PDF
    output_file_png = output_dir / "pi0_denoise_action_oi.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"Saved OI plot to {output_file_png}")
    
    output_file_pdf = output_dir / "pi0_denoise_action_oi.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved OI plot to {output_file_pdf}")
    
    plt.close()


def generate_all_plots(
    csv_path: str = "../perf_results/pi0_denoising_steps_action_lengths.csv",
    output_dir: str = "../paper_figures",
    system: str = "B100",
    show_titles: bool = SHOW_TITLES
):
    """
    Generate all plots for B100:
    1. Absolute E2E latency heatmap
    2. Relative E2E latency heatmap (normalized to 10 steps, 50 chunk size)
    3. Absolute diffusion model latency heatmap
    4. Relative diffusion model latency heatmap (normalized to 10 steps, 50 chunk size)
    5. OI vs action chunk size with B100 balanced OI line
    
    Each heatmap uses a different colormap from rotation.
    All plots saved in both PNG and PDF formats.
    
    Args:
        csv_path: Path to the CSV file with results
        output_dir: Directory to save the figures
        system: System to generate plots for (default: 'B100')
        show_titles: If True, display titles on all plots (default: SHOW_TITLES)
    """
    # Load data
    df = load_results(csv_path)
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Generating plots...")
    print(f"System: {system}")
    print(f"Colormaps rotating through: {CMAP_OPTIONS}")
    print(f"Baseline: {DEFAULT_DENOISING_STEPS} steps, {DEFAULT_ACTION_CHUNK_SIZE} chunk size")
    print(f"Show titles: {show_titles}")
    
    # Generate heatmaps for B100 with different colormaps
    print(f"\nGenerating heatmaps for {system}...")
    
    # E2E latency heatmaps
    print(f"  - E2E latency (absolute) with colormap {CMAP_OPTIONS[2]}...")
    plot_heatmap(df, system, 'e2e_ms', output_path, relative=False, cmap=CMAP_OPTIONS[2],
                show_title=show_titles)
    
    print(f"  - E2E latency (relative) with colormap {CMAP_OPTIONS[1]}...")
    plot_heatmap(df, system, 'e2e_ms', output_path, relative=True, 
                baseline_steps=DEFAULT_DENOISING_STEPS, 
                baseline_chunk=DEFAULT_ACTION_CHUNK_SIZE, cmap=CMAP_OPTIONS[1],
                show_title=show_titles)
    
    # Diffusion model latency heatmaps
    print(f"  - Diffusion model latency (absolute) with colormap {CMAP_OPTIONS[3]}...")
    plot_heatmap(df, system, 'action_ms', output_path, relative=False, cmap=CMAP_OPTIONS[3],
                show_title=show_titles)
    
    print(f"  - Diffusion model latency (relative) with colormap {CMAP_OPTIONS[0]}...")
    plot_heatmap(df, system, 'action_ms', output_path, relative=True,
                baseline_steps=DEFAULT_DENOISING_STEPS,
                baseline_chunk=DEFAULT_ACTION_CHUNK_SIZE, cmap=CMAP_OPTIONS[0],
                show_title=show_titles)
    
    # Generate OI plot
    print("\nGenerating OI plot...")
    plot_oi_vs_action_chunk_size(df, output_path, system=system, show_title=show_titles)
    
    print("\nAll plots generated successfully!")
    print(f"Figures saved to: {output_path}")


if __name__ == "__main__":
    generate_all_plots()
