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
Generate plots for Autoregressive vs Diffusion comparison.

This script loads results from pi0_autoregressive_vs_diffusion.csv 
and generates two figures comparing four solutions:
1. Diffusion (Pi-0 action expert)
2. Diffusion-Large (VLM-sized DiT)
3. Autoregressive (sequential decoding)
4. Autoregressive-Parallel (parallel decoding)

Figure 1: E2E latency vs action chunk size (DoF=14)
Figure 2: E2E latency vs DoF (chunk_size=1)

Figures are saved to ../paper_figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

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

# Title display option
SHOW_TITLES = False  # Set to True to enable titles on plots

# Default values
DEFAULT_DOF = 14
DEFAULT_CHUNK_SIZE = 1
DEFAULT_DENOISING_STEPS = 10  # For diffusion models

# Color scheme for the four solutions
COLORS = {
    'Diffusion': '#2ca02c',      # green
    'Diffusion-Large': '#ff7f0e',      # orange
    'Autoregressive': '#1f77b4',       # blue
    'Autoregressive-Parallel': '#d62728'  # red
}

# Marker styles
MARKERS = {
    'Diffusion': 'o',
    'Diffusion-Large': 's',
    'Autoregressive': '^',
    'Autoregressive-Parallel': 'D'
}


def load_results(csv_path: str = "../perf_results/pi0_autoregressive_vs_diffusion.csv") -> pd.DataFrame:
    """Load the autoregressive vs diffusion results from CSV."""
    script_dir = Path(__file__).parent
    full_path = script_dir / csv_path
    df = pd.read_csv(full_path)
    return df


def plot_latency_vs_chunk_size(
    df: pd.DataFrame, 
    system: str, 
    output_dir: Path,
    dof: int = DEFAULT_DOF,
    denoising_steps: int = DEFAULT_DENOISING_STEPS,
    show_title: bool = SHOW_TITLES
):
    """
    Plot E2E latency vs action chunk size for all four solutions.
    
    Compares:
    - Diffusion (B: Small Diffusion)
    - Diffusion-Large (C: Large Diffusion)
    - Autoregressive (A: Autoregressive)
    - Autoregressive-Parallel (D: Autoregressive Parallel)
    
    Args:
        df: DataFrame with results
        system: System name to plot
        output_dir: Directory to save the figure
        dof: DoF value to use (default: 14)
        denoising_steps: Number of denoising steps for diffusion models
        show_title: If True, display title on plot
    """
    # Filter data for this system and DoF
    df_system = df[(df['system'] == system) & (df['dof'] == dof)].copy()
    
    if df_system.empty:
        print(f"No data found for system {system} with DoF {dof}")
        return
    
    # Get unique chunk sizes, sorted
    chunk_sizes = sorted(df_system['action_chunk_size'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot each solution
    solutions = {
        'Autoregressive': ('A: Autoregressive', 'N/A'),
        'Diffusion': ('B: Small Diffusion', denoising_steps),
        'Autoregressive-Parallel': ('D: Autoregressive Parallel', 'N/A'),
        'Diffusion-Large': ('C: Large Diffusion', denoising_steps),
    }
    
    for solution_name, (setup_name, steps) in solutions.items():
        # Filter data for this solution
        if steps == 'N/A':
            solution_df = df_system[
                (df_system['setup'] == setup_name)
            ]
        else:
            solution_df = df_system[
                (df_system['setup'] == setup_name) & 
                (df_system['denoising_steps'] == steps)
            ]
        
        if solution_df.empty:
            print(f"  Warning: No data for {solution_name}")
            continue
        
        # Extract latency for each chunk size
        latencies = []
        for chunk_size in chunk_sizes:
            chunk_df = solution_df[solution_df['action_chunk_size'] == chunk_size]
            if not chunk_df.empty:
                latencies.append(chunk_df['e2e_ms'].values[0])
            else:
                latencies.append(np.nan)
        
        # Plot line
        ax.plot(chunk_sizes, latencies, 
                marker=MARKERS[solution_name], 
                linewidth=2.5, 
                markersize=9,
                label=solution_name, 
                color=COLORS[solution_name])
    
    # Labels and formatting
    ax.set_xlabel('Action Chunk Size', fontsize=LABEL_FONT)
    ax.set_ylabel('VLA Total Latency (ms)', fontsize=LABEL_FONT)
    if show_title:
        ax.set_title(f'E2E Latency vs Action Chunk Size ({rename_hardware(system)}, DoF={dof})', 
                    fontsize=TITLE_FONT, pad=15)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONT, loc='best', framealpha=0.9, frameon=False)
    
    # Set x-axis to show all chunk sizes
    ax.set_xticks(chunk_sizes)
    ax.set_xticklabels(chunk_sizes, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    
    # Use log scale for y-axis
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save figure in both PNG and PDF
    base_filename = f"pi0_autoregressive_vs_diffusion_chunk_size_{system}"
    
    output_file_png = output_dir / f"{base_filename}.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file_png}")
    
    output_file_pdf = output_dir / f"{base_filename}.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved plot to {output_file_pdf}")
    
    plt.close()


def plot_latency_vs_dof(
    df: pd.DataFrame, 
    system: str, 
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    denoising_steps: int = DEFAULT_DENOISING_STEPS,
    show_title: bool = SHOW_TITLES
):
    """
    Plot E2E latency vs DoF for all four solutions with chunk_size=1.
    
    Compares:
    - Diffusion (B: Small Diffusion)
    - Diffusion-Large (C: Large Diffusion)
    - Autoregressive (A: Autoregressive)
    - Autoregressive-Parallel (D: Autoregressive Parallel)
    
    Args:
        df: DataFrame with results
        system: System name to plot
        output_dir: Directory to save the figure
        chunk_size: Chunk size to use (default: 1)
        denoising_steps: Number of denoising steps for diffusion models
        show_title: If True, display title on plot
    """
    # Filter data for this system and chunk size
    df_system = df[(df['system'] == system) & (df['action_chunk_size'] == chunk_size)].copy()
    
    if df_system.empty:
        print(f"No data found for system {system} with chunk_size {chunk_size}")
        return
    
    # Get unique DoF values, sorted
    dof_values = sorted(df_system['dof'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot each solution
    # All solutions now have DoF variation data with chunk_size=1
    
    solutions = {
        'Autoregressive': ('A: Autoregressive (DoF Comparison)', 'N/A'),
        'Diffusion': ('B: Small Diffusion (DoF Comparison)', denoising_steps),
        'Autoregressive-Parallel': ('D: Autoregressive Parallel (DoF Comparison)', 'N/A'),
        'Diffusion-Large': ('C: Large Diffusion (DoF Comparison)', denoising_steps),
    }
    
    for solution_name, (setup_name, steps) in solutions.items():
        # Filter data for this solution
        if steps == 'N/A':
            solution_df = df_system[df_system['setup'] == setup_name]
        else:
            solution_df = df_system[
                (df_system['setup'] == setup_name) & 
                (df_system['denoising_steps'] == steps)
            ]
        
        if solution_df.empty:
            print(f"  Warning: No data for {solution_name}")
            continue
        
        # Extract latency for each DoF
        latencies = []
        available_dofs = []
        for dof in dof_values:
            dof_df = solution_df[solution_df['dof'] == dof]
            if not dof_df.empty:
                latencies.append(dof_df['e2e_ms'].values[0])
                available_dofs.append(dof)
        
        # Plot line
        if latencies:
            ax.plot(available_dofs, latencies, 
                    marker=MARKERS[solution_name], 
                    linewidth=2.5, 
                    markersize=9,
                    label=solution_name, 
                    color=COLORS[solution_name])
    
    # Labels and formatting
    ax.set_xlabel('Degrees of Freedom (DoF)', fontsize=LABEL_FONT)
    ax.set_ylabel('VLA Total Latency (ms)', fontsize=LABEL_FONT)
    if show_title:
        ax.set_title(f'E2E Latency vs DoF ({rename_hardware(system)}, Chunk Size={chunk_size})', 
                    fontsize=TITLE_FONT, pad=15)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONT, loc='best', framealpha=0.9, frameon=False)
    
    # Set x-axis to show all DoF values
    ax.set_xticks(dof_values)
    ax.set_xticklabels(dof_values, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    
    # Use log scale if range is large
    all_latencies = []
    for lats in [latencies]:
        all_latencies.extend([lat for lat in lats if not np.isnan(lat)])
    if all_latencies and max(all_latencies) / min(all_latencies) > 10:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure in both PNG and PDF
    base_filename = f"pi0_autoregressive_vs_diffusion_dof_{system}"
    
    output_file_png = output_dir / f"{base_filename}.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file_png}")
    
    output_file_pdf = output_dir / f"{base_filename}.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved plot to {output_file_pdf}")
    
    plt.close()


def generate_all_plots(
    csv_path: str = "../perf_results/pi0_autoregressive_vs_diffusion.csv",
    output_dir: str = "../paper_figures",
    systems: list = None,
    show_titles: bool = SHOW_TITLES
):
    """
    Generate all plots for autoregressive vs diffusion comparison:
    1. E2E latency vs action chunk size (DoF=14)
    2. E2E latency vs DoF (chunk_size=1)
    
    Args:
        csv_path: Path to the CSV file with results
        output_dir: Directory to save the figures
        systems: List of systems to generate plots for (default: all systems in data)
        show_titles: If True, display titles on all plots (default: SHOW_TITLES)
    """
    # Load data
    df = load_results(csv_path)
    
    # Get unique systems if not specified
    if systems is None:
        systems = sorted(df['system'].unique())
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Generating autoregressive vs diffusion comparison plots...")
    print(f"Systems: {systems}")
    print(f"Show titles: {show_titles}")
    
    # Generate plots for each system
    for system in systems:
        print(f"\nGenerating plots for {system}...")
        
        # Figure 1: Latency vs chunk size (DoF=14)
        print(f"  - E2E latency vs action chunk size (DoF={DEFAULT_DOF})...")
        plot_latency_vs_chunk_size(df, system, output_path, 
                                   dof=DEFAULT_DOF, 
                                   denoising_steps=DEFAULT_DENOISING_STEPS,
                                   show_title=show_titles)
        
        # Figure 2: Latency vs DoF (chunk_size=1)
        print(f"  - E2E latency vs DoF (chunk_size={DEFAULT_CHUNK_SIZE})...")
        plot_latency_vs_dof(df, system, output_path, 
                           chunk_size=DEFAULT_CHUNK_SIZE,
                           denoising_steps=DEFAULT_DENOISING_STEPS,
                           show_title=show_titles)
    
    print("\nAll plots generated successfully!")
    print(f"Figures saved to: {output_path}")


if __name__ == "__main__":
    # Generate plots for B100 by default
    generate_all_plots(systems=["B100"])
