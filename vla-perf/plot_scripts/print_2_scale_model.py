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
Generate plots and LaTeX tables for Pi0 model size scaling results.

This script loads results from:
- pi0_model_size_scaling.csv: Performance data
- pi0_model_total_params.csv: Total model parameter counts
- pi0_model_component_params.csv: Component-specific parameter counts

Generates:
1. Three individual component plots:
   - Vision: Vision encoder size vs vision latency
   - VLM: VLM size vs VLM latency
   - Action: Action expert size vs action latency
2. One combined figure with three subplots for the above three components
3. One separate plot for E2E: Total model size vs end-to-end latency
4. LaTeX table with E2E frequency (Hz) for each model variant

Note: Component plots show only unique component sizes (e.g., if multiple models
share the same vision encoder, there will be only one point per hardware).
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
LEGEND_FONT = 16

# Style options
STYLE_OPTIONS = {
    'colorblind': 'seaborn-v0_8-colorblind',
    'deep': 'seaborn-v0_8-deep',
    'pastel': 'seaborn-v0_8-pastel',
    'ggplot': 'ggplot'
}


def load_scaling_results(
    perf_csv_path: str = "../perf_results/pi0_model_size_scaling.csv",
    params_csv_path: str = "../perf_results/pi0_model_total_params.csv",
    component_params_csv_path: str = "../perf_results/pi0_model_component_params.csv"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the model scaling performance and parameter count results."""
    script_dir = Path(__file__).parent
    
    # Load performance data
    perf_full_path = script_dir / perf_csv_path
    df_perf = pd.read_csv(perf_full_path)
    
    # Load total parameter count data
    params_full_path = script_dir / params_csv_path
    df_params = pd.read_csv(params_full_path)
    
    # Load component parameter count data
    component_params_full_path = script_dir / component_params_csv_path
    df_component_params = pd.read_csv(component_params_full_path)
    
    # Filter for Batch=1, Chips=1
    df_perf_filtered = df_perf[
        (df_perf['batch_size'] == 1) & 
        (df_perf['hardware.num_chips'] == 1)
    ]
    
    return df_perf_filtered, df_params, df_component_params


def format_axis_real_numbers(value, pos):
    """Format axis labels as real numbers (1, 10, 100) instead of scientific notation."""
    if value >= 1:
        return f'{int(value)}'
    else:
        return f'{value:.1f}'


def plot_component_latency(
    df_perf: pd.DataFrame,
    df_params: pd.DataFrame,
    df_component_params: pd.DataFrame,
    output_dir: Path,
    component: str,
    hardware_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    style: str = 'colorblind'
) -> None:
    """
    Create a log-log plot of model size vs component latency.
    
    For component plots (vision, vlm, action): x-axis is component's own size
    For E2E plot: x-axis is total model size
    
    Args:
        df_perf: Performance data
        df_params: Total parameter counts
        df_component_params: Component-specific parameter counts
        output_dir: Directory to save the plot (should be paper_figures)
        component: Component name ('vision', 'vlm', 'action', 'e2e')
        hardware_list: List of hardware to plot
        style: Plot style ('colorblind', 'deep', 'pastel', 'ggplot')
    """
    # Set style
    if style in STYLE_OPTIONS:
        try:
            plt.style.use(STYLE_OPTIONS[style])
        except:
            print(f"Warning: Style {style} not available, using default")
    
    # For E2E, use total params; for components, use component params
    if component == 'e2e':
        # Merge performance with total parameter data
        df_merged = df_perf.merge(df_params, left_on='model.name', right_on='model')
        df_merged['params_B'] = df_merged['total_params_M'] / 1000.0
        size_label = 'Total Model Size (B parameters)'
    else:
        # Merge performance with component parameter data
        df_merged = df_perf.merge(df_component_params, left_on='model.name', right_on='model')
        
        # Map component to its parameter column
        param_col_map = {
            'vision': 'vision_params_M',
            'vlm': 'vlm_params_M',
            'action': 'action_params_M'
        }
        
        if component not in param_col_map:
            raise ValueError(f"Unknown component: {component}")
        
        # Convert M to B for x-axis
        df_merged['params_B'] = df_merged[param_col_map[component]] / 1000.0
        
        # Component-specific size label
        size_label_map = {
            'vision': 'Vision Encoder Size (B parameters)',
            'vlm': 'VLM Size (B parameters)',
            'action': 'Action Expert Size (B parameters)'
        }
        size_label = size_label_map[component]
    
    # Set up the plot - more square aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=((5.5, 3)))
    
    # Marker mapping (colors will come from style)
    markers = {
        "B100": "o",
        "RTX_4090": "s",
        "Jetson_AGX_Thor": "^"
    }
    
    # Component column mapping
    component_cols = {
        'vision': 'vision_time_ms',
        'vlm': 'vlm_time_ms',
        'action': 'action_time_ms',
        'e2e': 'e2e_time_ms'
    }
    
    if component not in component_cols:
        raise ValueError(f"Unknown component: {component}")
    
    time_col = component_cols[component]
    
    # Plot each hardware
    for hw in hardware_list:
        hw_data = df_merged[df_merged['hardware.name'] == hw].sort_values('params_B')
        
        if hw_data.empty:
            print(f"Warning: No data for hardware {hw}")
            continue
        
        # Group by component size to handle duplicate sizes
        # (e.g., same vision encoder used in multiple models)
        if component != 'e2e':
            # For component plots, group by component size and take mean latency
            hw_data_grouped = hw_data.groupby('params_B', as_index=False).agg({
                time_col: 'mean'  # Average latency for same component size
            })
            model_sizes_b = hw_data_grouped['params_B'].values
            latencies = hw_data_grouped[time_col].values
        else:
            # For E2E, each model size is unique
            model_sizes_b = hw_data['params_B'].values
            latencies = hw_data[time_col].values
        
        hw_display = rename_hardware(hw)
        
        ax.loglog(
            model_sizes_b,
            latencies,
            marker=markers.get(hw, 'o'),
            linewidth=2,
            markersize=MARKERSIZE,
            label=hw_display
        )
    
    # Component title mapping for plot titles
    plot_titles = {
        'vision': 'Scale up: Vision Encoder',
        'vlm': 'Scale up: VLM',
        'action': 'Scale up: Action Expert',
        'e2e': 'Scale up: Entire VLA'
    }
    
    ax.set_title(plot_titles[component], fontsize=LABEL_FONT, pad=25)
    ax.set_xlabel('Model Size (B params)', fontsize=LABEL_FONT)
    ax.set_ylabel('Latency (ms)', fontsize=LABEL_FONT)
    
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
    output_file = output_dir / f"pi0_model_scaling_{component}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also save as PDF for paper
    output_pdf = output_dir / f"pi0_model_scaling_{component}.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    plt.close()


def plot_three_components_combined(
    df_perf: pd.DataFrame,
    df_params: pd.DataFrame,
    df_component_params: pd.DataFrame,
    output_dir: Path,
    hardware_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    style: str = 'colorblind'
) -> None:
    """
    Create a single figure with three subplots for vision, VLM, and action components.
    
    Args:
        df_perf: Performance data
        df_params: Total parameter counts
        df_component_params: Component-specific parameter counts
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
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Marker mapping (colors will come from style)
    markers = {
        "B100": "o",
        "RTX_4090": "s",
        "Jetson_AGX_Thor": "^"
    }
    
    # Component configurations
    components_info = [
        {
            'name': 'vision',
            'param_col': 'vision_params_M',
            'time_col': 'vision_time_ms',
            'title': '(a) Vision Encoder',
            'size_label': 'Vision Encoder Size (B parameters)'
        },
        {
            'name': 'vlm',
            'param_col': 'vlm_params_M',
            'time_col': 'vlm_time_ms',
            'title': '(b) VLM',
            'size_label': 'VLM Size (B parameters)'
        },
        {
            'name': 'action',
            'param_col': 'action_params_M',
            'time_col': 'action_time_ms',
            'title': '(c) Action Expert',
            'size_label': 'Action Expert Size (B parameters)'
        }
    ]
    
    # Plot each component
    for idx, comp_info in enumerate(components_info):
        ax = axes[idx]
        component = comp_info['name']
        
        # Merge performance with component parameter data
        df_merged = df_perf.merge(df_component_params, left_on='model.name', right_on='model')
        
        # Convert M to B for x-axis
        df_merged['params_B'] = df_merged[comp_info['param_col']] / 1000.0
        
        # Plot each hardware
        for hw in hardware_list:
            hw_data = df_merged[df_merged['hardware.name'] == hw].sort_values('params_B')
            
            if hw_data.empty:
                print(f"Warning: No data for hardware {hw}")
                continue
            
            # Group by component size and take mean latency
            hw_data_grouped = hw_data.groupby('params_B', as_index=False).agg({
                comp_info['time_col']: 'mean'
            })
            model_sizes_b = hw_data_grouped['params_B'].values
            latencies = hw_data_grouped[comp_info['time_col']].values
            
            hw_display = rename_hardware(hw)
            
            ax.loglog(
                model_sizes_b,
                latencies,
                marker=markers.get(hw, 'o'),
                linewidth=2,
                markersize=MARKERSIZE,
                label=hw_display
            )
        
        ax.set_xlabel('Model Size (B params)', fontsize=LABEL_FONT)
        
        # Add subplot label below the x-axis
        ax.text(0.5, -0.4, comp_info['title'], fontsize=LABEL_FONT, 
                ha='center', va='top', transform=ax.transAxes)
        
        # Only show ylabel on the leftmost subplot
        if idx == 0:
            ax.set_ylabel('Latency (ms)', fontsize=LABEL_FONT)
        
        # Format axes with real numbers
        ax.xaxis.set_major_formatter(FuncFormatter(format_axis_real_numbers))
        ax.yaxis.set_major_formatter(FuncFormatter(format_axis_real_numbers))
        
        # Set tick font size
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT)
        
        # Add horizontal reference lines at 10 ms and 100 ms for clarity
        ax.axhline(y=10, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
        
        ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add a single legend for the entire figure at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=LEGEND_FONT, frameon=False, 
               loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"pi0_model_scaling_components.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined components plot saved to: {output_file}")
    
    # Also save as PDF for paper
    output_pdf = output_dir / f"pi0_model_scaling_components.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    plt.close()


def plot_all_components(
    df_perf: pd.DataFrame,
    df_params: pd.DataFrame,
    df_component_params: pd.DataFrame,
    figures_dir: Path,
    hardware_list: list[str] = ["B100", "RTX_4090", "Jetson_AGX_Thor"],
    style: str = 'colorblind'
) -> None:
    """
    Create plots for all components (individual plots + combined subplots, E2E separate).
    
    Args:
        df_perf: Performance data
        df_params: Total parameter counts
        df_component_params: Component-specific parameter counts
        figures_dir: Directory to save plots (paper_figures)
        hardware_list: List of hardware to plot
        style: Plot style
    """
    # Create individual plots for vision, VLM, action
    individual_components = ['vision', 'vlm', 'action']
    for component in individual_components:
        print(f"\nGenerating {component.upper()} plot...")
        plot_component_latency(
            df_perf=df_perf,
            df_params=df_params,
            df_component_params=df_component_params,
            output_dir=figures_dir,
            component=component,
            hardware_list=hardware_list,
            style=style
        )
    
    # Create combined plot for vision, VLM, action
    print(f"\nGenerating combined components plot (Vision, VLM, Action)...")
    plot_three_components_combined(
        df_perf=df_perf,
        df_params=df_params,
        df_component_params=df_component_params,
        output_dir=figures_dir,
        hardware_list=hardware_list,
        style=style
    )
    
    # Create separate E2E plot
    print(f"\nGenerating E2E plot...")
    plot_component_latency(
        df_perf=df_perf,
        df_params=df_params,
        df_component_params=df_component_params,
        output_dir=figures_dir,
        component='e2e',
        hardware_list=hardware_list,
        style=style
    )


def generate_scaling_latex_table(
    df_perf: pd.DataFrame,
    df_params: pd.DataFrame,
    df_component_params: pd.DataFrame,
    tables_dir: Path,
    hardware_list: list[str] = ["Jetson_AGX_Thor", "RTX_4090", "B100"]
) -> str:
    """
    Generate LaTeX table showing component details and performance for each model variant.
    
    Format: 
    - Rows: Model combinations (pi-0, pi-0-L, pi-0-XL, pi-0-XXL)
    - Columns: Vision, VLM, Action (name + size in one cell), Hardware (ms + Hz in one cell)
    
    Args:
        df_perf: Performance data
        df_params: Total parameter counts
        df_component_params: Component-specific parameter counts
        tables_dir: Directory to save the table (paper_tables)
        hardware_list: List of hardware in desired order
    
    Returns:
        LaTeX table string
    """
    # Get model order (sorted by size)
    model_order = df_params.sort_values('total_params_M')['model'].tolist()
    
    # Merge component params for easier access
    df_components = df_component_params.set_index('model')
    
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("    \\caption{Inference performance of scaled-up VLA models across different hardware platforms.}")
    latex_lines.append("    \\label{tab:pi0_model_scaling}")
    latex_lines.append("    % \\vspace{0.5em}")
    latex_lines.append("    \\scalebox{0.76}{")
    
    # Column format: Model | Vision | VLM | Action | Jetson Thor | RTX 4090 | B100
    col_format = "@{} L{6.5em} R{8em} R{8em} R{7em} R{5em} R{4.5em} R{4.5em} @{}"
    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\toprule")
    
    # Single header row
    header = ["\\textbf{Model}"]
    header.append("\\multicolumn{1}{c}{\\textbf{Vision Encoder}}")
    header.append("\\multicolumn{1}{c}{\\textbf{VLM}}")
    header.append("\\multicolumn{1}{c}{\\textbf{Action Expert}}")
    for hw in hardware_list:
        hw_display = rename_hardware(hw)
        header.append(f"\\multicolumn{{1}}{{c}}{{\\textbf{{{hw_display}}}}}")
    
    latex_lines.append("\n& ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows - one per model
    for i, model in enumerate(model_order):
        row_cells = []
        
        # Model name with total size
        total_params_m = df_params[df_params['model'] == model]['total_params_M'].iloc[0]
        total_params_b = total_params_m / 1000.0
        
        if model == "pi-0":
            model_display = f"$\\pi_0$ ({total_params_b:.1f}B)"
        else:
            variant = model.replace('pi-0-', '')
            model_display = f"$\\pi_0$-{variant} ({total_params_b:.1f}B)"
        row_cells.append(model_display)
        
        # Component information from df_component_params
        comp_row = df_components.loc[model]
        
        # Vision encoder - clean up name and combine with size (1 decimal place)
        vision_full = comp_row['vision_model']
        if 'so400m' in vision_full:
            vision_name = "SigLIP-So"
        elif 'giant' in vision_full:
            vision_name = "SigLIP-Giant"
        else:
            vision_name = vision_full.split('/')[-1].replace('siglip2-', '').replace('-vision', '')
        vision_params_b = comp_row['vision_params_M'] / 1000.0
        row_cells.append(f"{vision_name} ({vision_params_b:.1f}B)")
        
        # VLM - clean up name and combine with size (1 decimal place)
        vlm_full = comp_row['vlm_model']
        if 'gemma' in vlm_full.lower():
            vlm_name = "Gemma-2B"
        elif 'Llama-2-7B' in vlm_full or 'llama2_7b-no-vocab' in vlm_full:
            vlm_name = "Llama2-7B"
        elif 'Llama-2-13B' in vlm_full or 'llama2_13b-no-vocab' in vlm_full:
            vlm_name = "Llama2-13B"
        elif 'Llama-2-70B' in vlm_full or 'llama2_70b-no-vocab' in vlm_full:
            vlm_name = "Llama2-70B"
        else:
            vlm_name = vlm_full.split('/')[-1]
        vlm_params_b = comp_row['vlm_params_M'] / 1000.0
        row_cells.append(f"{vlm_name} ({vlm_params_b:.1f}B)")
        
        # Action expert - use model variant and combine with size (1 decimal place)
        if model == "pi-0":
            action_name = "Act-M"
        else:
            variant = model.replace('pi-0-', '')
            action_name = f"Act-{variant}"
        action_params_b = comp_row['action_params_M'] / 1000.0
        row_cells.append(f"{action_name} ({action_params_b:.1f}B)")
        
        # Hardware performance - show only Hz
        for hw in hardware_list:
            hw_model_data = df_perf[
                (df_perf['hardware.name'] == hw) & 
                (df_perf['model.name'] == model)
            ]
            
            if hw_model_data.empty:
                row_cells.append("N/A")
            else:
                row = hw_model_data.iloc[0]
                e2e_ms = row['e2e_time_ms']
                freq_hz = 1000 / e2e_ms
                row_cells.append(f"\\cellcolor{{lightgray}}{freq_hz:.1f} Hz")
        
        # Use hdashline between rows, bottomrule at the end
        line_ending = " \\\\\n\\hdashline" if i < len(model_order) - 1 else " \\\\"
        latex_lines.append("\n& ".join(row_cells) + line_ending)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table*}")
    
    # Save to file
    output_file = tables_dir / "2_scale_model_size.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table saved to: {output_file}")
    return '\n'.join(latex_lines)


def main(style: str = 'colorblind'):
    """
    Main function to generate plots and table.
    
    Args:
        style: Plot style ('colorblind', 'deep', 'pastel', 'ggplot')
               Default: 'colorblind'
    """
    print("="*80)
    print("Pi0 Model Size Scaling Visualization")
    print("="*80)
    print(f"Using plot style: {style}")
    
    # Load data
    print("\nLoading data...")
    df_perf, df_params, df_component_params = load_scaling_results()
    
    print(f"\nFound {len(df_perf['model.name'].unique())} model variants:")
    for model in df_params.sort_values('total_params_M')['model']:
        params_m = df_params[df_params['model'] == model]['total_params_M'].iloc[0]
        params_b = params_m / 1000.0
        print(f"  - {model}: {params_b:.2f}B ({params_m:.1f}M) parameters")
    
    # Print unique component sizes
    print(f"\nUnique component sizes:")
    vision_sizes = df_component_params['vision_params_M'].unique()
    print(f"  - Vision encoders: {len(vision_sizes)} unique ({', '.join([f'{v/1000:.2f}B' for v in sorted(vision_sizes)])})")
    vlm_sizes = df_component_params['vlm_params_M'].unique()
    print(f"  - VLMs: {len(vlm_sizes)} unique ({', '.join([f'{v/1000:.2f}B' for v in sorted(vlm_sizes)])})")
    action_sizes = df_component_params['action_params_M'].unique()
    print(f"  - Action experts: {len(action_sizes)} unique ({', '.join([f'{v/1000:.2f}B' for v in sorted(action_sizes)])})")
    
    print(f"\nFound {len(df_perf['hardware.name'].unique())} hardware configurations")
    
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
    
    # Generate all component plots
    print("\n" + "="*80)
    print("Generating Component Plots (Individual + Combined: Vision, VLM, Action; Separate: E2E)")
    print("="*80)
    plot_all_components(
        df_perf=df_perf,
        df_params=df_params,
        df_component_params=df_component_params,
        figures_dir=figures_dir,
        hardware_list=hardware_list,
        style=style
    )
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("Generating LaTeX Table")
    print("="*80)
    table = generate_scaling_latex_table(df_perf, df_params, df_component_params, tables_dir)
    print("\n" + table)
    
    print("\n" + "="*80)
    print("Done! Outputs saved to:")
    print(f"  - Figures: {figures_dir}")
    print(f"    * 3 individual component plots (vision, VLM, action) in PNG and PDF")
    print(f"    * 1 combined components plot (vision, VLM, action) in PNG and PDF")
    print(f"    * 1 E2E plot in PNG and PDF")
    print(f"  - Tables: {tables_dir}")
    print(f"    * 1 LaTeX table (2_scale_model_size.tex)")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Pi0 model scaling plots and tables')
    parser.add_argument(
        '--style',
        type=str,
        choices=['colorblind', 'deep', 'pastel', 'ggplot'],
        default='colorblind',
        help='Plot style (default: colorblind)'
    )
    
    args = parser.parse_args()
    main(style=args.style)
