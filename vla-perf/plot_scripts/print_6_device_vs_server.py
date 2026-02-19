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
Generate plots for Device vs Server comparison.

This script loads results from pi0_device_vs_server.csv 
and generates a figure comparing three hardware systems:
1. Jetson Thor (on-device)
2. RTX 4090 (edge-server)
3. B100 (edge-server and cloud)

The plot shows latency across different network configurations.

Figure is saved to ../paper_figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import system_configs and network configs
script_dir = Path(__file__).parent.resolve()
genz_systems_path = script_dir.parent.parent / "genz" / "Systems"
genz_systems_path = genz_systems_path.resolve()

if str(genz_systems_path) not in sys.path:
    sys.path.insert(0, str(genz_systems_path))

# Add vla-perf directory to path for network_latency
vla_perf_path = script_dir.parent
if str(vla_perf_path) not in sys.path:
    sys.path.insert(0, str(vla_perf_path))

from system_configs import system_configs
from plot_util import rename_hardware
from network_latency import (
    ETHERNET_1G_CONFIG,
    ETHERNET_10G_CONFIG,
    WIFI_6_CONFIG,
    WIFI_7_CONFIG,
    CELL_4G_LTE_CONFIG,
    CELL_5G_SA_CONFIG,
    CLOUD_FAST_CONFIG,
    CLOUD_SLOW_CONFIG,
)

# Plot styling constants
LABEL_FONT = 16
TICK_FONT = 13
TEXT_FONT = 14
LEGEND_FONT = 16
TITLE_FONT = 16

# Title display option
SHOW_TITLES = False  # Set to True to enable titles on plots

# Color scheme for the three hardware systems
# Reorder so Jetson is second (red color)
COLORS = {
    'B100': '#2ca02c',                # green
    'Jetson_AGX_Thor': '#d62728',     # red
    'RTX_4090': '#1f77b4',            # blue
}

# Marker styles
MARKERS = {
    'Jetson_AGX_Thor': '*',    # star
    'RTX_4090': 's',           # square
    'B100': 'o'                # circle
}

# Marker sizes
MARKER_SIZES = {
    'Jetson_AGX_Thor': 14,    # larger for star
    'RTX_4090': 9,
    'B100': 9
}


def load_results(csv_path: str = "../perf_results/pi0_device_vs_server.csv") -> pd.DataFrame:
    """Load the device vs server results from CSV."""
    script_dir = Path(__file__).parent
    full_path = script_dir / csv_path
    df = pd.read_csv(full_path)
    return df


def get_network_order():
    """
    Define the order of network configurations for x-axis.
    Returns list of (network_name, display_label) tuples.
    """
    return [
        ("N/A (Local)", "On-device"),
        ("Ethernet 10GbE", "Ethernet 10G"),
        ("Ethernet 1GbE", "Ethernet 1G"),
        ("WiFi 7 (802.11be)", "WiFi 7"),
        ("WiFi 6 (802.11ax)", "WiFi 6"),
        ("5G NR SA (sub-6GHz)", "5G"),
        ("4G LTE", "4G"),
        ("Wired + Fast Cloud", "Ethernet 10G +\nFast Cloud"),
        ("4G + Slow Cloud", "4G +\nSlow Cloud"),
    ]


def plot_latency_curves(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    show_title: bool = SHOW_TITLES,
):
    """
    Plot latency curves for three hardware systems across network configurations.
    
    Args:
        df: DataFrame with results
        model_name: Model name to plot (e.g., "pi0", "pi0.6")
        output_dir: Directory to save the figure
        show_title: If True, display title on plot
    """
    # Filter data for this model
    df_model = df[df['model'] == model_name].copy()
    
    if df_model.empty:
        print(f"No data found for model {model_name}")
        return
    
    # Get network order
    network_order = get_network_order()
    network_names = [net for net, _ in network_order]
    network_labels = [label for _, label in network_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 3.5))
    
    # Define the three hardware systems (order: B100, Jetson, RTX_4090)
    systems = ['B100', 'Jetson_AGX_Thor', 'RTX_4090']
    
    # Plot each system
    for system in systems:
        system_data = df_model[df_model['system'] == system]
        
        if system_data.empty:
            continue
        
        # Extract data for each network configuration
        x_positions = []
        y_values = []
        
        for i, network_name in enumerate(network_names):
            matching = system_data[system_data['network'] == network_name]
            
            if not matching.empty:
                # For on-device (Local), only plot Jetson
                if network_name == "N/A (Local)" and system != "Jetson_AGX_Thor":
                    continue
                
                value = matching.iloc[0]['e2e_total_ms']
                
                x_positions.append(i)
                y_values.append(value)
        
        # Plot line with markers
        if x_positions:
            system_label = rename_hardware(system)
            ax.plot(x_positions, y_values,
                   marker=MARKERS[system],
                   linewidth=2.5,
                   markersize=MARKER_SIZES[system],
                   label=system_label,
                   color=COLORS[system],
                   zorder=3)  # Ensure curves are on top
            
            # Add latency annotations on top of each point
            for x, y in zip(x_positions, y_values):
                ax.annotate(f'{y:.1f}',
                           xy=(x, y),
                           xytext=(0, 8),  # 8 points above the point
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=TICK_FONT - 1,
                           color=COLORS[system],
                        #    weight='bold'
                           )
    
    # Styling
    ax.set_xticks(range(len(network_labels)))
    ax.set_xticklabels(network_labels, fontsize=TICK_FONT, rotation=0, ha='center')
    
    ax.set_ylabel('End-to-End\nLatency (ms)', fontsize=LABEL_FONT)
    
    # Use log scale for latency to see differences better
    ax.set_yscale('log')
    
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Legend: frameless, 3 columns, on top of figure
    ax.legend(fontsize=LEGEND_FONT, loc='upper center', bbox_to_anchor=(0.5, 1.35),
              ncol=3, frameon=False)
    
    # Add vertical lines to separate categories
    # Line after on-device (position 0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    # Line after edge-server (position 6.5)
    ax.axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Add category labels at the top (using axis coordinates for y position)
    # This ensures consistent positioning regardless of data scale
    
    # # On-device label (centered at position 0)
    # ax.text(0, 1.08, 'On-device', 
    #         ha='center', va='bottom', fontsize=LABEL_FONT, weight='bold',
    #         transform=ax.get_xaxis_transform())
    
    # # Edge-server label (centered between positions 1 and 6)
    # ax.text(3.5, 1.08, 'Edge-server', 
    #         ha='center', va='bottom', fontsize=LABEL_FONT, weight='bold',
    #         transform=ax.get_xaxis_transform())
    
    # # Cloud label (centered between positions 7 and 8)
    # ax.text(7.5, 1.08, 'Cloud', 
    #         ha='center', va='bottom', fontsize=LABEL_FONT, weight='bold',
    #         transform=ax.get_xaxis_transform())
    
    if show_title:
        model_display = model_name.replace('pi0', '$\\pi_0$')
        ax.set_title(f'{model_display}: Hardware Comparison Across Networks', fontsize=TITLE_FONT)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{model_name}_device_vs_server_e2e_total_ms.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Also save as PDF
    output_file_pdf = output_dir / f"{model_name}_device_vs_server_e2e_total_ms.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved: {output_file_pdf}")
    
    plt.close()


def generate_network_config_table(output_dir: Path):
    """
    Generate a LaTeX table summarizing network configurations.
    
    Args:
        output_dir: Directory to save the LaTeX table
    """
    # Define network configurations to include
    network_configs = [
        ("Ethernet 1G", ETHERNET_1G_CONFIG),
        ("Ethernet 10G", ETHERNET_10G_CONFIG),
        ("WiFi 6", WIFI_6_CONFIG),
        ("WiFi 7", WIFI_7_CONFIG),
        ("4G", CELL_4G_LTE_CONFIG),
        ("5G", CELL_5G_SA_CONFIG),
        ("Slow Cloud", CLOUD_SLOW_CONFIG),
        ("Fast Cloud", CLOUD_FAST_CONFIG),
    ]
    
    # Helper function to format bandwidth with appropriate units
    def format_bandwidth(bw_mbps):
        """Format bandwidth with Gbps or Mbps units."""
        if bw_mbps >= 1000:
            return f"{bw_mbps/1000:.0f} Gbps"
        else:
            return f"{bw_mbps:.0f} Mbps"
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("    \\caption{Network configuration specifications.}")
    latex_lines.append("    \\label{tab:network_configs}")
    latex_lines.append("    \\vspace{0.5em}")
    latex_lines.append("    \\scalebox{0.72}{")
    
    # Specific column widths as requested
    col_spec = "@{} L{6em} M{6em} M{6em} M{4em} M{4em} M{4em} M{4em} M{5em} M{5em} @{}"
    
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Header row with multicolumn for centering
    header = "\\textbf{Metric}"
    for name, _ in network_configs:
        header += f" & \\multicolumn{{1}}{{c}}{{\\textbf{{{name}}}}}"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Upload bandwidth row
    upload_row = "Upload BW"
    for _, config in network_configs:
        upload_bw = config.bandwidth_mbps("upload")
        upload_row += f" & {format_bandwidth(upload_bw)}"
    upload_row += " \\\\"
    latex_lines.append(upload_row)
    latex_lines.append("\\hdashline")
    
    # Download bandwidth row
    download_row = "Download BW"
    for _, config in network_configs:
        download_bw = config.bandwidth_mbps("download")
        download_row += f" & {format_bandwidth(download_bw)}"
    download_row += " \\\\"
    latex_lines.append(download_row)
    latex_lines.append("\\hdashline")
    
    # Base latency row
    latency_row = "Base Latency"
    for _, config in network_configs:
        latency_row += f" & {config.base_latency_ms:.2f} ms"
    latency_row += " \\\\"
    latex_lines.append(latency_row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    output_file = output_dir / "6_network_configs.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Saved LaTeX table: {output_file}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("LaTeX Table: Network Configuration Specifications")
    print("=" * 80)
    for line in latex_lines:
        print(line)
    print("=" * 80)



def plot_device_server_collab_comparison(
    df_device_server: pd.DataFrame,
    df_edge_cloud: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    show_title: bool = SHOW_TITLES,
):
    """
    Plot comparison of three deployment scenarios:
    1. On-device (Jetson Thor)
    2. Server-side (B100)
    3. Device-server collaboration (Jetson Thor + B100)
    
    Args:
        df_device_server: DataFrame from device_vs_server experiment
        df_edge_cloud: DataFrame from edge_cloud_collaboration experiment
        model_name: Model name to plot
        output_dir: Directory to save the figure
        show_title: If True, display title on plot
    """
    # Filter data for this model
    df_ds = df_device_server[df_device_server['model'] == model_name].copy()
    df_ec = df_edge_cloud[df_edge_cloud['model'] == model_name].copy()
    
    if df_ds.empty and df_ec.empty:
        print(f"No data found for model {model_name}")
        return
    
    # Get network order
    network_order = get_network_order()
    network_names = [net for net, _ in network_order]
    network_labels = [label for _, label in network_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 3.5))
    
    # Define three scenarios with colors and markers
    scenarios = [
        {
            'name': 'On-device (Jetson Thor)',
            'short_name': 'On-device',
            'color': COLORS['Jetson_AGX_Thor'],
            'marker': MARKERS['Jetson_AGX_Thor'],
            'size': MARKER_SIZES['Jetson_AGX_Thor'],
        },
        {
            'name': 'Server-side (B100)',
            'short_name': 'Server-side',
            'color': COLORS['B100'],
            'marker': MARKERS['B100'],
            'size': MARKER_SIZES['B100'],
        },
        {
            'name': 'Device-server collab.',
            'short_name': 'Device-server',
            'color': '#ff7f0e',  # Orange
            'marker': 'D',  # Diamond
            'size': 9,
        },
    ]
    
    # Plot each scenario
    for scenario in scenarios:
        x_positions = []
        y_values = []
        
        for i, network_name in enumerate(network_names):
            if scenario['short_name'] == 'On-device':
                # On-device: only Jetson Thor, only for "N/A (Local)"
                if network_name == "N/A (Local)":
                    matching = df_ds[(df_ds['system'] == 'Jetson_AGX_Thor') & 
                                    (df_ds['network'] == network_name)]
                    if not matching.empty:
                        x_positions.append(i)
                        y_values.append(matching.iloc[0]['e2e_total_ms'])
            
            elif scenario['short_name'] == 'Server-side':
                # Server-side: Get from device_server_collab CSV "Server Only" category
                if network_name != "N/A (Local)":
                    matching = df_ec[(df_ec['category'] == 'Server Only') & 
                                    (df_ec['network'] == network_name)]
                    if not matching.empty:
                        x_positions.append(i)
                        y_values.append(matching.iloc[0]['e2e_total_ms'])
            
            elif scenario['short_name'] == 'Device-server':
                # Device-server: device-server collaboration for all networks (except on-device)
                if network_name != "N/A (Local)":
                    matching = df_ec[(df_ec['category'] == 'Device-Server Collaboration') & 
                                    (df_ec['network'] == network_name)]
                    if not matching.empty:
                        x_positions.append(i)
                        y_values.append(matching.iloc[0]['e2e_total_ms'])
        
        # Plot line with markers
        if x_positions:
            ax.plot(x_positions, y_values,
                   marker=scenario['marker'],
                   linewidth=2.5,
                   markersize=scenario['size'],
                   label=scenario['name'],
                   color=scenario['color'],
                   zorder=3)
            
            # Add latency annotations
            for x, y in zip(x_positions, y_values):
                ax.annotate(f'{y:.1f}',
                           xy=(x, y),
                           xytext=(0, 8),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=TICK_FONT - 1,
                           color=scenario['color'])
    
    # Styling
    ax.set_xticks(range(len(network_labels)))
    ax.set_xticklabels(network_labels, fontsize=TICK_FONT, rotation=0, ha='center')
    
    ax.set_ylabel('End-to-End\nLatency (ms)', fontsize=LABEL_FONT)
    
    # Use log scale for latency
    ax.set_yscale('log')
    
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Legend: frameless, 3 columns, on top of figure
    ax.legend(fontsize=LEGEND_FONT, loc='upper center', bbox_to_anchor=(0.5, 1.35),
              ncol=3, frameon=False)
    
    # Add vertical lines to separate categories
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    ax.axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    
    if show_title:
        model_display = model_name.replace('pi0', '$\\pi_0$')
        ax.set_title(f'{model_display}: Deployment Scenario Comparison', fontsize=TITLE_FONT)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{model_name}_device_server_collab.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Also save as PDF
    output_file_pdf = output_dir / f"{model_name}_device_server_collab.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved: {output_file_pdf}")
    
    plt.close()


def main():
    """Main function to generate all plots."""
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "paper_figures"
    output_dir.mkdir(exist_ok=True)
    output_dir_tables = script_dir.parent / "paper_tables"
    output_dir_tables.mkdir(exist_ok=True)
    
    # Load device vs server results
    print("Loading device vs server results...")
    df_device_server = load_results()
    
    if df_device_server.empty:
        print("Error: No device vs server results found!")
        return
    
    print(f"Loaded {len(df_device_server)} rows from device_vs_server")
    print(f"Models: {df_device_server['model'].unique()}")
    print(f"Systems: {df_device_server['system'].unique()}")
    print(f"Networks: {df_device_server['network'].unique()}")
    
    # Load device-server collaboration results
    print("\nLoading device-server collaboration results...")
    device_server_collab_path = script_dir / "../perf_results/pi0_device_server_collaboration.csv"
    try:
        df_device_server_collab = pd.read_csv(device_server_collab_path)
        print(f"Loaded {len(df_device_server_collab)} rows from device_server_collaboration")
        print(f"Models: {df_device_server_collab['model'].unique()}")
        print(f"Networks: {df_device_server_collab['network'].unique()}")
    except FileNotFoundError:
        print(f"Warning: Device-server collaboration results not found at {device_server_collab_path}")
        df_device_server_collab = pd.DataFrame()
    
    # Generate plots for each model
    models = df_device_server['model'].unique()
    
    for model in models:
        print(f"\nGenerating plots for {model}...")
        
        # Main plot: latency curves by hardware
        plot_latency_curves(df_device_server, model, output_dir)
        
        # Deployment comparison plot (if device-server collaboration data available)
        if not df_device_server_collab.empty:
            plot_device_server_collab_comparison(df_device_server, df_device_server_collab, model, output_dir)
        
    # Generate LaTeX table for network configurations (once, not per model)
    print("\nGenerating network configuration table...")
    generate_network_config_table(output_dir_tables)
    
    print("\n=== All plots generated successfully ===")


if __name__ == "__main__":
    main()

