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
Generate table for Synchronous vs Asynchronous Inference comparison.

This script loads results from pi0_device_vs_server.csv 
and generates a LaTeX table comparing synchronous and asynchronous 
inference throughput across different network and hardware configurations.

Table is saved to ../paper_tables/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import system_configs and network configs
script_dir = Path(__file__).parent.resolve()
genz_systems_path = script_dir.parent.parent / "genz" / "Systems"
genz_systems_path = genz_systems_path.resolve()

if str(genz_systems_path) not in sys.path:
    sys.path.insert(0, str(genz_systems_path))

# Add vla-perf directory to path
vla_perf_path = script_dir.parent
if str(vla_perf_path) not in sys.path:
    sys.path.insert(0, str(vla_perf_path))

from plot_util import rename_hardware


def load_results(csv_path: str = "../perf_results/pi0_device_vs_server.csv") -> pd.DataFrame:
    """Load the device vs server results from CSV."""
    script_dir = Path(__file__).parent
    full_path = script_dir / csv_path
    df = pd.read_csv(full_path)
    return df


def get_network_display_name(network_name: str) -> str:
    """Convert network name to display format."""
    mapping = {
        "N/A (Local)": "On-device",
        "Ethernet 10GbE": "Ethernet 10G",
        "Ethernet 1GbE": "Ethernet 1G",
        "WiFi 7 (802.11be)": "WiFi 7",
        "WiFi 6 (802.11ax)": "WiFi 6",
        "5G NR SA (sub-6GHz)": "5G",
        "4G LTE": "4G",
        # "Wired + Fast Cloud": "Ethernet 10G + Fast Cloud",
        # "4G + Slow Cloud": "4G + Slow Cloud",
    }
    return mapping.get(network_name, network_name)


def generate_async_inference_table(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
):
    """
    Generate LaTeX table comparing synchronous and asynchronous inference.
    
    Args:
        df: DataFrame with results
        model_name: Model name to generate table for
        output_dir: Directory to save the LaTeX table
    """
    # Filter data for this model
    df_model = df[df['model'] == model_name].copy()
    
    if df_model.empty:
        print(f"No data found for model {model_name}")
        return
    
    # Filter to only keep specific hardware and networks
    # On-device: only Jetson Thor
    # Edge-server: only B100
    # Cloud: only B100
    # Networks: Ethernet 10G, Ethernet 1G, WiFi 7, 5G, Wired + Fast Cloud, 4G + Slow Cloud
    
    allowed_networks = [
        "N/A (Local)",
        "Ethernet 10GbE",
        "Ethernet 1GbE",
        "WiFi 7 (802.11be)",
        "5G NR SA (sub-6GHz)",
        "4G LTE",
        "Wired + Fast Cloud",
        "4G + Slow Cloud"
    ]
    
    # Filter by network
    df_model = df_model[df_model['network'].isin(allowed_networks)].copy()
    
    # Filter by hardware: only Jetson Thor for on-device, only B100 for server-side and cloud
    df_model = df_model[
        # ((df_model['category'] == 'On-device') & (df_model['system'] == 'Jetson_AGX_Thor')) |
        ((df_model['category'] == 'Edge-server') & (df_model['system'] == 'B100')) |
        ((df_model['category'] == 'Cloud') & (df_model['system'] == 'B100'))
    ].copy()
    
    if df_model.empty:
        print(f"No data found for model {model_name} after filtering")
        return
    
    # Calculate async speedup
    df_model['async_speedup'] = df_model['freq_async_hz'] / df_model['frequency_hz']
    
    # Sort by category, then by latency
    df_model['category_order'] = df_model['category'].map({
        'On-device': 1,
        'Edge-server': 2,
        'Cloud': 3
    })
    df_model = df_model.sort_values(['category_order', 'e2e_total_ms'])
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    model_display = model_name.replace('pi0', '$\\pi_0$')
    latex_lines.append("    \\caption{Inference frequency of synchronous and asynchronous systems.}")
    latex_lines.append("    \\label{tab:async_inference_robot}")
    latex_lines.append("    \\vspace{0.5em}")
    latex_lines.append("    \\scalebox{0.85}{")
    
    # Column specification: Hardware, Network, Latency, Sync Hz, Async Hz, Async Speedup
    col_spec = "@{} L{5em} L{8em} R{4em} R{6em} R{6em} R{4em} @{}"
    
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Header row (single line, units in cells)
    header = "\\textbf{Hardware} & \\textbf{Network} & \\textbf{Latency} & \\textbf{Freq. (Sync)} & \\textbf{Freq. (Async)} & \\textbf{Speedup} \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Track current category for section breaks
    current_category = None
    
    for idx, row in df_model.iterrows():
        # Add section break if category changes
        if current_category is not None and row['category'] != current_category:
            latex_lines.append("\\midrule")
        current_category = row['category']
        
        # Format hardware name
        hardware = rename_hardware(row['system'])
        
        # Format network name
        network = get_network_display_name(row['network'])
        
        # Format values (include units in cells)
        latency = f"{row['e2e_total_ms']:.1f} ms"
        sync_hz = f"{row['frequency_hz']:.1f} Hz"
        async_hz = f"{row['freq_async_hz']:.1f} Hz"
        speedup = f"{row['async_speedup']:.2f}$\\times$"
        
        # Create row with gray background for speedup column
        table_row = f"{hardware} & {network} & {latency} & {sync_hz} & {async_hz} & \\cellcolor{{lightgray}} {speedup} \\\\"
        latex_lines.append(table_row)
        
        # Add dashed line after each row (except before category breaks)
        # We'll add hdashline, but skip it if next row changes category
        if idx != df_model.index[-1]:  # Not the last row
            # Check if next row has different category
            next_idx_loc = df_model.index.get_loc(idx) + 1
            if next_idx_loc < len(df_model):
                next_row = df_model.iloc[next_idx_loc]
                if next_row['category'] == row['category']:
                    latex_lines.append("\\hdashline")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    output_file = output_dir / f"7_{model_name}_async_inference_robot.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Saved LaTeX table: {output_file}")
    
    # Also print to console
    print("\n" + "=" * 100)
    print(f"LaTeX Table: {model_display} Synchronous vs. Asynchronous Inference")
    print("=" * 100)
    for line in latex_lines:
        print(line)
    print("=" * 100)
    
    # Print summary statistics
    print(f"\nSummary for {model_name}:")
    print(f"Average async speedup: {df_model['async_speedup'].mean():.2f}x")
    print(f"Max async speedup: {df_model['async_speedup'].max():.2f}x")
    print(f"Min async speedup: {df_model['async_speedup'].min():.2f}x")


def calculate_two_system_async_performance(
    df: pd.DataFrame,
    model_name: str,
    cap_hz_values: list = [5, 10, 20],
):
    """
    Calculate performance for two-system asynchronous operation.
    
    System 1: image upload + vision + diffusion + action download
    System 2: vision + VLM (uses latest uploaded image, capped at specified Hz)
    
    Args:
        df: DataFrame with results
        model_name: Model name to analyze
        cap_hz_values: List of cap frequencies for System 2 (e.g., [5, 10, 20])
    """
    print("\n" + "=" * 100)
    print(f"Two-System Asynchronous Performance Analysis for {model_name}")
    print("=" * 100)
    
    # Filter data for this model
    df_model = df[df['model'] == model_name].copy()
    
    if df_model.empty:
        print(f"No data found for model {model_name}")
        return
    
    # Filter configurations: Thor with no network, or B100 with specific networks
    allowed_configs = [
        ("Jetson_AGX_Thor", "N/A (Local)"),
        ("B100", "Ethernet 10GbE"),
        ("B100", "WiFi 7 (802.11be)"),
        ("B100", "5G NR SA (sub-6GHz)"),
    ]
    
    results = []
    
    for system, network in allowed_configs:
        matching = df_model[(df_model['system'] == system) & (df_model['network'] == network)]
        
        if matching.empty:
            continue
        
        row = matching.iloc[0]
        
        # System 2 latency: vision + VLM (ms)
        system_2_latency_ms = row['vlm_ms']
        
        # System 1 latency: image upload + vision + diffusion + action download (ms)
        system_1_latency_ms = row['network_image_ms'] + row['vision_ms'] + row['action_ms'] + row['network_action_ms']
        
        # Original sync Hz
        sync_hz = row['frequency_hz']
        
        # Calculate for each cap Hz value
        for cap_hz in cap_hz_values:
            # Formula: (1 - cap_hz * latency_system2_sec) / latency_system1_sec
            # Convert ms to seconds
            system_2_latency_sec = system_2_latency_ms / 1000.0
            system_1_latency_sec = system_1_latency_ms / 1000.0
            
            # Calculate effective frequency
            numerator = 1.0 - (cap_hz * system_2_latency_sec)
            
            if numerator <= 0:
                # System 2 takes up all the time, can't run System 1
                calculated_hz = 0
            else:
                calculated_hz = numerator / system_1_latency_sec
            
            # If calculated Hz < sync Hz, use sync Hz as result
            effective_hz = max(calculated_hz, sync_hz)
            
            results.append({
                'system': system,
                'network': network,
                'cap_hz': cap_hz,
                'system_1_latency_ms': system_1_latency_ms,
                'system_2_latency_ms': system_2_latency_ms,
                'sync_hz': sync_hz,
                'calculated_hz': calculated_hz,
                'effective_hz': effective_hz,
                'speedup_vs_sync': effective_hz / sync_hz if sync_hz > 0 else 0,
            })
    
    # Print results as a formatted table in bash
    print("\nConfiguration Details:")
    print(f"  System 1: image upload + vision + diffusion + action download")
    print(f"  System 2: vision + VLM (capped at specified Hz)")
    print()
    
    # Print table header
    print("\n" + "=" * 150)
    header = f"{'Hardware':<15} {'Network':<20} {'S2 Cap':<10} {'S1 Lat':<12} {'S2 Lat':<12} {'Sync Hz':<12} {'Async Hz':<12} {'Speedup':<10}"
    print(header)
    print(f"{'':15} {'':20} {'(Hz)':10} {'(ms)':12} {'(ms)':12} {'':12} {'':12} {'':10}")
    print("=" * 150)
    
    for result in results:
        hardware = rename_hardware(result['system'])
        network = get_network_display_name(result['network'])
        
        row = (f"{hardware:<15} {network:<20} {result['cap_hz']:<10.0f} "
               f"{result['system_1_latency_ms']:<12.2f} {result['system_2_latency_ms']:<12.2f} "
               f"{result['sync_hz']:<12.2f} {result['effective_hz']:<12.2f} "
               f"{result['speedup_vs_sync']:<10.2f}x")
        print(row)
    
    print("=" * 150)
    
    return results


def generate_two_system_latex_table(
    results: list,
    model_name: str,
    output_dir: Path,
):
    """
    Generate LaTeX table for two-system async performance.
    
    Args:
        results: List of result dictionaries from calculate_two_system_async_performance
        model_name: Model name
        output_dir: Directory to save the LaTeX table
    """
    if not results:
        print("No results to generate LaTeX table")
        return
    
    # Group results by hardware+network configuration
    configs = {}
    for result in results:
        hardware = rename_hardware(result['system'])
        network = get_network_display_name(result['network'])
        config_key = (hardware, network)
        
        if config_key not in configs:
            configs[config_key] = {
                'hardware': hardware,
                'network': network,
                's1_lat': result['system_1_latency_ms'],
                's2_lat': result['system_2_latency_ms'],
                'sync_hz': result['sync_hz'],
                'caps': {}
            }
        
        configs[config_key]['caps'][result['cap_hz']] = {
            'async_hz': result['effective_hz'],
            'speedup': result['speedup_vs_sync']
        }
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("\\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("\\caption{Performance gains by using dual-system inference.}")
    latex_lines.append(f"\\label{{tab:{model_name}_system_1_2}}")
    latex_lines.append("\\vspace{0.3em}")
    latex_lines.append("\\scalebox{0.76}{")
    
    # Column specification with empty columns for spacing
    col_spec = ("@{}\n"
                "L{5em} L{5em}\n"
                "R{3em} R{3em} R{5em}\n"
                "M{0em}\n"
                "R{5.5em} R{3.5em}\n"
                "M{0em}\n"
                "R{5.5em} R{3.5em}\n"
                "@{}")
    
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Multi-row header (units will be in cells)
    header1 = ("\\multirow{2}{*}{\\textbf{Hardware}}\n"
               "& \\multirow{2}{*}{\\textbf{Network}}\n"
               "& \\multirow{2}{*}{\\textbf{S1 Lat.}}\n"
               "& \\multirow{2}{*}{\\textbf{S2 Lat.}}\n"
               "& \\multirow{2}{*}{\\textbf{Freq. (Sync)}}\n"
               "& \n"
               "& \\multicolumn{2}{c}{\\textbf{S2 Cap = 5 Hz}}\n"
               "& \n"
               "& \\multicolumn{2}{c}{\\textbf{S2 Cap = 10 Hz}} \\\\")
    latex_lines.append(header1)
    
    # Add cmidrules
    latex_lines.append("%")
    latex_lines.append("\\cmidrule(lr){7-8}")
    latex_lines.append("\\cmidrule(lr){10-11}")
    latex_lines.append("%")
    
    # Second header row
    header2 = ("& & & & &\n"
               "& \\textbf{Freq. (Async)} & \\textbf{Speedup}\n"
               "&\n"
               "& \\textbf{Freq. (Async)} & \\textbf{Speedup} \\\\")
    latex_lines.append(header2)
    latex_lines.append("\\midrule")
    
    # Data rows
    first_row = True
    for config_key, config_data in configs.items():
        if not first_row:
            latex_lines.append("\\hdashline")
        first_row = False
        
        hardware = config_data['hardware']
        network = config_data['network']
        # Replace spaces in WiFi with non-breaking space
        network = network.replace("WiFi 7", "WiFi~7")
        
        # Add units to cell values
        s1_lat = f"{config_data['s1_lat']:.1f} ms"
        s2_lat = f"{config_data['s2_lat']:.1f} ms"
        sync_hz = f"{config_data['sync_hz']:.1f} Hz"
        
        # Get data for each cap (5 Hz and 10 Hz)
        cap_5 = config_data['caps'].get(5, {})
        cap_10 = config_data['caps'].get(10, {})
        
        async_5 = f"{cap_5.get('async_hz', 0):.1f} Hz" if cap_5 else "--"
        speedup_5 = f"{cap_5.get('speedup', 0):.2f}$\\times$" if cap_5 else "--"
        
        async_10 = f"{cap_10.get('async_hz', 0):.1f} Hz" if cap_10 else "--"
        speedup_10 = f"{cap_10.get('speedup', 0):.2f}$\\times$" if cap_10 else "--"
        
        # Create row with gray background for speedup cells
        row = (f"{hardware}\n"
               f"& {network}\n"
               f"& {s1_lat}\n"
               f"& {s2_lat}\n"
               f"& {sync_hz}\n"
               f"&& {async_5} & \\cellcolor{{lightgray}} {speedup_5}\n"
               f"&& {async_10} & \\cellcolor{{lightgray}} {speedup_10} \\\\")
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    output_file = output_dir / f"7_{model_name}_system_1_2.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\nSaved LaTeX table: {output_file}")
    
    # Also print to console
    print("\n" + "=" * 100)
    print(f"LaTeX Table: Two-System Asynchronous Inference")
    print("=" * 100)
    for line in latex_lines:
        print(line)
    print("=" * 100)


def main():
    """Main function to generate async inference table."""
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "paper_tables"
    output_dir.mkdir(exist_ok=True)
    
    # Load device vs server results
    print("Loading device vs server results...")
    df = load_results()
    
    if df.empty:
        print("Error: No device vs server results found!")
        return
    
    print(f"Loaded {len(df)} rows from device_vs_server")
    print(f"Models: {df['model'].unique()}")
    print(f"Systems: {df['system'].unique()}")
    print(f"Networks: {df['network'].unique()}")
    
    # Generate table for each model
    models = df['model'].unique()
    
    for model in models:
        print(f"\nGenerating async inference table for {model}...")
        generate_async_inference_table(df, model, output_dir)
        
        # Calculate two-system async performance
        print(f"\nCalculating two-system async performance for {model}...")
        results = calculate_two_system_async_performance(df, model, cap_hz_values=[5, 10, 20])
        
        # Generate LaTeX table for two-system performance
        if results:
            generate_two_system_latex_table(results, model, output_dir)
    
    print("\n=== All tables generated successfully ===")


if __name__ == "__main__":
    main()

