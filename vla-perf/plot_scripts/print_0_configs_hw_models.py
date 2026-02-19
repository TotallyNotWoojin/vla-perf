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
Generate LaTeX tables for hardware and model configuration parameters.

This script generates two LaTeX tables:
1. Hardware parameters table: Compute, memory, bandwidth specs for selected hardware
2. Model parameters table: Architecture details for π0's components (Vision, VLM, Action Expert)
"""

import sys
from pathlib import Path
import re

# Add parent directory to path to import system_configs
script_dir = Path(__file__).parent.resolve()
genz_systems_path = script_dir.parent.parent / "genz" / "Systems"
genz_systems_path = genz_systems_path.resolve()

if str(genz_systems_path) not in sys.path:
    sys.path.insert(0, str(genz_systems_path))

from system_configs import system_configs
from GenZ.Models.default_models import MODEL_DICT

# Import plot utilities
from plot_util import rename_hardware

# Add GenZ path for perf_utils dependencies
genz_path = script_dir.parent.parent / "genz"
if str(genz_path) not in sys.path:
    sys.path.insert(0, str(genz_path))

# Add perf_utils path for parameter calculation functions
perf_utils_path = script_dir.parent
if str(perf_utils_path) not in sys.path:
    sys.path.insert(0, str(perf_utils_path))

from perf_utils import calculate_transformer_params, format_param_count


def load_pi0_configs_from_model_dict():
    """
    Load π0 model configurations using MODEL_DICT.
    This follows the same approach as pi0_perf.py.
    """
    
    vision_config = MODEL_DICT.get_model("pi0-vision")
    vlm_config = MODEL_DICT.get_model("pi0-vlm")
    action_config = MODEL_DICT.get_model("pi0-action-expert")
    
    return vision_config, vlm_config, action_config


# Load model configurations
pi0_vision_config, pi0_vlm_config, pi0_action_expert_config = load_pi0_configs_from_model_dict()


def generate_hardware_table(output_dir: Path) -> str:
    """
    Generate Table: Hardware specifications for selected hardware.
    
    Hardware: Jetson Thor, RTX 4090, A100, H100, B100
    Columns: Hardware | FP32 | FP16 | INT8 | Memory (GB) | Memory BW (GB/s)
    """
    latex_lines = []
    latex_lines.append("\\begin{table*}[h]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("    \\begin{footnotesize}")
    latex_lines.append("    \\caption{Hardware specifications for the GPUs used in our evaluation.}")
    latex_lines.append("    % \\vspace{-1em}")
    latex_lines.append("    \\label{tab:hardware_specs}")
    latex_lines.append("    \\scalebox{1.0}{")
    latex_lines.append("        \\begin{tabular}{L{6em} R{6em} R{6em} R{6em} R{4em} R{5em}}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Hardware}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{FP32}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{FP16}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{INT8}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Memory}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Memory BW}} \\\\")
    latex_lines.append("\\midrule")
    
    # Hardware list in order
    hardware_order = ["Jetson_AGX_Thor", "RTX_4090", "A100_80GB", "H100", "B100"]
    
    for i, hw in enumerate(hardware_order):
        if hw not in system_configs:
            continue
        
        hw_config = system_configs[hw]
        
        # Get compute capabilities (TFLOPS for FP32/FP16, TOPS for INT8)
        flops = hw_config.get('Flops', {})
        if isinstance(flops, dict):
            fp32_flops = flops.get('fp32', 0)
            fp16_flops = flops.get('bf16', flops.get('fp16', 0))
            int8_flops = flops.get('int8', 0)
        else:
            # legacy format - assume it's bf16/fp16
            fp32_flops = 0
            fp16_flops = flops
            int8_flops = 0
        
        # Get memory size (GB)
        memory_gb = hw_config.get('Memory_size', 0)
        
        # Get memory bandwidth (GB/s)
        memory_bw = hw_config.get('Memory_BW', 0)
        
        # Format hardware name using rename utility
        hw_display = rename_hardware(hw)
        
        # Use hdashline between rows, bottomrule at the end
        line_ending = " \\\\\n\\hdashline" if i < len(hardware_order) - 1 else " \\\\"
        
        # Format compute values with units - show "-" if 0
        # FP32 and FP16 use TFLOP/s, INT8 uses TOP/s
        fp32_str = f"{fp32_flops:.0f} TFLOP/s" if fp32_flops > 0 else "-"
        fp16_str = f"{fp16_flops:.0f} TFLOP/s" if fp16_flops > 0 else "-"
        int8_str = f"{int8_flops:.0f} TOP/s" if int8_flops > 0 else "-"
        
        latex_lines.append(
            f"{hw_display}\n"
            f"& {fp32_str}\n"
            f"& {fp16_str}\n"
            f"& {int8_str}\n"
            f"& {memory_gb:.0f} GB\n"
            f"& {memory_bw:.0f} GB/s{line_ending}"
        )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("        \\end{tabular}")
    latex_lines.append("    }")
    latex_lines.append("    \\end{footnotesize}")
    latex_lines.append("\\end{table*}")
    
    table_str = "\n".join(latex_lines)
    
    # Save to file
    output_path = output_dir / "0_hardware_specs.tex"
    with open(output_path, 'w') as f:
        f.write(table_str)
    
    print(f"Hardware table saved to: {output_path}")
    print("\n" + "="*80)
    print(table_str)
    print("="*80 + "\n")
    
    return table_str


def generate_model_table(output_dir: Path) -> str:
    """
    Generate Table: π0 model component parameters.
    
    Components: Vision Encoder, VLM Backbone, Action Expert
    Columns: Component | Layers | Hidden Dim | Interm. Dim | Query Heads | KV Heads | Params
    """
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\setlength\\dashlinedash{0.2pt}")
    latex_lines.append("    \\setlength\\dashlinegap{1.5pt}")
    latex_lines.append("    \\begin{footnotesize}")
    latex_lines.append("    \\caption{Parameter specifications for $\\pi_0$ model components (without vocabulary table).}")
    latex_lines.append("    % \\vspace{-1em}")
    latex_lines.append("    \\label{tab:pi0_model_specs}")
    latex_lines.append("    \\scalebox{1.0}{")
    latex_lines.append("        \\begin{tabular}{L{7em} R{3em} R{5em} R{5em} R{4em} R{4.5em} R{3.5em}}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Component}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Layers}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Hidden Dim}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Interm. Dim}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Q Heads}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{KV Heads}}")
    latex_lines.append("& \\multicolumn{1}{c}{\\textbf{Params}} \\\\")
    latex_lines.append("\\midrule")
    
    # Model components in order
    components = [
        ("Vision Encoder", pi0_vision_config),
        ("VLM Backbone", pi0_vlm_config),
        ("Action Expert", pi0_action_expert_config),
    ]
    
    for i, (name, config) in enumerate(components):
        # Get architecture parameters
        hidden_size = config.hidden_size
        num_layers = config.num_encoder_layers + config.num_decoder_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        ffn_size = config.intermediate_size
        
        # Calculate total parameters
        total_params = calculate_transformer_params(config)
        params_str = format_param_count(total_params)
        
        # Use hdashline between rows, bottomrule at the end
        line_ending = " \\\\\n\\hdashline" if i < len(components) - 1 else " \\\\"
        
        latex_lines.append(
            f"{name}\n"
            f"& {num_layers}\n"
            f"& {hidden_size:,}\n"
            f"& {ffn_size:,}\n"
            f"& {num_heads}\n"
            f"& {num_kv_heads}\n"
            f"& {params_str}{line_ending}"
        )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("        \\end{tabular}")
    latex_lines.append("    }")
    latex_lines.append("    \\end{footnotesize}")
    latex_lines.append("\\end{table}")
    
    table_str = "\n".join(latex_lines)
    
    # Save to file
    output_path = output_dir / "0_model_specs.tex"
    with open(output_path, 'w') as f:
        f.write(table_str)
    
    print(f"Model table saved to: {output_path}")
    print("\n" + "="*80)
    print(table_str)
    print("="*80 + "\n")
    
    return table_str


def main():
    """Generate both hardware and model configuration tables."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "paper_tables"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Generating Hardware and Model Configuration Tables")
    print("=" * 80)
    print()
    
    # Generate hardware table
    print("1. Generating hardware specifications table...")
    generate_hardware_table(output_dir)
    
    # Generate model table
    print("2. Generating model specifications table...")
    generate_model_table(output_dir)
    
    print("=" * 80)
    print("All tables generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

