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
Script to print parameter counts for all models defined in vla_models.py

This script calculates and displays the total parameter count for each model
configuration defined in genz/GenZ/Models/Model_sets/vla_models.py
"""

import sys
from pathlib import Path

from GenZ.Models.Model_sets.vla_models import vla_models
# from GenZ.Models.Model_sets.google import google_models as vla_models
# from GenZ.Models.Model_sets.meta import meta_models as vla_models

# Add parent directory to sys.path for local module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from perf_utils import calculate_transformer_params, format_param_count

def main():
    """Print parameter counts for all VLA models."""
    print("=" * 100)
    print("VLA Models Parameter Counts")
    print("=" * 100)
    print()
    
    # Sort models by name for consistent output
    sorted_models = sorted(vla_models.items(), key=lambda x: x[0])
    
    # Print header
    print(f"{'Model Name':<50} {'Parameters':<20} {'Formatted':<15}")
    print("-" * 100)
    
    # Calculate and print for each model
    for model_name, config in sorted_models:
        param_count = calculate_transformer_params(config)
        formatted = format_param_count(param_count)
        
        print(f"{model_name:<50} {param_count:<20,} {formatted:<15}")
    
    print("-" * 100)
    print()
    
    # Print summary statistics
    total_params = sum(calculate_transformer_params(config) for config in vla_models.values())
    num_models = len(vla_models)
    avg_params = total_params / num_models if num_models > 0 else 0
    
    print(f"Summary:")
    print(f"  Total models: {num_models}")
    print(f"  Total parameters (all models): {format_param_count(total_params)} ({total_params:,})")
    print(f"  Average parameters per model: {format_param_count(avg_params)} ({avg_params:,.0f})")
    print()


if __name__ == "__main__":
    main()
