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
Utility functions for plotting and table generation.
"""

def rename_hardware(hw_name: str) -> str:
    """
    Rename hardware names for display in tables and plots.
    
    Args:
        hw_name: Hardware name as stored (e.g., "A100_80GB", "Jetson_AGX_Thor")
    
    Returns:
        Formatted hardware name for display (e.g., "A100", "Jetson Thor")
    """
    rename_map = {
        "A100_80GB": "A100",
        "RTX_4090": "RTX 4090",
        "Jetson_AGX_Thor": "Jetson Thor",
    }
    
    # If exact match exists, use it
    if hw_name in rename_map:
        return rename_map[hw_name]
    
    # Otherwise, just replace underscores with spaces
    return hw_name.replace("_", " ")
