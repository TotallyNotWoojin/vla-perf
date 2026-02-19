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
For VLA (Vision-Language-Action) vision models, get the performance across systems, models, and batch sizes.

The user can tune the configuration of the sweeps defined in the following function:
    get_perf_vla_vision()

Reference files or folders in GenZ:

genz/tests/test_prefill_time.py -> prefill functionalities
genz/Systems/system_configs.py -> supported hardware
genz/GenZ/Models/Model_sets/vla_models.py -> supported VLA vision models
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Add parent directory to sys.path for local module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from GenZ import get_configs
from perf_utils import (
    get_powers_of_two_up_to,
    get_optimal_df,
    collect_prefill_perf,
    RESULT_COLUMNS,
    setup_logging,
)


def get_perf_list(
    model_list=["siglip2-base-patch16-224-vision"],
    system_list=["A100_80GB", "H100", "B100"],
    num_device_list=None,
    bits="bf16",
    logger=None,
):
    """
    Collect performance data for VLA vision encoder models.
    
    Output CSV headers:
    model.name,model.stage,model.dec_steps,model.seq_len_inference_prefill,hardware.name,hardware.num_chips,batch_size,time_ms
    
    Note: VLA vision models are encoder-only, so we only test prefill stage.
    The input_tokens for each model is set to its max_model_len from the model config.
    
    Args:
        model_list: List of model names (short names without "vla/" prefix)
        system_list: List of system names
        num_device_list: List of number of devices (powers of two)
        bits: Bit precision (e.g., "bf16")
        logger: Logger instance for logging messages
        
    Returns:
        DataFrame with performance results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if num_device_list is None:
        num_device_list = get_powers_of_two_up_to(16)
    
    result_list = []
    
    for model in model_list:
        try:
            # Get model config to extract max_model_len
            model_config = get_configs(model)
            input_tokens = model_config.max_model_len
            
            logger.info(f"Processing model: {model} (max_model_len={input_tokens})")
            
            for system in system_list:
                logger.info(f"  System: {system}")
                for num_devices in num_device_list:
                    logger.info(f"    Devices: {num_devices}")
                    
                    # Use collect_prefill_perf from perf_utils
                    model_results = collect_prefill_perf(
                        model=model,
                        system=system,
                        num_devices=num_devices,
                        input_tokens=input_tokens,
                        bits=bits,
                    )
                    
                    if model_results:
                        result_list.extend(model_results)
                        logger.info(f"      Collected {len(model_results)} results")
                    else:
                        logger.warning(f"      No results collected for {model} on {system} with {num_devices} devices")
                        
        except Exception as e:
            logger.error(f"Error processing model {model}: {e}")
            continue
    
    # Create DataFrame
    if not result_list:
        logger.warning("No results collected. Returning empty DataFrame.")
        return pd.DataFrame(columns=RESULT_COLUMNS)
    
    df = pd.DataFrame(result_list, columns=RESULT_COLUMNS)
    logger.info(f"Total results collected: {len(df)}")
    
    # Use get_optimal_df from perf_utils to filter optimal results
    df_optimal = get_optimal_df(df, apply_pareto=True)
    logger.info(f"Optimal results after filtering: {len(df_optimal)}")
    
    return df_optimal


"""
User-defined sweep parameters starts:
"""


def get_perf_vla_vision():
    """
    Test performance of VLA vision models (SigLIP2 variants).
    These are encoder-only vision transformer models.
    """
    # Set up logging
    logger = setup_logging("perf_results/vision_encoder_perf.log")
    logger.info("=" * 80)
    logger.info("Starting VLA Vision Encoder Performance Evaluation")
    logger.info("=" * 80)
    
    # Use the short model names (without "vla/" prefix) as they are registered in MODEL_DICT
    model_list = [
        "siglip2-base-patch16-224-vision",
        "siglip2-large-patch16-384-vision",
        "siglip2-so400m-patch14-384-vision",
        "siglip2-giant-opt-patch16-384-vision",
    ]
    system_list = ["A100_80GB", "H100", "B100"]
    num_device_list = get_powers_of_two_up_to(4)
    bits = "bf16"
    
    logger.info(f"Models: {model_list}")
    logger.info(f"Systems: {system_list}")
    logger.info(f"Number of devices: {num_device_list}")
    logger.info(f"Bits: {bits}")
    logger.info("Note: input_tokens will be set to each model's max_model_len from config")
    
    df_results = get_perf_list(
        model_list=model_list,
        system_list=system_list,
        num_device_list=num_device_list,
        bits=bits,
        logger=logger,
    )
    
    # Save results
    output_file_path = "perf_results/vla_vision_perf.csv"
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_results.to_csv(output_file_path, index=False)
    logger.info(f"DataFrame successfully written to {output_file_path}")
    logger.info(f"Total rows: {len(df_results)}")
    
    # Print summary
    if not df_results.empty:
        logger.info("\n" + "=" * 80)
        logger.info("Performance Summary:")
        logger.info("=" * 80)
        logger.info(f"\n{df_results.to_string()}")
    else:
        logger.warning("No results to display")
    
    logger.info("=" * 80)
    logger.info("Performance evaluation completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Get the performance of VLA vision models
    get_perf_vla_vision()
