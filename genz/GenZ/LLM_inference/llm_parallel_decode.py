# SPDX-FileCopyrightText: Copyright (c) 2024 Multifidelity Roofline Analysis
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT. Portions are Apache-2.0 while others are MIT.
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
#
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import create_full_parallel_decode_model

unit = Unit()

def parallel_decode_modeling(
    model='BERT',
    batch_size=1,
    input_tokens=4096,
    output_tokens_parallel=8,
    self_attention=True,
    system_name='A100_40GB',
    system_eff=1,
    bits='bf16',
    debug=False,
    model_profilling=False,
    tensor_parallel=1,
    pipeline_parallel=1,
    expert_parallel=1,
    collective_strategy='GenZ',
    network_config=None,
    parallelism_heirarchy="TP{1}_EP{1}_PP{1}",
    model_offload=False,
    ceff=None,
    meff=None,
):
    """
    Model parallel decode performance where multiple tokens are generated simultaneously.
    
    This is useful for diffusion models for action prediction, where multiple action tokens
    are decoded in parallel conditioned on a KV cache (e.g., from a VLM).
    
    Args:
        model: Model name or ModelConfig object
        batch_size: Batch size
        input_tokens: Number of input tokens (context length / KV cache size)
        output_tokens_parallel: Number of tokens to decode in parallel (e.g., action tokens)
        self_attention: If True, parallel tokens also attend to each other (useful for diffusion models)
        system_name: System name (e.g., "A100_80GB")
        system_eff: System efficiency factor
        bits: Bit precision (e.g., "bf16", "int8")
        debug: Whether to print debug information
        model_profilling: Whether to return model profiling data
        tensor_parallel: Degree of tensor parallelism
        pipeline_parallel: Degree of pipeline parallelism
        expert_parallel: Degree of expert parallelism
        collective_strategy: Collective communication strategy
        network_config: Network configuration
        parallelism_heirarchy: Parallelism hierarchy string
        model_offload: Whether to enable model offloading
        ceff: Compute efficiency (overrides system_eff if provided)
        meff: Memory efficiency (overrides system_eff if provided)
        
    Returns:
        ModdelingOutput: Object containing latency, throughput, and other performance metrics
    """
    if pipeline_parallel > 1:
        ub = max(batch_size // pipeline_parallel, 1)
        num_micro_batches = batch_size // ub
        if batch_size < pipeline_parallel:
            warnings.warn(
                f"Batch size is divided into micro batches for pipeline parallel, "
                f"micro batch size:{ub}, consider increasing batch size"
            )
    else:
        ub = batch_size

    ##################################################################################################
    ### System Declaration
    ##################################################################################################

    system = get_inference_system(
        system_name=system_name,
        bits=bits,
        ceff=ceff if ceff is not None else system_eff,
        meff=meff if meff is not None else system_eff,
        network_config=network_config,
        collective_strategy=collective_strategy,
        parallelism_heirarchy=parallelism_heirarchy,
    )

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    model_parallel_decode = create_full_parallel_decode_model(
        name=model,
        input_sequence_length=input_tokens,
        output_gen_tokens_parallel=output_tokens_parallel,
        self_attention=self_attention,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        expert_parallel=expert_parallel,
    )

    model_df = get_model_df(
        model_parallel_decode,
        system=system,
        batch_size=ub,
        intermediate_on_chip=True,
        beam_merge=False,
        beam_size=1,
        model_characterstics=True,
    )
    summary_table = get_summary_table(model_df, unit, model_characterstics=True)

    model_weights = summary_table[f'Total Weights ({unit.unit_mem})'].values[0]  ## In MB
    kv_cache = summary_table[f'KV Cache ({unit.unit_mem})'].values[0]  ## In MB
    unused_weights = summary_table[f'Unused Weights ({unit.unit_mem})'].values[0]  ## In MB

    total_memory_req = model_weights + kv_cache
    # print(f"{model}, {input_tokens} input tokens; Total memory required: {total_memory_req} MB")
    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()  ## MB
    if per_chip_memory < total_memory_req / pipeline_parallel:
        if model_offload:
            system = get_offload_system(
                system=system,
                total_memory_req=total_memory_req / pipeline_parallel,
                debug=debug,
            )
            warnings.warn(
                f"Some Parameter offloaded, effective Memory BW:"
                f"{unit.raw_to_unit(system.offchip_mem_bw, type='BW')} "
            )
            is_offloaded = True
        elif model_profilling:
            warnings.warn(
                f"All params would not fit on chip. System Memory Cap:"
                f"{per_chip_memory/1024} GB , Weights : {model_weights/1024} GB, "
                f"KV Cache:{kv_cache/1024} "
            )
        else:
            raise ValueError(
                f"All params would not fit on chip. System Memory Cap:"
                f"{per_chip_memory/1024} GB , Weights : {model_weights/1024} GB, "
                f"KV Cache:{kv_cache/1024}. \n System:{system_name}"
            )

    ## for tensor sharing per layer.
    assert pipeline_parallel >= 1, "Pipeline parallel must be >= 1"
    assert tensor_parallel >= 1, f"Tensor parallel must be >= 1, {tensor_parallel}"

    if model_profilling:
        return model_df, summary_table

    ##################################################################################################
    ### Parallel decode generation time
    ##################################################################################################
    model_df = get_model_df(
        model_parallel_decode, system, unit, ub, intermediate_on_chip=True
    )
    summary_table = get_summary_table(model_df, unit)

    if debug:
        display_df(simplify_df(model_df))
        display(summary_table)
    parallel_decode_latency = summary_table[
        f'Latency ({unit.unit_time})'
    ].values[0]  # Latency in msec

    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    thrpt = 1000 * batch_size / parallel_decode_latency

    linear_time = summary_table[f'Linear Latency ({unit.unit_time})'].values[
        0
    ]  ## In milliseconds
    attn_time = summary_table[f'Attn Latency ({unit.unit_time})'].values[
        0
    ]  ## In milliseconds
    total_communication_delay = summary_table[
        f'Comm Latency ({unit.unit_time})'
    ].values[0]  ## In milliseconds
    total_time = linear_time + attn_time + total_communication_delay
    runtime_breakdown = get_runtime_breakdown(model_df)
    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
        Latency=parallel_decode_latency,
        Throughput=thrpt,
        Runtime_breakdown=runtime_breakdown,
        is_offload=is_offloaded,
        model_df=model_df,
        summary_table=summary_table,
    )
