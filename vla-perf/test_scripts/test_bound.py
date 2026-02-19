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


from GenZ import (
    prefill_moddeling,
    get_model_df,
    get_configs,
    System,
    create_inference_moe_prefill_layer,
    get_AR_time,
)
import os
import pandas as pd
import numpy as np


TPU = System(
    flops=300,
    offchip_mem_bw=1200,
    compute_efficiency=0.8,
    memory_efficiency=0.8,
    bits="bf16",
)
Model = "pi0-vlm"
prefill_output = prefill_moddeling(
    model=Model,
    batch_size=1,
    input_tokens=800,
    system_name=TPU,
    bits="bf16",
    tensor_parallel=1,
    pipeline_parallel=1,
    debug=False,
)

print(prefill_output.keys())
for key in prefill_output.keys():
    print("-" * 20 + key + "-" * 20)
    print(prefill_output[key])


# Note: Op Intensity is not simply Num ops (MFLOP) / Total Data (MB)
# because Total Data (MB) is not necessary all fetched from memory.
df = prefill_output["model_df"]
print(df.keys())
print(
    df[
        [
            "Layer Name",
            "Op Intensity",
            "Num ops (MFLOP)",
            # "Input_a (MB)",
            # "Input_w (MB)",
            # "Output (MB)",
            "Total Data (MB)",
			# "Compute time (msec)",
            # "Memory time (msec)",
            # "Communication time (msec)",
            "Bound",
        ]
    ]
)

# Compute total times
total_compute_time = df["Compute time (msec)"].sum()
total_memory_time = df["Memory time (msec)"].sum()
total_communication_time = df["Communication time (msec)"].sum()

print(f"Total compute time (msec): {total_compute_time}")
print(f"Total memory time (msec): {total_memory_time}")
print(f"Total communication time (msec): {total_communication_time}")

# Compute overall bound: Compute : 0, Memory : 1, Communication : 2
max_time = max(total_compute_time, total_memory_time, total_communication_time)
if max_time == total_compute_time:
    overall_bound = "Compute"
elif max_time == total_memory_time:
    overall_bound = "Memory"
else:
    overall_bound = "Communication"

print(f"Overall Bound: {overall_bound}")

# Example output:
# [1 rows x 11 columns]
# Index(['Layer Name', 'Op Type', 'Dimension', 'Bound', 'C/M ratio',
#        'Op Intensity', 'Latency (msec)', 'Cycles', 'C Effcy',
#        'Num ops (MFLOP)', 'Input_a (MB)', 'Input_w (MB)', 'Output (MB)',
#        'Total Data (MB)', 'Throughput (Tflops)', 'Compute time (msec)',
#        'Memory time (msec)', 'Communication time (msec)', 'Compute cycle',
#        'Memory cycle', 'Communication cycle', 'Compute Utilization',
#        'Memory Utilization', 'Communication Utilization'],
#       dtype='object')
#    Layer Name Op Intensity Compute time (msec) Memory time (msec) Communication time (msec)       Bound
# 0  embeddings   638.519989            1.053966           0.384319                       0.0     Compute
# 1      Repeat            0                 0.0                0.0                       0.0  Collective
# 2         QKV   504.986301            0.048318           0.022278                       0.0     Compute
# 3       Logit    62.060606            0.085899           0.030599                       0.0     Compute
# 4      Attend    62.060606            0.085899           0.030599                       0.0     Compute
# 5    Out Proj   351.085714            0.016106           0.010681                       0.0     Compute
# 6     up+gate    534.26087            0.064425           0.028076                       0.0     Compute
# 7        down    534.26087            0.064425           0.028076                       0.0     Compute
# 8  End Repeat            0                 0.0                0.0                       0.0  Collective
# 9  classifier   638.519989            1.053966           0.384319                       0.0     Compute
