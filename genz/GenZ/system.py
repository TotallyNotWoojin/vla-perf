# MIT License

# Copyright (c) 2024 Multifidelity Roofline Analysis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import math
from GenZ.unit import Unit
import json
import warnings

class System(object):
    # Memory size multiplier per data type (in bytes)
    mem_multiplier = {
        'fp32': 4, 'f32': 4, 'tf32': 4,
        'bf16': 2, 'fp16': 2,
        'fp8': 1, 'int8': 1,
        'fp6': 0.75,
        'fp4': 0.5, 'int4': 0.5,
        'int2': 0.25
    }

    # Supported precision types (for validation)
    supported_precisions = {'fp32', 'f32', 'tf32', 'bf16', 'fp16', 'fp8', 'int8', 'fp6', 'fp4', 'int4', 'int2'}

    def __init__(self, unit=None,
                flops=123, mxu_shape=None,
                onchip_mem_bw=18000, on_chip_mem_size=float('Inf'),
                offchip_mem_bw=900, off_chip_mem_size=float('Inf'),
                external_mem_bw=0,
                frequency=940, bits='bf16',
                compute_efficiency=1, memory_efficiency=1, comm_efficiency=1,
                interchip_link_bw = 25, num_nodes = 1, interchip_link_latency=1.9,
                compute_engine='GenZ',    # GenZ or Scale-sim or real-HW
                collective_strategy='GenZ',    # GenZ or ASTRA-SIM
                topology='FullyConnected',
                parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
                network_config = None,
                gear_params = None,
                ):

        if unit is None:
            self.unit = Unit()
        else:
            self.unit = unit

        # Handle per-precision Flops (new format) or single value (legacy format)
        if isinstance(flops, dict):
            # New format: dict mapping precision -> TFLOPS
            self.flops_per_precision = {k: self.unit.unit_to_raw(v, type='C') for k, v in flops.items()}
            # Use bf16/fp16 as default, or first available precision
            if bits in self.flops_per_precision:
                self.flops = self.flops_per_precision[bits]
            elif 'bf16' in self.flops_per_precision:
                self.flops = self.flops_per_precision['bf16']
            elif 'fp16' in self.flops_per_precision:
                self.flops = self.flops_per_precision['fp16']
            else:
                # Use first available
                self.flops = list(self.flops_per_precision.values())[0]
        else:
            # Legacy format: single value assumed to be bf16 TFLOPS
            self.flops = self.unit.unit_to_raw(flops, type='C')
            self.flops_per_precision = None  # Will use legacy compute_multiplier fallback

        self.op_per_sec = self.flops / 2

        self.frequency = self.unit.unit_to_raw(frequency, type='F')
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')
        self.interchip_link_bw = self.unit.unit_to_raw(interchip_link_bw, type='BW')
        self.interchip_link_latency = interchip_link_latency * 1e-6     ## us
        self.external_mem_bw = self.unit.unit_to_raw(external_mem_bw, type='BW')
        self.on_chip_mem_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.on_chip_mem_left_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.off_chip_mem_size = self.unit.unit_to_raw(off_chip_mem_size, type='M')
        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency
        self.comm_efficiency = comm_efficiency
        self.mxu_shape = mxu_shape

        self.compute_engine = compute_engine
        assert self.compute_engine.lower() in ['genz', 'scale-sim', 'real-hw'], "Invalid compute_engine. Must be one of: GenZ, Scale-sim, real-HW"

        self.collective_strategy = collective_strategy
        assert self.collective_strategy in ['GenZ', 'ASTRA-SIM'], "Invalid collective_strategy. Must be one of: GenZ, ASTRA-SIM"
        self.num_nodes = num_nodes
        self.topology = topology
        self.bits = bits
        self.parallelism_heirarchy = parallelism_heirarchy   ## TP{1}_EP{1}_PP{1}
        self.network_config = network_config

        # Validate and set precision
        self._validate_and_set_precision(bits)

        if gear_params:
            self.gear_r = gear_params['r']
            self.gear_s = gear_params['s']
            self.gear_b = gear_params['b']
            self.quantization_type = 'gear'
        else:
            self.quantization_type = None

    def _validate_and_set_precision(self, bits):
        """Validate precision is supported by this hardware configuration."""
        if self.flops_per_precision is not None:
            if bits not in self.flops_per_precision:
                available = list(self.flops_per_precision.keys())
                raise ValueError(
                    f"Precision '{bits}' is not supported by this hardware. "
                    f"Available precisions: {available}"
                )
            # Update flops and op_per_sec for the selected precision
            self.flops = self.flops_per_precision[bits]
            self.op_per_sec = self.flops / 2

    def get_supported_precisions(self):
        """Return list of precisions supported by this hardware."""
        if self.flops_per_precision is not None:
            return list(self.flops_per_precision.keys())
        else:
            # Legacy mode: assume all precisions are supported
            return list(self.supported_precisions)

    def get_flops_for_precision(self, precision=None):
        """Get TFLOPS for a specific precision (or current precision if None)."""
        if precision is None:
            precision = self.bits
        if self.flops_per_precision is not None:
            if precision not in self.flops_per_precision:
                raise ValueError(f"Precision '{precision}' not supported. Available: {list(self.flops_per_precision.keys())}")
            return self.flops_per_precision[precision]
        else:
            # Legacy fallback
            return self.flops

    def __str__(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS ({self.bits}), Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Nodes = {self.num_nodes} \n"
        b = f"On-Chip mem size: {unit.raw_to_unit(self.on_chip_mem_size, type='M')} MB , Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')} MB\n"
        c = f"On-Chip mem BW: {unit.raw_to_unit(self.onchip_mem_bw, type='BW')} GB/s , Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s,\n"
        if self.flops_per_precision:
            d = f"Supported precisions: {list(self.flops_per_precision.keys())}\n"
            return a + b + c + d
        return a + b + c

    def get_params(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS ({self.bits}), Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Nodes = {self.num_nodes}"
        b = f" Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')/1024} GB "
        c = f" Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s"
        return a + b + c

    @classmethod
    def from_dict(cls, config_dict):
        init_params = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
        filtered_params = {k: v for k, v in config_dict.items() if k in init_params}
        return cls(**filtered_params)

    @classmethod
    def from_json(cls, json_str):
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def set_onchip_mem_bw(self,onchip_mem_bw):
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')

    def set_offchip_mem_bw(self,offchip_mem_bw):
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')

    def get_offchip_mem_bw(self):
        return self.unit.raw_to_unit(self.offchip_mem_bw,type='BW')

    def get_external_mem_bw(self):
        return self.unit.raw_to_unit(self.external_mem_bw,type='BW')

    def get_interchip_link_bw(self):
        return self.unit.raw_to_unit(self.interchip_link_bw,type='BW')

    def get_off_chip_mem_size(self):
        return self.unit.raw_to_unit(self.off_chip_mem_size,type='M')


    def claim_onchip_mem(self, data_sz):
        if data_sz > self.on_chip_mem_left_size:
            raise ValueError(f'Not enough on-chip memory: Need {data_sz}, only has {self.on_chip_mem_size}')
        self.on_chip_mem_left_size -= data_sz
        return self.on_chip_mem_left_size

    def release_onchip_mem(self, data_sz):
        self.on_chip_mem_left_size = max(self.on_chip_mem_size, data_sz + self.on_chip_mem_left_size)
        return self.on_chip_mem_left_size

    def get_bit_multiplier(self, type='C', data='w', operators=None):
        if type == 'C':
            # With per-precision Flops, compute multiplier is always 1.0
            # because flops already reflects the correct throughput for the precision
            return 1.0
        elif type == 'M':
            if self.quantization_type == 'gear':
                if data == 'k' or data == 'v':
                    return (    self.mem_multiplier[self.gear_b]
                                + (self.gear_s/100) * self.mem_multiplier[self.bits]
                                + ((np.prod(operators[:-2])/np.prod(operators)) * (operators[-2]*self.gear_r + operators[-1]*self.gear_r) * self.mem_multiplier[self.bits])
                    )
            return self.mem_multiplier[self.bits]
