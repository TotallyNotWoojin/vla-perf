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

# based on the first letters of the units trying to get values in readable form.
class Unit(object):
    unit_dicts = {'K': 1e3, 'M': 1e6, 'G':1e9, 'T':1e12, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}
    binary_dicts = {'K': 2**10, 'M': 2**20, 'G':2**30, 'T':2**40, 'm': 2**-10, 'u': 2**-20, 'n': 2**-30, 'p': 2**-40}

    def __init__(self, unit_mem='MB', unit_compute='Tflops',
                    unit_time='msec', unit_bw='GBsec', unit_freq='MHz',
                    unit_energy='pJ', unit_flop='MFLOP'):
        self.unit_mem = unit_mem
        self.unit_compute = unit_compute
        self.unit_time = unit_time
        self.unit_bw = unit_bw
        self.unit_freq = unit_freq
        self.unit_energy = unit_energy
        self.unit_flop = unit_flop

    def get_unit_value(self, type):
        if type =='C':          ## Compute
            unit_value = self.unit_dicts[self.unit_compute[0]]
        elif type == 'M':       ## Memory
            unit_value = self.binary_dicts[self.unit_mem[0]]
        elif type == 'T':       ## Time
            unit_value = self.unit_dicts[self.unit_time[0]]
        elif type == 'BW':      ## Bandwidth
            unit_value = self.binary_dicts[self.unit_bw[0]]
        elif type == 'F':       ## Frequency
            unit_value = self.unit_dicts[self.unit_freq[0]]
        elif type == 'E':       ## Energy
            unit_value = self.unit_dicts[self.unit_energy[0]]
        elif type == 'O':       ## Floting point operations
            unit_value = self.unit_dicts[self.unit_flop[0]]
        else:
            raise ValueError(f'Wrong unit type: {type}')
        return unit_value

    def raw_to_unit(self, data, type='C'):
        unit_value = self.get_unit_value(type=type)
        return data / unit_value

    def unit_to_raw(self, data, type='C'):
        unit_value = self.get_unit_value(type=type)
        return data * unit_value

