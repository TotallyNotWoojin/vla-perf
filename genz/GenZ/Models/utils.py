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


from enum import IntEnum

class OpType(IntEnum):
    ## MAINTIAIN PARITY IN operator.py and operator_base.py
    FC = 0
    CONV2D = 1
    DWCONV = 2
    GEMM = 3
    Logit = 4
    Attend = 5
    Sync = 6
    Logit_BM_PREFILL = 9
    Attend_BM_PREFILL = 10
    CONV1D = 11
    EINSUM = 12
    REPEAT = 13
    ENDREPEAT = 14
    Norm = 15
    Avg = 16
    Special_Func = 17


class ResidencyInfo(IntEnum):
    All_offchip = 0
    A_onchip = 1
    B_onchip = 2
    C_onchip = 3
    AB_onchip = 4
    AC_onchip = 5
    BC_onchip = 6
    All_onchip = 7

from enum import IntEnum

class CollectiveType(IntEnum):
    AllReduce = 1
    All2All = 2
    AllGather = 3
    ReduceScatter = 4
    MessagePass = 5

class SpecialFuncType(IntEnum):
    gelu = 0
    relu = 1
    softmax = 2
    tanh = 3
    silu = 4
    gelu_pytorch_tanh = 5
    gelu_new = 6
    gegelu = 7

class NormType(IntEnum):
    LayerNorm = 0
    BatchNorm = 1
    InstanceNorm = 2
    GroupNorm = 3


def parse_einsum_expression(expression, *tensors):
    einsum_vars = {}
    input_subscripts, output_subscript = expression.split('->')
    input_subscripts = input_subscripts.split(',')

    for tensor, subscripts in zip(tensors, input_subscripts):
        for dim, subscript in zip(tensor, subscripts):
            if subscript not in einsum_vars:
                einsum_vars[subscript] = dim
            elif einsum_vars[subscript] != dim:
                raise ValueError(f"Dimension mismatch for subscript: {subscript}, Got: {dim}, Expected: {einsum_vars[subscript]}")

    for subscript in output_subscript:
        if subscript not in einsum_vars:
            einsum_vars[subscript] = None

    return einsum_vars