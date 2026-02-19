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

class ParallelismConfig():
    r"""
    This is the configuration class to store the configuration of a Model Splitting.
    It is used to instantiate an LLM into multiple parallel units
    according to the specified arguments, defining the degree of various parallelism.
    Args:

    """
    def __init__(
        self,
        tensor_parallel=1,
        pipeline_parallel=1,
        data_parallel=1,
        expert_parallel=1,
        sequence_parallel=1,
        **kwargs,
    ):
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        self.data_parallel = data_parallel
        self.expert_parallel = expert_parallel
        self.sequence_parallel = sequence_parallel
        self.total_chips = np.prod([
                            self.data_parallel,
                            self.expert_parallel,
                            self.sequence_parallel,
                            self.pipeline_parallel,
                            self.tensor_parallel])

        super().__init__(**kwargs)

    def __str__(self):
        return str(vars(self))