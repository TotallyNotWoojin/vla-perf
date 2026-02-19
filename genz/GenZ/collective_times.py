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


def get_AR_time(data, numNodes, system):
    """get_AR_time

    Args:
        data (int): Message size(Bytes) per node to complete all reduce.
        num_AR_nodes (int): Number of nodes among which all-reduce is performed
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to complete the All-Reduce
    """
    if data == 0 or numNodes == 1:
        return 0
    ## Ring AR Time = Start Latency + N*Tlink +  2M*(N-1)/(N*BW)
    ## Source:  https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/
    allReduceTime = (5e-6 + 2*(numNodes-1)*system.interchip_link_latency +  2 * (numNodes-1) * (data/numNodes) / system.interchip_link_bw)*1000

    return allReduceTime

def get_AG_time(data, numNodes, system):
    """get_AG_time

    Args:
        data (int): Message size(Bytes) per node to complete all gather.
        num_AG_nodes (int): Number of nodes among which all-gather is performed
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to complete the All-Gather
    """
    if data == 0 or numNodes == 1:
        return 0
    ## Ring AG Time = Start Latency + N*Tlink +  2M*(N-1)/(N*BW)
    ## Source:  https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/
    allGatherDuration = (5e-6 + (numNodes-1)*system.interchip_link_latency +  (numNodes-1) * (data/numNodes) / system.interchip_link_bw)*1000

    return allGatherDuration

def get_message_pass_time(data, system):
    """get_message_pass_time

    Args:
        data (int): Message size(Bytes) per node to pass from 1 decide to next.
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to pass the Message from 1 node to next
    """
    if data == 0:
        return 0
    msg_pass_time = (system.interchip_link_latency +  data / system.interchip_link_bw)*1000
    return msg_pass_time


def get_A2A_time(data, numNodes, system):
    """get_A2A_time

    Args:
        data (int): Total message size (Bytes) per node to be exchanged in all-to-all.
        num_A2A_nodes (int): Number of nodes participating in the all-to-all operation.
        system (System object): Object of class System

    Returns:
        time (float): Total time (msec) to complete the All-to-All operation
    """

    ## BWeff = 4B/N if Ring of size N
    ## BWeff = 4B/T if 2D Torus of size TxT

    # M = E/ep * D/tp * F * B * bb

    # A2A time = Start Latency + (N-1) * Tlink + (N-1) * M / BW
    # Where N is the number of nodes, M is the message size per node pair,
    # Tlink is the inter-chip link latency, and BW is the inter-chip memory bandwidth

    if data == 0 or numNodes == 1:
        return 0
    message_size_per_pair = data / numNodes
    A2A_time = (5e-6 + (numNodes - 1) * system.interchip_link_latency +
                (numNodes - 1) * message_size_per_pair / system.interchip_link_bw) * 1000

    return A2A_time