import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: 
            'encodings': Encoding map mapping each weight parameter to its Huffman coding.
            'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    """
    """
    Generate Huffman Coding and Frequency Map according to incoming weights and centers (KMeans centriods).
    --------------Your Code---------------------
    """
    frequency = {}
    encodings = {}
    weight_1d = weight.flatten()
    for i in centers:
        frequency[i] = 0
    for i in weight_1d[weight_1d != 0]:
        frequency[i] += 1
    Node_list = []
    for i in range(len(frequency)):
        Node = node(list(frequency)[i],list(frequency.values())[i])
        Node_list.append(Node)
    tree = generate_tree(Node_list)
    tree.get_all_code()
    encodings = tree.encodings
    return encodings, frequency

class node():
    def __init__(self,key,frequency):
        self.key = key
        self.frequency = frequency
        self.left = None
        self.right = None

class generate_tree():
    def __init__(self,Node_list):
        self.Node_list = Node_list
        self.length = len(Node_list)
        self.encodings = {}
        while len(self.Node_list) > 1:
            self.Node_list.sort(key=lambda node:node.frequency,reverse=True)
            left = self.Node_list.pop()
            right = self.Node_list.pop()
            new_node = node(left.key+right.key,left.frequency+right.frequency)
            new_node.left = left
            new_node.right = right
            self.Node_list.append(new_node)
        self.root = self.Node_list[0]

    def get_all_code(self):
        path = ''
        self.get_code(self.root,path)

    def get_code(self,node,path):
        if node is None:
            return
        if node.left:
            self.get_code(node.left,'1'+path)
        if node.right:
            self.get_code(node.right,'0'+path)
        if node.left is None and node.right is None:
            self.encodings[node.key] = path
        




def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param 'encodings': Encoding map mapping each weight parameter to its Huffman coding.
    :param 'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)

    return freq_map, encodings_map