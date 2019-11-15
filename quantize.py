import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            weight = m.conv.weight.data.cpu().numpy()
            old_weight = weight.flatten()
            old_weight_nonzero = old_weight[old_weight != 0]
            old_weight_nonzero = old_weight.reshape(-1,1)
            kmeans = KMeans(n_clusters=2**bits,random_state=0)
            kmeans.fit(old_weight_nonzero)
            centroids = kmeans.cluster_centers_.flatten()
            cluster_centers.append(centroids)
            new_weight = []
            for i in range(len(old_weight)):
                if old_weight[i] == 0:
                    new_weight.append(0)
                else:
                    new_weight.append(centroids[kmeans.predict(np.array([[old_weight[i]]]))[0]])
            new_weight = np.reshape(np.array(new_weight),[m.out_channels, m.in_channels, m.kernel_size, m.kernel_size])
            m.conv.weight.data = torch.from_numpy(new_weight).float().cuda()           
            """
            --------------------------------------------
            """
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            weight = m.linear.weight.data.cpu().numpy()
            old_weight = weight.flatten()
            old_weight_nonzero = old_weight[old_weight != 0]
            old_weight_nonzero = old_weight.reshape(-1,1)
            kmeans = KMeans(n_clusters=2**bits,random_state=0)
            kmeans.fit(old_weight_nonzero)
            centroids = kmeans.cluster_centers_.flatten()
            cluster_centers.append(centroids)
            new_weight = []
            for i in range(len(old_weight)):
                if old_weight[i] == 0:
                    new_weight.append(0)
                else:
                    new_weight.append(centroids[kmeans.predict(np.array([[old_weight[i]]]))[0]])
            new_weight = np.reshape(np.array(new_weight),[weight.shape[0],weight.shape[1]])
            m.linear.weight.data = torch.from_numpy(new_weight).float().cuda()
            """
            --------------------------------------------
            """
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

