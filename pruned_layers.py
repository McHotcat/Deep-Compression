import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.mask = np.ones([self.out_features, self.in_features])
        m = self.in_features
        n = self.out_features
        self.sparsity = 1.0
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weight = self.linear.weight.data.cpu().numpy()
        old_weight = weight.flatten()
        sorted_weight = np.sort(abs(old_weight))
        limit = sorted_weight[int(q/100*len(old_weight))]
        for i in range(len(old_weight)):
            if abs(old_weight[i]) < limit:
                old_weight[i] = 0
        new_weight = np.reshape(old_weight,(weight.shape[0],weight.shape[1]))
        self.linear.weight.data = torch.from_numpy(new_weight).cuda()
        self.sparsity = np.count_nonzero(new_weight==0)/len(new_weight)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if new_weight[i][j] == 0:
                    self.mask[i][j] = 0


    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """

        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weight = self.linear.weight.data.cpu().numpy()
        old_weight = np.reshape(weight,(weight.shape[0]*weight.shape[1]))
        limit = np.std(old_weight)*s
        for i in range(len(old_weight)):
            if abs(old_weight[i]) <= limit:
                old_weight[i] = 0
        new_weight = np.reshape(old_weight,(weight.shape[0],weight.shape[1]))
        self.linear.weight.data = torch.from_numpy(new_weight).cuda()
        self.sparsity = np.count_nonzero(new_weight==0)/len(new_weight)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if new_weight[i][j] == 0:
                    self.mask[i][j] = 0

class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Expand and Transpose to match the dimension
        self.mask = np.ones_like([out_channels, in_channels, kernel_size, kernel_size])

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weight = self.conv.weight.data.cpu().numpy()
        old_weight = weight.flatten()
        sorted_weight = np.sort(abs(old_weight))
        limit = sorted_weight[int(q/100*len(old_weight))]
        for i in range(len(old_weight)):
            if abs(old_weight[i]) <= limit:
                old_weight[i] = 0
        new_weight = np.reshape(old_weight,[self.out_channels, self.in_channels, 
            self.kernel_size, self.kernel_size])
        self.conv.weight.data = torch.from_numpy(new_weight).cuda()
        self.sparsity = np.count_nonzero(new_weight==0)/len(new_weight)
        mask =  new_weight.flatten()
        for i in range(len(mask)):
            if mask[i] !=0:
                mask[i] = 1
        self.mask = np.reshape(mask,[self.out_channels, self.in_channels, 
            self.kernel_size, self.kernel_size])

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weight = self.conv.weight.data.cpu().numpy()
        old_weight = np.reshape(weight,self.out_channels*self.in_channels*self.kernel_size*
            self.kernel_size)
        limit = np.std(old_weight)*s
        for i in range(len(old_weight)):
            if abs(old_weight[i]) < limit:
                old_weight[i] = 0
        new_weight = np.reshape(old_weight,[self.out_channels, self.in_channels, 
            self.kernel_size, self.kernel_size])
        self.conv.weight.data = torch.from_numpy(new_weight).cuda()
        self.sparsity = np.count_nonzero(new_weight==0)/len(new_weight)
        mask =  new_weight.flatten()
        for i in range(len(mask)):
            if mask[i] !=0:
                mask[i] = 1
        self.mask = np.reshape(mask,[self.out_channels, self.in_channels, 
            self.kernel_size, self.kernel_size])