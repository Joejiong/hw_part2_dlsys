import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_apis import MatrixMuliplication
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')


class LeNet_300_100(nn.Module):
    def __init__(self, prune):
        super(LeNet_300_100, self).__init__()

        # self._w1_T = nn.Parameter(torch.randn(784, 300))
        # self._w2_T = nn.Parameter(torch.randn(300, 100))
        # self._bias1 = nn.Parameter(torch.randn(300))
        # self._bias2 = nn.Parameter(torch.randn(100))
        self.weight1 = nn.Parameter(torch.empty(784, 300))
        self.bias1 = nn.Parameter(torch.empty(300))

        self.weight2 = nn.Parameter(torch.empty(300,100))
        self.bias2 = nn.Parameter(torch.empty(100))
        self.reset_parameters()
        # self.ip1 = nn.Linear(28*28, 300) 784*300
        # self.ip1 = sgemm(x, self._w_T) + self._bias

        self.relu_ip1 = nn.ReLU(inplace=True)
        
        # self.ip2 = nn.Linear(300, 100) # 300*100
        # self.ip1 = sgemm(x, self._w_T) + self._bias 300*100

        self.relu_ip2 = nn.ReLU(inplace=True)
       
        self.ip3 = nn.Linear(100, 10)
        # softmax

        self.device = torch.device('cuda')
        return
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias1, -bound, bound)

        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias2 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias2, -bound, bound)
        

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        # x = self.ip1(x)
        x = MatrixMuliplication(x, self.weight1, x.size(0), 300, self.device) + self.bias1
        x = self.relu_ip1(x)
        
        # x = self.ip2(x)
        x = MatrixMuliplication(x, self.weight2, x.size(0), 100, self.device) + self.bias2
        x = self.relu_ip2(x)
        
        x = self.ip3(x)
        # x = F.softmax(x, dim=1)
        return x