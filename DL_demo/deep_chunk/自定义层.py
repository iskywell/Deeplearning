import torch
import torch.nn.functional as F
from torch import nn

#简单层的定义
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

# 测试
# layer = CenteredLayer()
# layer(torch.FloatTensor([1, 2, 3, 4, 5]))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# test
# Y = net(torch.rand(4, 8))
# Y.mean()

#带参数层的定义
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
#test
linear = MyLinear(5, 3)
# linear.weight

#test 前向传播 X = torch.rand(2,5)
# linear(torch.rand(2, 5))

# 嵌入到顺序流
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))