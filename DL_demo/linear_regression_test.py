#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2022 2022/3/29 4:15 下午
@Author  : keyoung
@File    : linear_regression_test.py
"""
# 通过使用深度学习框架来简洁地实现线性回归模型
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#读取一小部分数据集
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# next(iter(data_iter))
# print(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1))

#随机初始化参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
#损失函数 计算均方误差
loss = nn.MSELoss()

#优化定义
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)