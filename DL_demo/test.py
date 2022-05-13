#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2022 2022/3/24 10:23 上午
@Author  : keyoung
@File    : test.py
"""
import torch

# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# H = X.sum(0, keepdim=True)
# Y = X.sum(1, keepdim=True)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat[[0, 1], y] 在y_hat中找出第y[0](=0) 和 y[1](=2)个元素的概率 即找到真实的概率

#交叉熵损失
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

# cross_entropy(y_hat, y)

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

#   accuracy(y_hat, y) / len(y)
# 我们将继续使用之前定义的变量y_hat和y分别作为预测的概率分布和标签。
# 可以看到，第一个样本的预测类别是2（该行的最大元素为0.6，索引为2），这与实际标签0不一致。
# 第二个样本的预测类别是2（该行的最大元素为0.5，索引为2），这与实际标签2一致。
# 因此，这两个样本的分类精度率为0.5。