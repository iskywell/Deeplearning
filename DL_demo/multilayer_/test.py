#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2022 2022/4/4 4:24 下午
@Author  : keyoung
@File    : test.py
"""

import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

y2 = torch.sigmoid(x)
d2l.plot(x.detach(), y2.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))