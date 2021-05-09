import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


m = nn.BatchNorm1d(5)
input = torch.randn(1, 5, 540)
output = m(input)
avgout = torch.mean(input, dim=1, keepdim=True)
maxout, _ = torch.max(input, dim=1, keepdim=True)
x = torch.cat([avgout, maxout], dim=1)
print(avgout.size())
