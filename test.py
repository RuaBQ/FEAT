import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


m = nn.AdaptiveAvgPool1d(1)
input = torch.randn(2, 2, 2)
output = m(input)
avgout = torch.mean(input, dim=1, keepdim=True)
maxout, _ = torch.max(input, dim=1, keepdim=True)
x = torch.cat([avgout, maxout], dim=1)
print(output)
print(output.chunk(3, dim=1)[0].shape)

a = torch.randn(2, 1, 2)
b = torch.randn(2, 1, 2)
c = torch.randn(2, 1, 2)

print(a)
print(b)
print(c)
a, b, c = torch.mean(torch.cat([
    a.view(*(-1,)+(1, 2)),
    b.view(*(-1,)+(1, 2)),
    c.view(*(-1,)+(1, 2))], dim=1), dim=2).chunk(3, dim=1)
print(a)
print(b)
print(c)
