import openpyxl
import numpy as np
import torch
import pyro.distributions as dist

from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix
from utils.wordcloud_util import create_wordcloud

DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

# b = torch.zeros((3), device=DEVICE)
# # b.index_put_([torch.tensor([0, 0], device=DEVICE)], torch.tensor(1., device=DEVICE), accumulate=True)
# # b.index_put_([torch.tensor([0, 0], device=DEVICE)], torch.tensor([1., 1.], device=DEVICE), accumulate=True)
# # b.index_add_(0, torch.tensor([0, 0], device=DEVICE), torch.tensor(1., device=DEVICE))
# b.index_add_(0, torch.tensor([0, 0], device=DEVICE), torch.tensor([1., 1.], device=DEVICE))

# create_wordcloud(["apple", "banana", "orange"], "aaa.png")

# m = torch.tensor([0, 0], dtype=torch.float64)
# s = torch.tensor([1, 1], dtype=torch.float64)
# x = torch.randn([5, 2])
# a = dist.Normal(m, s).log_prob(x).exp()

# b = torch.zeros((3), device=DEVICE)
# torch.save(b, "model.pickle")

# b = torch.tensor([0.4, float('nan'), -1], device=DEVICE)
# # if(torch.any(torch.isnan(b))):
# #     print("nan")
# # if(torch.any(b <= 0)):
# #     print("negative")
# a = dist.Categorical(b).sample()

a = float("")

print("aaaaaa")
