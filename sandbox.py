import openpyxl
import numpy as np
import torch
import pyro.distributions as dist

from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix
from utils.wordcloud_util import create_wordcloud

# DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

# b = torch.zeros((3), device=DEVICE)
# # b.index_put_([torch.tensor([0, 0], device=DEVICE)], torch.tensor(1., device=DEVICE), accumulate=True)
# # b.index_put_([torch.tensor([0, 0], device=DEVICE)], torch.tensor([1., 1.], device=DEVICE), accumulate=True)
# # b.index_add_(0, torch.tensor([0, 0], device=DEVICE), torch.tensor(1., device=DEVICE))
# b.index_add_(0, torch.tensor([0, 0], device=DEVICE), torch.tensor([1., 1.], device=DEVICE))

create_wordcloud(["apple", "banana", "orange"], "aaa.png")

print("aaaaaa")
