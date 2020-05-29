from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix
import openpyxl
import numpy as np

import torch
import pyro.distributions as dist

a = torch.rand((10, 100))
idx = torch.tensor([1])
b = a[idx, idx]

print("aaaaaa")
