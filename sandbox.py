import openpyxl
import numpy as np
import torch
import pyro.distributions as dist
import cv2
import time
import multiprocessing
import psutil
import time
import os

from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix
from utils.wordcloud_util import create_wordcloud
from utils.graphic_util import makeColorMap


def main():

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

    # a = float("")

    # x = np.random.randn(5, 10)
    # m = np.full_like(x, True)
    # img = makeColorMap(x, 20, 20, axis=0, border_mask=m)
    # cv2.imwrite("img.png", img)

    # a = np.full((5, 5), 10)
    # c = np.full((5, 5), 20)
    # b = np.full((5, 5), False)
    # b[2, 3] = True
    # a[b] = c[b]

    # a = torch.randn((5, 6))
    # c = torch.arange(0, 5)
    # b = torch.argmax(a, dim=1)
    # a[c, b] = 100


#     multiprocessing.set_start_method('spawn')
#     print(psutil.Process().nice(-20))
#     processes = 4
#     with multiprocessing.Pool(processes=processes, maxtasksperchild=None) as p:
#         p.map(func, np.arange(10).tolist())
#         # p.map(func, np.arange(10).tolist(), chunksize=(10 - 1) // processes + 1)

    a = [None]

    print("aaaaaa")


def func(num):
    print(f"start: {num}, pid: {os.getpid()}, nice: {psutil.Process().nice()}")
    time.sleep(10)


if __name__ == '__main__':
    main()
