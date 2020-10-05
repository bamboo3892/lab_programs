# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from utils.general_util import min_max_normalize, simple_moving_average
from main_ubuntu import RESULT
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')


os.chdir("/home/takeda.masaki/lab/program")


# %%
pathResult = RESULT.joinpath("multi_channel", "torch", "MCLDA", "tmp")
history = torch.load(pathResult.joinpath("history.pickle"))
phis = np.array([hist["phis"] for hist in history])  # [step, rt, k, v]
mus = np.array([hist["mus"] for hist in history])  # [step, rm, k]
rhos = np.array([hist["rhos"] for hist in history])  # [step, rh, k]

num = 50
window = np.ones(num) / num

phis = simple_moving_average(phis, window)
mus = simple_moving_average(mus, window)
rhos = simple_moving_average(rhos, window)


# %%
# phi, r固定, v固定, k間比較

fig = plt.figure(figsize=[25, 10])
rt = 0
for v in range(10):
    ax = fig.add_subplot(2, 5, v + 1)
    for k in range(10):
        ax.set_title(f"vocab{v+1}")
        ax.plot(phis[:, rt, k, v], label=f"topic{k+1}")
fig.legend([f"topic{k+1}" for k in range(10)])
fig.show()


# %%
# phi, rt固定, k固定, v間比較

fig = plt.figure(figsize=[25, 10])
rt = 0
phis_n = min_max_normalize(phis[:, rt, :, :], axis=(0, 1))
for k in range(10):
    ax = fig.add_subplot(2, 5, k + 1)
    ax.set_ylim((0, 1))
    for v in range(10):
        ax.set_title(f"topic{k+1}")
        ax.plot(phis_n[:, k, v], label=f"vocab{v+1}")
fig.legend([f"vocab{v+1}" for v in range(10)])
fig.show()


# %%
# mu, r固定, k間比較

fig = plt.figure(figsize=[25, 10])

for rm in range(12):
    ax = fig.add_subplot(3, 5, rm + 1)
    for k in range(10):
        ax.set_title(f"rm={rm+1}")
        ax.plot(mus[:, rm, k])
fig.legend([f"topic{k+1}" for k in range(10)])
fig.show()


# %%
# mu, k固定, r間比較 (step, rに関して正規化)

mus_n = min_max_normalize(mus, axis=(0, 2))

fig = plt.figure(figsize=[25, 10])

for k in range(10):
    ax = fig.add_subplot(2, 5, k + 1)
    ax.set_ylim((0., 1.))
    for rm in range(12):
        ax.set_title(f"topic{k+1}")
        ax.plot(mus_n[:, rm, k])
fig.legend([f"rm={rm+1}" for rm in range(12)])
fig.show()


# %%
# rho, r固定, k間比較

fig = plt.figure(figsize=[25, 10])

for rh in range(11):
    ax = fig.add_subplot(3, 5, rh + 1)
    for k in range(10):
        ax.set_title(f"rh={rh+1}")
        ax.plot(rhos[:, rh, k])
fig.legend([f"topic{k+1}" for k in range(10)])
fig.show()


# %%
# rho, k固定, r間比較 (step, rに関して正規化)

rhos_n = min_max_normalize(rhos, axis=(0, 2))

fig = plt.figure(figsize=[25, 10])

for k in range(10):
    ax = fig.add_subplot(2, 5, k + 1)
    ax.set_ylim((0., 1.))
    for rh in range(11):
        ax.set_title(f"topic{k+1}")
        ax.plot(rhos_n[:, rh, k])
fig.legend([f"rh={rh+1}" for rh in range(11)])
fig.show()
