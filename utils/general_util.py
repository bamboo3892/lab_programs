import numpy as np


class Args:

    def __init__(self):
        pass


class cycleArray:

    def __init__(self, array):
        self.array = array

    def __getitem__(self, item):
        return self.array[item % len(self.array)]


def min_max_normalize(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result


def simple_moving_average(x, window):
    num = len(window)

    if(len(x.shape) == 2):
        rtn = np.zeros((x.shape[0] - num + 1, x.shape[1]))
        for i1 in range(x.shape[1]):
            rtn[:, i1] = np.convolve(x[:, i1], window, mode='valid')
        return rtn

    if(len(x.shape) == 3):
        rtn = np.zeros((x.shape[0] - num + 1, x.shape[1], x.shape[2]))
        for i1 in range(x.shape[1]):
            for i2 in range(x.shape[2]):
                rtn[:, i1, i2] = np.convolve(x[:, i1, i2], window, mode='valid')
        return rtn

    elif(len(x.shape) == 4):
        rtn = np.zeros((x.shape[0] - num + 1, x.shape[1], x.shape[2], x.shape[3]))
        for i1 in range(x.shape[1]):
            for i2 in range(x.shape[2]):
                for i3 in range(x.shape[3]):
                    rtn[:, i1, i2, i3] = np.convolve(x[:, i1, i2, i3], window, mode='valid')
        return rtn
