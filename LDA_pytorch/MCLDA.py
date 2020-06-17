import random
import pickle
import numpy as np
import torch
import pyro.distributions as dist
import openpyxl

from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix


class MCLDA:


    def __init__(self, args, data):
        """
        pytorchでgibbs sampling

        data:
            [doc[sentence[word, word, ...], ...], ...]
        args:
            K,
            auto_beta, auto_alpha,  # bool
            coef_beta, coef_alpha,  # float
            device
        """

        pass
