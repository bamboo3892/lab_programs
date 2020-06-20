import random
import pickle
import numpy as np
import torch
import pyro.distributions as dist
import openpyxl

from LDA_pytorch.ModelBase import MCMCModel
from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix
from utils.wordcloud_util import create_wordcloud


class MCLDA(MCMCModel):

    def __init__(self, args, data):
        """
        Parameters
        ----------
        args : object
            K,
            n_rh,  # number of categories for record "rh"
            auto_beta, auto_alpha,
            coef_beta, coef_alpha,
            device

        data : object
            [[doc[sentence[word, word, ...], ...], ...], [doc], ...],  # docs
            [Rm, D],  # measurements(continueous value)
            [Rh, D]]  # habits(categorical value)
        """
        super().__init__()

        docs = data[0]
        measurements = torch.tensor(data[1])  # [Rm, D]
        habits = torch.tensor(data[2])        # [Rh, D]

        # fields
        self.args = args
        self.data = data
        self.device = args.device

        self.K = K = args.K
        self.D = D = len(docs[0])
        self.Rt = len(docs)
        self.Rm = len(measurements)
        self.Rh = len(habits)
        self.V_rt = [0 for _ in range(self.Rt)]            # [Rt]
        self.totalN_rt = [0 for _ in range(self.Rt)]       # [Rt]
        self.n_rh = args.n_rh                              # [Rh]
        self.word_count_rt = [{} for _ in range(self.Rt)]  # [Rt][V_rt]
        self.word_dict_rt = [[] for _ in range(self.Rt)]   # [Rt][V_rt]
        self.wordids_rt = [[] for _ in range(self.Rt)]     # [Rt][totalN_rt]
        self.docids_rt = [[] for _ in range(self.Rt)]      # [Rt][totalN_rt]
        self.z_rt = [None for _ in range(self.Rt)]         # [Rt][totalN_rt]
        self.z_rm = None                                   # [Rm, D]
        self.z_rh = None                                   # [Rh, D]

        self.alpha = torch.full([K], args.coef_alpha, device=self.device, dtype=torch.float64)                           # [K]
        self.beta_rt = [None for _ in range(self.Rt)]                                                                    # [Rt][V_rt]
        self.mu_h_rm = torch.mean(measurements, 1)                                                                       # [Rm]
        self.sigma_h_rm = torch.std(measurements, 1)                                                                     # [Rm]
        self.rho_h_rh = [torch.ones([self.n_rh[rh]], device=self.device, dtype=torch.float64) for rh in range(self.Rh)]  # [Rh][n_rh]

        self._tpd = torch.zeros((D, K), device=self.device, dtype=torch.int64)  # [D, K]
        self._nd = torch.zeros(D, device=self.device, dtype=torch.int64)        # [D]
        self._wpt_rt = [None for _ in range(self.Rt)]                           # [Rt][K, V_rt]
        self._wt_rt = [None for _ in range(self.Rt)]                            # [Rt][K]

        # init wordcount...
        for rt, documents in enumerate(docs):
            for d, doc in enumerate(documents):
                for sentence in doc:
                    for word in sentence:
                        if(word not in self.word_count_rt[rt]):
                            self.word_count_rt[rt][word] = 0
                        self.word_count_rt[rt][word] += 1
            self.word_dict_rt[rt] = [a[0] for a in sorted(self.word_count_rt[rt].items(), key=lambda x: -x[1])]
            self.V_rt[rt] = len(self.word_dict_rt[rt])
            self.beta_rt[rt] = torch.full([self.V_rt[rt]], args.coef_beta, device=self.device, dtype=torch.float64)

        # init wordids_rt...
        for rt, documents in enumerate(docs):
            for d, doc in enumerate(documents):
                for sentence in doc:
                    for word in sentence:
                        self.wordids_rt[rt].append(self.word_dict_rt[rt].index(word))
                        self.docids_rt[rt].append(d)
            self.totalN_rt[rt] = len(self.wordids_rt[rt])
            self.wordids_rt[rt] = torch.tensor(self.wordids_rt[rt], device=self.device, dtype=torch.int64)
            self.docids_rt[rt] = torch.tensor(self.docids_rt[rt], device=self.device, dtype=torch.int64)
            self.z_rt[rt] = torch.randint(0, K, (self.totalN_rt[rt],), device=self.device, dtype=torch.int64)
        self.z_rm = torch.randint(0, K, (self.Rm, D), device=self.device, dtype=torch.int64)
        self.z_rh = torch.randint(0, K, (self.Rh, D), device=self.device, dtype=torch.int64)

        # init wpt_rt...
        for rt in range(self.Rt):
            ones = torch.ones(self.totalN_rt[rt], device=self.device, dtype=torch.int64)
            self._tpd.index_put_([self.docids_rt[rt], self.z_rt[rt]], ones, accumulate=True)
            self._wpt_rt[rt] = torch.zeros((K, self.V_rt[rt]), device=self.device, dtype=torch.int64)
            self._wpt_rt[rt].index_put_([self.z_rt[rt], self.wordids_rt[rt]], ones, accumulate=True)
            self._wt_rt[rt] = torch.sum(self._wpt_rt[rt], 1)
        self._nd = torch.sum(self._tpd, 1)


    def _sampling(self, idx):
        pass


    def _update_parameters(self):
        # TODO
        pass


    def log_perplexity(self, testset=None):
        return 0


    def summary(self, summary_args):
        pass
