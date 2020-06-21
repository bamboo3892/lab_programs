import random
import pickle
import numpy as np
import torch
import pyro.distributions as dist
import openpyxl

from LDA_pytorch.ModelBase import LDABase
from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix
from utils.wordcloud_util import create_wordcloud


class MCLDA(LDABase):

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
        measurements = torch.tensor(data[1], device=args.device, dtype=torch.float64)  # [Rm, D]
        habits = torch.tensor(data[2], device=args.device, dtype=torch.int64)        # [Rh, D]

        # fields
        self.args = args
        self.data = data
        self.device = args.device

        self.K = K = args.K
        self.D = D = len(docs[0]) if len(docs) != 0 else (len(measurements[0]) if len(measurements) != 0 else (len(habits[0]) if len(habits) != 0 else 0))
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
        self.x_rm = measurements                           # [Rm, D]
        self.x_rh = habits                                 # [Rh, D]
        self.z_rm = None                                   # [Rm, D]
        self.z_rh = None                                   # [Rh, D]

        self.alpha = torch.full([K], args.coef_alpha, device=self.device, dtype=torch.float64)                           # [K]
        self.beta_rt = [None for _ in range(self.Rt)]                                                                    # [Rt][V_rt]
        # self.mu_h_rm = torch.mean(measurements, 1)                                                                       # [Rm]
        # self.sigma_h_rm = torch.std(measurements, 1)                                                                     # [Rm]
        self.rho_h_rh = [torch.ones([self.n_rh[rh]], device=self.device, dtype=torch.float64) for rh in range(self.Rh)]  # [Rh][n_rh]

        self._tpd = torch.zeros((D, K), device=self.device, dtype=torch.int64)              # [D, K]
        self._nd = torch.zeros(D, device=self.device, dtype=torch.int64)                    # [D]
        self._wpt_rt = [None for _ in range(self.Rt)]                                       # [Rt][K, V_rt]
        self._wt_rt = [None for _ in range(self.Rt)]                                        # [Rt][K]
        self._mean_rm = torch.zeros((self.Rm, K), device=self.device, dtype=torch.float64)  # [Rm, K]
        self._std_rm = torch.zeros((self.Rm, K), device=self.device, dtype=torch.float64)   # [Rm, K]
        self._xpt_rh = [None for _ in range(self.Rh)]                                       # [Rh][K, n_rh]
        self._xt_rh = [None for _ in range(self.Rh)]                                        # [Rh][K]

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

        # init latent variables...
        for rt in range(self.Rt):
            self.z_rt[rt] = torch.randint(0, K, (self.totalN_rt[rt],), device=self.device, dtype=torch.int64)
            ones = torch.ones(self.totalN_rt[rt], device=self.device, dtype=torch.int64)
            self._tpd.index_put_([self.docids_rt[rt], self.z_rt[rt]], ones, accumulate=True)
            self._wpt_rt[rt] = torch.zeros((K, self.V_rt[rt]), device=self.device, dtype=torch.int64)
            self._wpt_rt[rt].index_put_([self.z_rt[rt], self.wordids_rt[rt]], ones, accumulate=True)
            self._wt_rt[rt] = torch.sum(self._wpt_rt[rt], 1)
        self.z_rm = torch.randint(0, K, [self.Rm, D], device=self.device, dtype=torch.int64)
        ones = torch.ones(self.D, device=self.device, dtype=torch.int64)
        idx = torch.arange(0, self.D, device=self.device, dtype=torch.int64)
        for rm in range(self.Rm):
            self._tpd.index_put_([idx, self.z_rm[rm]], ones, accumulate=True)
            for k in range(K):
                self._mean_rm[rm, k] = torch.mean(self.x_rm[rm][self.z_rm[rm] == k])
                self._std_rm[rm, k] = torch.std(self.x_rm[rm][self.z_rm[rm] == k])
        self.z_rh = torch.randint(0, K, [self.Rh, D], device=self.device, dtype=torch.int64)
        for rh in range(self.Rh):
            self._tpd.index_put_([idx, self.z_rh[rh]], ones, accumulate=True)
            self._xpt_rh[rh] = torch.zeros((K, self.n_rh[rh]), device=self.device, dtype=torch.int64)
            self._xpt_rh[rh].index_put_([self.z_rh[rh], self.x_rh[rh]], ones, accumulate=True)
            self._xt_rh[rh] = torch.sum(self._xpt_rh[rh], 1)
        self._nd = torch.sum(self._tpd, 1)


    def step(self, num_subsample_partitions, parameter_update=False):
        """
        1. update z, wpt, ...
        2. update parameters if neseccary
        """

        # generate sabsampling batches
        rand_perm_rt = []
        rand_perm_rm = []
        rand_perm_rh = []
        for rt in range(self.Rt):
            rand_perm_rt.append(torch.randperm(self.totalN_rt[rt], device=self.device))
        for rm in range(self.Rm):
            rand_perm_rm.append(torch.randperm(self.D, device=self.device))
        for rh in range(self.Rh):
            rand_perm_rh.append(torch.randperm(self.D, device=self.device))

        # sampling
        for n in range(num_subsample_partitions):
            # texts
            for rt in range(self.Rt):
                s = (self.totalN_rt[rt] // num_subsample_partitions + 1) * n
                e = (self.totalN_rt[rt] // num_subsample_partitions + 1) * (n + 1) - 1
                e = e if e < self.totalN_rt[rt] else self.totalN_rt[rt] - 1
                self._sampling_rt(rt, rand_perm_rt[rt][s:e])
            # measurements
            for rm in range(self.Rm):
                s = (self.D // num_subsample_partitions + 1) * n
                e = (self.D // num_subsample_partitions + 1) * (n + 1) - 1
                e = e if e < self.D else self.D - 1
                self._sampling_rm(rm, rand_perm_rm[rm][s:e])
            # habits
            for rh in range(self.Rh):
                s = (self.D // num_subsample_partitions + 1) * n
                e = (self.D // num_subsample_partitions + 1) * (n + 1) - 1
                e = e if e < self.D else self.D - 1
                self._sampling_rh(rh, rand_perm_rh[rh][s:e])

        # update parameters
        if(parameter_update):
            self._update_parameters()

        """ calculation checking """
        # for i in range(100):
        #     k = random.randrange(self.K)
        #     d = random.randrange(self.D)
        #     s = torch.zeros(1, device=self.device, dtype=torch.int64)
        #     for rt in range(self.Rt):
        #         s += torch.sum(torch.logical_and(self.docids_rt[rt] == d, self.z_rt[rt] == k))
        #     s += torch.sum(self.z_rm[:, d] == k)
        #     s += torch.sum(self.z_rh[:, d] == k)
        #     assert self._tpd[d, k] == s

        return self.log_perplexity()


    def _sampling_rt(self, rt, idx):
        """
        1. sampleing z
        2. update wpt, tpd, wt
        """

        before = self.z_rt[rt][idx].clone()

        # zをサンプリング
        probs = self._get_z_rt_sampling_probs(rt, idx)
        self.z_rt[rt][idx] = dist.Categorical(probs).sample()

        # wptとか更新
        ones = torch.ones(len(idx), device=self.device, dtype=torch.int64)
        self._wpt_rt[rt].index_put_([before, self.wordids_rt[rt][idx]], -ones, accumulate=True)
        self._wpt_rt[rt].index_put_([self.z_rt[rt][idx], self.wordids_rt[rt][idx]], ones, accumulate=True)
        self._tpd.index_put_([self.docids_rt[rt][idx], before], -ones, accumulate=True)
        self._tpd.index_put_([self.docids_rt[rt][idx], self.z_rt[rt][idx]], ones, accumulate=True)
        self._wt = torch.sum(self._wpt_rt[rt], 1)

        """ calculation checking """
        # for i in range(100):
        #     k = random.randrange(self.K)
        #     v = random.randrange(self.V_rt[rt])
        #     assert self._wpt_rt[rt][k, v] == torch.sum(torch.logical_and(self.z_rt[rt] == k, self.wordids_rt[rt] == v))


    def _get_z_rt_sampling_probs(self, rt, idx):
        subsample_size = len(idx)
        probs = torch.ones((subsample_size, self.K), device=self.device, dtype=torch.float64)  # [subsample_size, K]
        a = torch.zeros((subsample_size, self.K), device=self.device, dtype=torch.float64)     # [subsample_size, K]
        a[torch.arange(0, subsample_size, device=self.device, dtype=torch.int64), self.z_rt[rt][idx]] = 1

        probs *= self._wpt_rt[rt][:, self.wordids_rt[rt][idx]].T + self.beta_rt[rt][self.wordids_rt[rt][idx]][:, None] - a
        probs /= self._wt_rt[rt][None, :] + self.beta_rt[rt].sum() - a
        probs *= self._tpd[self.docids_rt[rt][idx], :] + self.alpha[None, :] - a
        probs /= self._nd[self.docids_rt[rt][idx]][:, None] + self.alpha.sum() - a

        """ calculation checking """
        # for i in range(100):
        #     n = random.randrange(subsample_size)  # any
        #     k = random.randrange(self.K)  # any
        #     z = self.z_rt[rt][idx[n]]
        #     v = self.wordids_rt[rt][idx[n]]
        #     d = self.docids_rt[rt][idx[n]]
        #     a0 = (1 if z == k else 0)
        #     assert a[n, k] == a0
        #     assert(probs[n, k] == (self._wpt_rt[rt][k, v] - a0 + self.beta_rt[rt][v]) / (self._wt_rt[rt][k] + self.beta_rt[rt].sum() - a0)
        #            * (self._tpd[d, k] - a0 + self.alpha[k]) / (self._nd[d] + self.alpha.sum() - a0))

        probs /= torch.sum(probs, dim=1, keepdim=True)
        return probs


    def _sampling_rm(self, rm, idx):
        """
        1. sampleing z
        2. update tpd, mean_rm, std_rm
        """

        before = self.z_rm[rm][idx].clone()

        # zをサンプリング
        probs = self._get_z_rm_sampling_probs(rm, idx)
        self.z_rm[rm][idx] = dist.Categorical(probs).sample()

        # tpdとか更新
        ones = torch.ones(len(idx), device=self.device, dtype=torch.int64)
        self._tpd.index_put_([idx, self.z_rm[rm][idx]], ones, accumulate=True)
        self._tpd.index_put_([idx, before], -ones, accumulate=True)
        for k in range(self.K):
            self._mean_rm[rm, k] = torch.mean(self.x_rm[rm][self.z_rm[rm] == k])
            self._std_rm[rm, k] = torch.std(self.x_rm[rm][self.z_rm[rm] == k])


    def _get_z_rm_sampling_probs(self, rm, idx):
        """
        本来_mean_rmと_std_rmは自分の影響を除いて計算するべきだけど，近似的にかわらないとした
        """
        subsample_size = len(idx)
        probs = torch.ones((subsample_size, self.K), device=self.device, dtype=torch.float64)                    # [subsample_size, K]
        a = torch.zeros((subsample_size, self.K), device=self.device, dtype=torch.float64)                       # [subsample_size, K]
        a[torch.arange(0, subsample_size, device=self.device, dtype=torch.int64), self.z_rm[rm][idx]] = 1
        x = torch.zeros([subsample_size, self.K], device=self.device, dtype=torch.float64) + self.x_rm[rm, idx][:, None]  # [subsample_size, K]

        probs *= self._tpd[idx, :] + self.alpha[None, :] - a
        probs /= self._nd[idx][:, None] + self.alpha.sum() - a
        probs *= dist.Normal(self._mean_rm[rm], self._std_rm[rm]).log_prob(x).exp()

        """ calculation checking """
        # for i in range(100):
        #     n = random.randrange(subsample_size)  # any
        #     k = random.randrange(self.K)  # any
        #     d = idx[n]
        #     z = self.z_rm[rm][idx[n]]
        #     a0 = (1 if z == k else 0)
        #     x = self.x_rm[rm, d]
        #     assert torch.allclose(probs[n, k], (self._tpd[d, k] - a0 + self.alpha[k]) / (self._nd[d] + self.alpha.sum() - a0)
        #                           * dist.Normal(self._mean_rm[rm, k], self._std_rm[rm, k]).log_prob(x).exp())

        probs /= torch.sum(probs, dim=1, keepdim=True)
        return probs


    def _sampling_rh(self, rh, idx):
        """
        1. sampleing z
        2. update tpd, xpt, xt
        """

        before = self.z_rh[rh][idx].clone()

        # zをサンプリング
        probs = self._get_z_rh_sampling_probs(rh, idx)
        self.z_rh[rh][idx] = dist.Categorical(probs).sample()

        # tpdとか更新
        ones = torch.ones(len(idx), device=self.device, dtype=torch.int64)
        self._tpd.index_put_([idx, before], -ones, accumulate=True)
        self._tpd.index_put_([idx, self.z_rh[rh][idx]], ones, accumulate=True)
        self._xpt_rh[rh].index_put_([before, self.x_rh[rh][idx]], -ones, accumulate=True)
        self._xpt_rh[rh].index_put_([self.z_rh[rh][idx], self.x_rh[rh][idx]], ones, accumulate=True)
        self._xt_rh[rh] = torch.sum(self._xpt_rh[rh], 1)

        """ calculation checking """
        # for i in range(100):
        #     k = random.randrange(self.K)
        #     x = random.randrange(self.n_rh[rh])
        #     assert self._xpt_rh[rh][k, x] == torch.sum(torch.logical_and(self.z_rh[rh] == k, self.x_rh[rh] == x))


    def _get_z_rh_sampling_probs(self, rh, idx):
        subsample_size = len(idx)
        probs = torch.ones((subsample_size, self.K), device=self.device, dtype=torch.float64)  # [subsample_size, K]
        a = torch.zeros((subsample_size, self.K), device=self.device, dtype=torch.float64)     # [subsample_size, K]
        a[torch.arange(0, subsample_size, device=self.device, dtype=torch.int64), self.z_rh[rh][idx]] = 1

        probs *= self._xpt_rh[rh][:, self.x_rh[rh, idx]].T + self.rho_h_rh[rh][self.x_rh[rh][idx]][:, None] - a
        probs /= self._xt_rh[rh][None, :] + self.rho_h_rh[rh].sum() - a
        probs *= self._tpd[idx, :] + self.alpha[None, :] - a
        probs /= self._nd[idx][:, None] + self.alpha.sum() - a

        """ calculation checking """
        # for i in range(100):
        #     n = random.randrange(subsample_size)  # any
        #     k = random.randrange(self.K)  # any
        #     d = idx[n]
        #     z = self.z_rh[rh, d]
        #     x = self.x_rh[rh, d]
        #     a0 = (1 if z == k else 0)
        #     assert a[n, k] == a0
        #     assert(probs[n, k] == (self._xpt_rh[rh][k, x] + self.rho_h_rh[rh][x] - a0) / (self._xt_rh[rh][k] + self.rho_h_rh[rh].sum() - a0)
        #            * (self._tpd[d, k] - a0 + self.alpha[k]) / (self._nd[d] + self.alpha.sum() - a0))

        probs /= torch.sum(probs, dim=1, keepdim=True)
        return probs


    def _update_parameters(self):
        # TODO
        pass


    def theta(self, to_cpu=True):
        """
        毎回計算するから何度も呼び出さないこと
        """
        theta = (self._tpd + self.alpha[None, :]) / (self._nd[:, None] + self.alpha.sum())  # [D, K]
        if(to_cpu):
            return theta.cpu().detach().numpy().copy()
        else:
            return theta


    def phi(self, rt, to_cpu=True):
        """
        毎回計算するから何度も呼び出さないこと
        """
        phi = (self._wpt_rt[rt] + self.beta_rt[rt][None, :]) / (self._wt_rt[rt][:, None] + self.beta_rt[rt].sum())  # [K, V]
        if(to_cpu):
            return phi.cpu().detach().numpy().copy()
        else:
            return phi


    def mu(self, rm, to_cpu=True):
        mu = self._mean_rm[rm]
        if(to_cpu):
            return mu.cpu().detach().numpy().copy()
        else:
            return mu


    def sigma(self, rm, to_cpu=True):
        sigma = self._std_rm[rm]
        if(to_cpu):
            return sigma.cpu().detach().numpy().copy()
        else:
            return sigma


    def rho(self, rh, to_cpu=True):
        """
        毎回計算するから何度も呼び出さないこと
        """
        rho = (self._xpt_rh[rh] + self.rho_h_rh[rh][None, :]) / (self._xt_rh[rh][:, None] + self.rho_h_rh[rh].sum())  # [K, n_rh]
        if(to_cpu):
            return rho.cpu().detach().numpy().copy()
        else:
            return rho


    def log_perplexity(self, testset=None):
        if(testset is None):
            p = 0.
            theta = self.theta(to_cpu=False)
            for rt in range(self.Rt):
                dv = torch.mm(theta, self.phi(rt, to_cpu=False))
                p += dv[self.docids_rt[rt], self.wordids_rt[rt]].log().sum().item()
            idx = torch.arange(0, self.D, device=self.device, dtype=torch.int64)
            for rm in range(self.Rm):
                mu = self.mu(rm, to_cpu=False)
                sigma = self.sigma(rm, to_cpu=False)
                x = torch.zeros([self.D, self.K], device=self.device, dtype=torch.float64) + self.x_rm[rm][:, None]
                dk = dist.Normal(mu, sigma).log_prob(x)
                p += dk[idx, self.z_rm[rm]].sum().item()
                # p += (theta.log() + dk).sum().item()
            for rh in range(self.Rh):
                dx = torch.mm(theta, self.rho(rh, to_cpu=False))
                p += dx[idx, self.x_rh[rh]].log().sum().item()
            return p
        else:
            # TODO
            return None


    def summary(self, summary_args):
        pass
