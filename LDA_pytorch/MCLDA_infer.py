import random
import numpy as np
import torch
import pyro.distributions as dist


class MCLDA_infer:
    """
    未知のデータセットの欠損値を予測するためのクラス
    testsetに対するaccuracyを評価するためにも使用する
    """

    def __init__(self, mclda_obj, data, masked_records):
        """
        mclda_obj:      予測に使うMCLDAオブジェクト
        data:           未知のデータセット，リストの形式や大きさはmclda_objの作成時に渡したdataと全く同じ
                        masked_recordsで指定された記録は欠損値扱いとなり，その記録の数値は完全に無視される
        masked_records: {"rt": [masked_record_id, ...], "rm": [...], "rh": [...]}
        """

        self.M = mclda_obj
        self.device = self.M.device
        self.mask = _mask_validate(masked_records)

        docs = data[0]
        measurements = torch.tensor(data[1], device=self.device, dtype=torch.float64)  # [Rm, D]
        habits = torch.tensor(data[2], device=self.device, dtype=torch.int64)          # [Rh, D]

        # fields
        self.D = D = len(docs[0]) if len(docs) != 0 else (len(measurements[0]) if len(measurements) != 0 else (len(habits[0]) if len(habits) != 0 else 0))
        self.totalN_rt = [0 for _ in range(self.M.Rt)]                                       # [Rt]
        self.wordids_rt = [[] for _ in range(self.M.Rt)]                                     # [Rt][totalN_rt]
        self.docids_rt = [[] for _ in range(self.M.Rt)]                                      # [Rt][totalN_rt]
        self.x_rm = measurements                                                           # [Rm, D]
        self.x_rh = habits                                                                 # [Rh, D]
        self.z_rt = [None for _ in range(self.M.Rt)]                                         # [Rt][totalN_rt]
        self.z_rm = torch.full([self.M.Rm, D], -1, device=self.device, dtype=torch.int64)  # [Rm, D]
        self.z_rh = torch.full([self.M.Rh, D], -1, device=self.device, dtype=torch.int64)  # [Rh, D]

        self.alpha = torch.full([self.M.K], self.M.args.coef_alpha, device=self.device, dtype=torch.float64)

        self._tpd = torch.zeros((D, self.M.K), device=self.device, dtype=torch.int64)  # [D, K]
        self._nd = torch.zeros(D, device=self.device, dtype=torch.int64)               # [D]

        # init wordids_rt...
        for rt, documents in enumerate(docs):
            for d, doc in enumerate(documents):
                for sentence in doc:
                    for word in sentence:
                        if(word in self.M.word_dict_rt[rt]):
                            self.wordids_rt[rt].append(self.M.word_dict_rt[rt].index(word))
                            self.docids_rt[rt].append(d)
            self.totalN_rt[rt] = len(self.wordids_rt[rt])
            self.wordids_rt[rt] = torch.tensor(self.wordids_rt[rt], device=self.device, dtype=torch.int64)
            self.docids_rt[rt] = torch.tensor(self.docids_rt[rt], device=self.device, dtype=torch.int64)

        self.change_mask(self.mask)

        pass


    def change_mask(self, mask):
        """
        change mask and reset latent variables
        """

        self.mask = _mask_validate(mask)
        self.z_rt = [None for _ in range(self.M.Rt)]                                         # [Rt][totalN_rt]
        self.z_rm = torch.full([self.M.Rm, self.D], -1, device=self.device, dtype=torch.int64)  # [Rm, D]
        self.z_rh = torch.full([self.M.Rh, self.D], -1, device=self.device, dtype=torch.int64)  # [Rh, D]

        for rt in range(self.M.Rt):
            if(rt not in self.mask["rt"]):
                self.z_rt[rt] = torch.randint(0, self.M.K, (self.totalN_rt[rt],), device=self.device, dtype=torch.int64)
                ones = torch.ones(self.totalN_rt[rt], device=self.device, dtype=torch.int64)
                self._tpd.index_put_([self.docids_rt[rt], self.z_rt[rt]], ones, accumulate=True)
        ones = torch.ones(self.D, device=self.device, dtype=torch.int64)
        idx = torch.arange(0, self.D, device=self.device, dtype=torch.int64)
        for rm in range(self.M.Rm):
            if(rm not in self.mask["rm"]):
                self.z_rm[rm] = torch.randint(0, self.M.K, [self.D], device=self.device, dtype=torch.int64)
                self._tpd.index_put_([idx, self.z_rm[rm]], ones, accumulate=True)
        for rh in range(self.M.Rh):
            self.z_rh[rh] = torch.randint(0, self.M.K, [self.D], device=self.device, dtype=torch.int64)
            self._tpd.index_put_([idx, self.z_rh[rh]], ones, accumulate=True)
        self._nd = torch.sum(self._tpd, 1)


    def step(self, num_subsample_partitions, parameter_update=False):
        """
        1. update z, wpt, ...
        2. update parameters if neseccary
        """

        # generate sabsampling batches
        rand_perm_rt = [None for _ in range(self.M.Rt)]
        rand_perm_rm = [None for _ in range(self.M.Rm)]
        rand_perm_rh = [None for _ in range(self.M.Rh)]
        for rt in range(self.M.Rt):
            if rt not in self.mask["rt"]:
                rand_perm_rt[rt] = torch.randperm(self.totalN_rt[rt], device=self.device)
        for rm in range(self.M.Rm):
            if rm not in self.mask["rm"]:
                rand_perm_rm[rm] = torch.randperm(self.D, device=self.device)
        for rh in range(self.M.Rh):
            if rh not in self.mask["rh"]:
                rand_perm_rh[rh] = torch.randperm(self.D, device=self.device)

        # sampling
        for n in range(num_subsample_partitions):
            # texts
            for rt in range(self.M.Rt):
                if rt not in self.mask["rt"]:
                    s = (self.totalN_rt[rt] // num_subsample_partitions + 1) * n
                    e = (self.totalN_rt[rt] // num_subsample_partitions + 1) * (n + 1) - 1
                    e = e if e < self.totalN_rt[rt] else self.totalN_rt[rt] - 1
                    self._sampling_rt(rt, rand_perm_rt[rt][s:e])
            # measurements
            for rm in range(self.M.Rm):
                if rm not in self.mask["rm"]:
                    s = (self.D // num_subsample_partitions + 1) * n
                    e = (self.D // num_subsample_partitions + 1) * (n + 1) - 1
                    e = e if e < self.D else self.D - 1
                    self._sampling_rm(rm, rand_perm_rm[rm][s:e])
            # habits
            for rh in range(self.M.Rh):
                if rh not in self.mask["rh"]:
                    s = (self.D // num_subsample_partitions + 1) * n
                    e = (self.D // num_subsample_partitions + 1) * (n + 1) - 1
                    e = e if e < self.D else self.D - 1
                    self._sampling_rh(rh, rand_perm_rh[rh][s:e])

        # update parameters
        if(parameter_update):
            pass

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

        return self.log_probability()


    def _sampling_rt(self, rt, idx):
        """
        1. sampleing z
        2. update tpd
        """

        before = self.z_rt[rt][idx].clone()

        # zをサンプリング
        probs = self._get_z_rt_sampling_probs(rt, idx)
        self.z_rt[rt][idx] = dist.Categorical(probs).sample()

        # tpdとか更新
        ones = torch.ones(len(idx), device=self.device, dtype=torch.int64)
        self._tpd.index_put_([self.docids_rt[rt][idx], before], -ones, accumulate=True)
        self._tpd.index_put_([self.docids_rt[rt][idx], self.z_rt[rt][idx]], ones, accumulate=True)


    def _get_z_rt_sampling_probs(self, rt, idx):
        subsample_size = len(idx)
        probs = torch.ones((subsample_size, self.M.K), device=self.device, dtype=torch.float64)  # [subsample_size, K]
        a = torch.zeros((subsample_size, self.M.K), device=self.device, dtype=torch.float64)     # [subsample_size, K]
        a[torch.arange(0, subsample_size, device=self.device, dtype=torch.int64), self.z_rt[rt][idx]] = 1

        probs *= self.M._wpt_rt[rt][:, self.wordids_rt[rt][idx]].T + self.M.beta_rt[rt][self.wordids_rt[rt][idx]][:, None]
        probs /= self.M._wt_rt[rt][None, :] + self.M.beta_rt[rt].sum()
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
        2. update tpd
        """

        before = self.z_rm[rm][idx].clone()

        # zをサンプリング
        probs = self._get_z_rm_sampling_probs(rm, idx)
        self.z_rm[rm][idx] = dist.Categorical(probs).sample()

        # tpdとか更新
        ones = torch.ones(len(idx), device=self.device, dtype=torch.int64)
        self._tpd.index_put_([idx, self.z_rm[rm][idx]], ones, accumulate=True)
        self._tpd.index_put_([idx, before], -ones, accumulate=True)


    def _get_z_rm_sampling_probs(self, rm, idx):
        subsample_size = len(idx)
        probs = torch.ones((subsample_size, self.M.K), device=self.device, dtype=torch.float64)                    # [subsample_size, K]
        a = torch.zeros((subsample_size, self.M.K), device=self.device, dtype=torch.float64)                       # [subsample_size, K]
        a[torch.arange(0, subsample_size, device=self.device, dtype=torch.int64), self.z_rm[rm][idx]] = 1
        x = torch.zeros([subsample_size, self.M.K], device=self.device, dtype=torch.float64) + self.x_rm[rm, idx][:, None]  # [subsample_size, K]

        probs *= self._tpd[idx, :] + self.alpha[None, :] - a
        probs /= self._nd[idx][:, None] + self.alpha.sum() - a

        # probs *= dist.Normal(self.M._mean_rm[rm], self.M._std_rm[rm]).log_prob(x).exp()
        std2 = self.M._std_rm[rm] ** 2
        mean = self.M._xt_rm[rm] * self.M.sigma2_h_rm[rm] * self.M._mean_rm[rm] + std2 * self.M.mu_h_rm[rm]  # [K]
        mean /= self.M._xt_rm[rm] * self.M.sigma2_h_rm[rm] + std2
        mean[torch.isnan(mean)] = self.M.mu_h_rm[rm]
        var = (self.M.nu_h_rm * self.M.sigma2_h_rm[rm] + self.M._xt_rm[rm] * std2) / (self.M.nu_h_rm + self.M._xt_rm[rm] - 2)  # [K]
        probs *= dist.Normal(mean, var).log_prob(x).exp()

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
        2. update tpd
        """

        before = self.z_rh[rh][idx].clone()

        # zをサンプリング
        probs = self._get_z_rh_sampling_probs(rh, idx)
        self.z_rh[rh][idx] = dist.Categorical(probs).sample()

        # tpdとか更新
        ones = torch.ones(len(idx), device=self.device, dtype=torch.int64)
        self._tpd.index_put_([idx, before], -ones, accumulate=True)
        self._tpd.index_put_([idx, self.z_rh[rh][idx]], ones, accumulate=True)


    def _get_z_rh_sampling_probs(self, rh, idx):
        subsample_size = len(idx)
        probs = torch.ones((subsample_size, self.M.K), device=self.device, dtype=torch.float64)  # [subsample_size, K]
        a = torch.zeros((subsample_size, self.M.K), device=self.device, dtype=torch.float64)     # [subsample_size, K]
        a[torch.arange(0, subsample_size, device=self.device, dtype=torch.int64), self.z_rh[rh][idx]] = 1

        probs *= self.M._xpt_rh[rh][:, self.x_rh[rh, idx]].T + self.M.rho_h_rh[rh][self.x_rh[rh][idx]][:, None]
        probs /= self.M._xt_rh[rh][None, :] + self.M.rho_h_rh[rh].sum()
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


    def log_probability(self):
        return self._log_prob(self.mask)


    def _log_prob(self, mask={}):
        mask = _mask_validate(mask)

        p = 0.
        theta = self.theta(to_cpu=False)
        idx = torch.arange(0, self.D, device=self.device, dtype=torch.int64)
        for rt in range(self.M.Rt):
            if rt not in mask["rt"]:
                p += self._log_prob_rt(rt, theta)
        for rm in range(self.M.Rm):
            if rm not in mask["rm"]:
                p += self._log_prob_rm(rm, theta, idx)
        for rh in range(self.M.Rh):
            if rh not in mask["rh"]:
                p += self._log_prob_rh(rh, theta, idx)
        return p


    def _log_prob_rt(self, rt, theta, mean=False):
        dv = torch.mm(theta, self.M.phi(rt, to_cpu=False))
        if(not mean):
            return dv[self.docids_rt[rt], self.wordids_rt[rt]].log().sum().item()
        else:
            return dv[self.docids_rt[rt], self.wordids_rt[rt]].mean().item()


    def _log_prob_rm(self, rm, theta, idx, mean=False):
        mu = self.M.mu(rm, to_cpu=False)
        sigma = self.M.sigma(rm, to_cpu=False)
        x = torch.zeros([self.D, self.M.K], device=self.device, dtype=torch.float64) + self.x_rm[rm][:, None]
        dk = dist.Normal(mu, sigma).log_prob(x)
        if(not mean):
            # return (theta.log() + dk).sum().item()
            return dk[idx, self.z_rm[rm]].sum().item()
        else:
            return dk[idx, self.z_rm[rm]].exp().mean().item()


    def _log_prob_rh(self, rh, theta, idx, mean=False):
        dx = torch.mm(theta, self.M.rho(rh, to_cpu=False))
        if(not mean):
            return dx[idx, self.x_rh[rh]].log().sum().item()
        else:
            return dx[idx, self.x_rh[rh]].mean().item()


    def theta(self, to_cpu=True):
        theta = (self._tpd + self.alpha[None, :]) / (self._nd[:, None] + self.alpha.sum())  # [D, K]
        if(to_cpu):
            return theta.cpu().detach().numpy().copy()
        else:
            return theta


    def calc_mean_accuracy(self, test_records):
        """
        平均正解率を計算
        テストするデータはこのクラスを作成したときに渡したものをそのまま使う
        Parameters
        ----------
        test_records: {"rt": [masked_record_id, ...], "rm": [...], "rh": [...]}
                      基本的にはtest_records in masked_recordsとして使う
        Return
        ------
        mean_accuracy: {"rt": [mean_accuracy, ...], "rm": [...], "rh": [...]}
                       the same shape of test_records argument
        """
        test_records = _mask_validate(test_records)
        accuracy = _mask_validate({})

        theta = self.theta(to_cpu=False)
        idx = torch.arange(0, self.D, device=self.device, dtype=torch.int64)
        for rt in test_records["rt"]:
            accuracy["rt"].append(self._log_prob_rt(rt, theta, mean=True))
        for rm in test_records["rm"]:
            accuracy["rm"].append(self._log_prob_rm(rm, theta, idx, mean=True))
        for rh in test_records["rh"]:
            accuracy["rh"].append(self._log_prob_rh(rh, theta, idx, mean=True))

        return accuracy


def _mask_validate(mask):
    if("rt" not in mask):
        mask["rt"] = []
    if("rm" not in mask):
        mask["rm"] = []
    if("rh" not in mask):
        mask["rh"] = []
    return mask
