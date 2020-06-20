from abc import ABCMeta, abstractmethod
import torch


class MCMCModel(metaclass=ABCMeta):

    def step(self, subsample_size, parameter_update=False):
        """
        subsample_sizeはgpuのコア数の整数倍が良さそう
        1. update z, wpt, ...
        2. update parameters if neseccary
        """

        rand_perm = torch.randperm(self.totalN, device=self.device)
        for n in range(self.totalN // subsample_size + 1):
            end = self.totalN if n == (self.totalN // subsample_size) else subsample_size * (n + 1)
            idx = rand_perm[subsample_size * n: end]
            self._sampling(idx)

        if(parameter_update):
            self._update_parameters()

        return self.log_perplexity()


    @abstractmethod
    def log_perplexity(self, testset=None):
        pass


    @abstractmethod
    def _sampling(self, idx):
        """
        idx: batch sample indexes
        """
        pass


    @abstractmethod
    def _update_parameters(self):
        pass


    @abstractmethod
    def summary(self):
        pass
