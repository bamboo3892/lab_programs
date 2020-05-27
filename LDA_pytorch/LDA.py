import numpy as np
import torch


class LDA:

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
        super().__init__()

        # fields
        self.K = args.K
        self.D = len(data)
        self.V = 0
        self.totalN = 0
        self.word_count = {}  # [V]
        self.word_dict = []  # [V]
        self.wordIDs = []  # [totalN]
        self.docIDs = []  # [totalN]
        self.args = args
        self.data = data

        self._z = None  # [totalN]
        self._wpt = None  # [K, V]
        self._tpd = torch.zeros((D, K), device=self.args.device, dtype=torch.int64)  # [D, K]
        self._wt = torch.zeros((K), device=self.args.device, dtype=torch.int64)  # [K]

        # init wordcount...
        for d, doc in enumerate(data):
            for sentence in doc:
                for word in sentence:
                    if(word not in self.word_count):
                        self.word_count[word] = 0
                    self.word_count[word] += 1
        self.word_dict = [a[0] for a in sorted(self.word_count.items(), key=lambda x: -x[1])]
        self.V = len(self.word_dict)

        # init wordIDs...
        for d, doc in enumerate(data):
            for sentence in doc:
                for word in sentence:
                    self.wordIDs.append(self.word_dict.index(word))
                    self.docIDs.append(d)
        self.totalN = len(self.wordIDs)
        self._z = torch.randint(0, K, (self.totalN), device=self.args.device, dtype=torch.int64)  # [totalN]
        self._wpt = torch.zeros((K, V), device=self.args.device, dtype=torch.int64)  # [K, V]

        # init wpt...
        self._wpt =

        print("")


    def _step(self):
        # (totalN // subsamplesize + 1)回繰り返し、subsamplesizeはgpuのコア数の整数倍が良さそう
        # 一部zサンプル
        # wptとか更新tpd
        pass


    def summary(self, summary_args):
        """
        summary_args:
            path
        """
        pass
