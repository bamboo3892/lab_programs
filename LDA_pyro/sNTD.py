import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.poutine as poutine
from pyro import distributions as dist


def model(data, args):
    """
    data: [tensor[D, V], attributions[S]]
    """

    phi = pyro.sample("phi", dist.Dirichlet(torch.ones([args.K1, args.V]) * args.coef_phi).independent(1))
    sigma = pyro.sample("sigma", dist.Dirichlet(torch.ones([args.S, args.K2, args.K1]) * args.coef_sigma).independent(2))

    with pyro.plate("documents", args.D) as d:

        theta = pyro.sample("theta", dist.Dirichlet(torch.ones([args.K2]) * args.coef_theta))

        w_d = data[0][d] + args.eps
        w_d = w_d / torch.sum(w_d, dim=1, keepdim=True)

        p1 = torch.matmul(sigma[data[1], :, :], phi)  # [D, K2, K1] * [K1, V] = [D, K2, V]
        p = torch.bmm(theta[:, None, :], p1)  # [D, 1, K2] x [D, K2, V] = [D, 1, V]
        p = p[:, 0, :]  # [D, V]
        p = p / torch.sum(p, dim=1, keepdim=True)  # w ~ Categorical(p)

        N = torch.sum(data[0][d], dim=1)  # [D]
        p = p * N[:, None] + 1  # Dirichlet(p)は最頻値(極大値)がp(w)になる分布

        w = pyro.sample("w", dist.Dirichlet(p), obs=w_d)

    return phi, theta, w


def guide(data, args):
    beta = pyro.param("beta", torch.ones([args.K1, args.V]), constraint=constraints.positive)
    gamma = pyro.param("gamma", torch.ones([args.S, args.K2, args.K1]), constraint=constraints.positive)
    alpha = pyro.param("alpha", torch.ones([args.D, args.K2]), constraint=constraints.positive)

    phi = pyro.sample("phi", dist.Dirichlet(beta).independent(1))
    sigma = pyro.sample("sigma", dist.Dirichlet(gamma).independent(2))

    with pyro.plate("documents", args.D) as d:
        theta = pyro.sample("theta", dist.Dirichlet(alpha[d]))


def summary(data, args, words, reviews):
    beta = pyro.param("beta")
    alpha = pyro.param("alpha")
    phi = dist.Dirichlet(beta).independent(1).mean.detach()
    theta = dist.Dirichlet(alpha).independent(1).mean.detach()

    phi_ = phi.numpy()
    theta_ = theta.numpy()
    for k in range(args.K1):
        msg = "Topic {:2d}: ".format(k)
        ind = np.argsort(phi_[k, :])[::-1]
        for i in range(20):
            msg += "{} ".format(words[ind[i]])
        print(msg)
    for k in range(args.K1):
        msg = "Topic {:2d}: ".format(k)
        ind = np.argsort(phi_[k, :])[::-1]
        for i in range(20):
            msg += "{:6f} ".format(phi_[k, ind[i]])
        print(msg)
    for d in range(3):
        msg = "Documents {:2d}: ".format(d)
        for k in range(args.K2):
            msg += "{:6f} ".format(theta_[d, k])
        print(msg)

    gamma = pyro.param("gamma")
    sigma = dist.Dirichlet(gamma).independent(2).mean.detach()
    sigma_ = sigma.numpy()

    p = torch.matmul(sigma, phi)  # [S, K2, K1] * [K1, V] = [S, K2, V]
    phi2 = p / torch.sum(p, dim=2, keepdim=True)  # w ~ Categorical(p)
    phi2_ = phi2.numpy()

    print("sigma #################")
    for s in range(args.S):
        for k2 in range(args.K2):
            msg = "S{} Topic {:2d}: ".format(s, k2)
            for k1 in range(args.K1):
                msg += "{:6f} ".format(sigma_[s, k2, k1])
            print(msg)
    print("sigma*phi #################")
    for s in range(args.S):
        for k in range(args.K2):
            msg = "Topic {:2d}: ".format(k)
            ind = np.argsort(phi2_[s, k, :])[:: -1]
            for i in range(20):
                msg += "{} ".format(words[ind[i]])
            print(msg)
        for k in range(args.K2):
            msg = "Topic {:2d}: ".format(k)
            ind = np.argsort(phi2_[s, k, :])[:: -1]
            for i in range(20):
                msg += "{:6f} ".format(phi2_[s, k, ind[i]])
            print(msg)
