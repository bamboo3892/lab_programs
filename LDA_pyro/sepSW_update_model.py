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

    beta_model = pyro.param("beta_model", torch.ones([args.V]), constraint=constraints.positive)
    gamma_model = pyro.param("gamma_model", torch.ones([args.V]), constraint=constraints.positive)
    alpha_model = pyro.param("alpha_model", torch.ones([args.K]), constraint=constraints.positive)

    with pyro.plate("topics", args.K):
        phi = pyro.sample("phi", dist.Dirichlet(beta_model))
    with pyro.plate("attributes", args.S):
        sigma = pyro.sample("sigma", dist.Dirichlet(gamma_model))
    # phi = pyro.sample("phi", dist.Dirichlet(torch.ones([args.K, args.V]) * args.coef_phi).independent(1))
    # sigma = pyro.sample("sigma", dist.Dirichlet(torch.ones([args.S, args.V]) * args.coef_sigma).independent(1))

    with pyro.plate("documents", args.D) as d:
        theta = pyro.sample("theta", dist.Dirichlet(alpha_model))
        # theta = pyro.sample("theta", dist.Dirichlet(torch.ones([args.K]) * args.coef_theta))

        w_d = data[0][d] + args.eps
        w_d = w_d / torch.sum(w_d, dim=1, keepdim=True)

        p1 = torch.mm(theta, phi)  # [D, V]
        p2 = sigma[data[1]]  # [D, V]
        p = p1 * p2
        p = p / torch.sum(p, dim=1, keepdim=True)  # w ~ Categorical(p)

        N = torch.sum(data[0][d], dim=1)  # [D]
        p = p * N[:, None] + 1  # Dirichlet(p)は最頻値(極大値)がp(w)になる分布

        w = pyro.sample("w", dist.Dirichlet(p), obs=w_d)

    return phi, theta, w


def guide(data, args):
    alpha = pyro.param("alpha", torch.ones([args.D, args.K]), constraint=constraints.positive)
    beta = pyro.param("beta", torch.ones([args.K, args.V]), constraint=constraints.positive)
    gamma = pyro.param("gamma", torch.ones([args.S, args.V]), constraint=constraints.positive)

    phi = pyro.sample("phi", dist.Dirichlet(beta).independent(1))
    sigma = pyro.sample("sigma", dist.Dirichlet(gamma).independent(1))

    with pyro.plate("documents", args.D) as d:
        theta = pyro.sample("theta", dist.Dirichlet(alpha[d]))


def summary(data, args, words, reviews):
    beta = pyro.param("beta")
    alpha = pyro.param("alpha")
    phi = dist.Dirichlet(beta).independent(1).mean
    theta = dist.Dirichlet(alpha).independent(1).mean

    phi_ = phi.detach().numpy()
    theta_ = theta.detach().numpy()
    for k in range(args.K):
        msg = "Topic {:2d}: ".format(k)
        ind = np.argsort(phi_[k, :])[::-1]
        for i in range(10):
            msg += "{} ".format(words[ind[i]])
        print(msg)
    for k in range(args.K):
        msg = "Topic {:2d}: ".format(k)
        ind = np.argsort(phi_[k, :])[::-1]
        for i in range(10):
            msg += "{:6f} ".format(phi_[k, ind[i]])
        print(msg)
    for d in range(3):
        msg = "Documents {:2d}: ".format(d)
        for k in range(args.K):
            msg += "{:6f} ".format(theta_[d, k])
        print(msg)

    gamma = pyro.param("gamma")
    sigma = dist.Dirichlet(gamma).independent(1).mean
    sigma_ = sigma.detach().numpy()
    for s in range(args.S):
        msg = "Topic {:2d}: ".format(s)
        ind = np.argsort(sigma_[s, :])[::-1]
        for i in range(10):
            msg += "{} ".format(words[ind[i]])
        print(msg)
    for s in range(args.S):
        msg = "Topic {:2d}: ".format(s)
        ind = np.argsort(sigma_[s, :])[::-1]
        for i in range(10):
            msg += "{:6f} ".format(sigma_[s, ind[i]])
        print(msg)

    p = phi_ * sigma_[3, :][None, :]  # [K, S]
    phi2_ = p / np.sum(p, axis=1, keepdims=True)
    for k in range(args.K):
        msg = "Topic {:2d}: ".format(k)
        ind = np.argsort(phi2_[k, :])[::-1]
        for i in range(10):
            msg += "{} ".format(words[ind[i]])
        print(msg)
    for k in range(args.K):
        msg = "Topic {:2d}: ".format(k)
        ind = np.argsort(phi2_[k, :])[::-1]
        for i in range(10):
            msg += "{:6f} ".format(phi2_[k, ind[i]])
        print(msg)
