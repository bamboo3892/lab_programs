import numpy as np
from scipy.stats import norm, entropy



def calc_JSdivergence_for_gaussian(mu1, sigma1, mu2, sigma2):
    lower = min([mu1 - 10 * sigma1, mu2 - 10 * sigma2])
    upper = max([mu1 + 10 * sigma1, mu2 + 10 * sigma2])
    x = np.linspace(lower, upper, 10000)
    p1 = norm.pdf(x, loc=mu1, scale=sigma1)
    p2 = norm.pdf(x, loc=mu2, scale=sigma2)

    return calc_JSdivergence_for_multinomial(p1, p2)


def calc_KLdivergence_for_gaussian(mu1, sigma1, mu2, sigma2):
    # rtn = np.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5

    lower = min([mu1 - 10 * sigma1, mu2 - 10 * sigma2])
    upper = max([mu1 + 10 * sigma1, mu2 + 10 * sigma2])
    x = np.linspace(lower, upper, 10000)
    p1 = norm.pdf(x, loc=mu1, scale=sigma1)
    p2 = norm.pdf(x, loc=mu2, scale=sigma2)

    return calc_KLdivergence_for_multinomial(p1, p2)


def calc_JSdivergence_for_multinomial(p1, p2):
    m = (p1 + p2) / 2
    kl1 = calc_KLdivergence_for_multinomial(p1, m)
    kl2 = calc_KLdivergence_for_multinomial(p2, m)
    return (kl1 + kl2) / 2


def calc_KLdivergence_for_multinomial(p1, p2):
    return entropy(p1, p2)
