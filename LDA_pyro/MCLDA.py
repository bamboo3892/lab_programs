import pickle
import numpy as np
import openpyxl
from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix

import torch
from torch.distributions import constraints
import pyro
import pyro.poutine as poutine
from pyro import distributions as dist


def model(data, args):
    """
    data:
        [tensor[D, V1], tensor[D, V2], ...]
    args:
        K,
        eps
        autoHyperParam
        coef_beta, coef_alpha
        device
    """
    R = len(data)
    D = data[0].shape[0]
    V = [data[r].shape[1] for r in range(R)]
    K = args.K

    if(args.autoHyperParam):
        alpha_hyper = pyro.param("alpha_hyper", torch.ones([K], device=args.device, dtype=torch.float64) * args.coef_alpha, constraint=constraints.positive)
        beta_hyper = [0] * R
        for r in pyro.plate("records1", R):
            beta_hyper[r] = pyro.param(f"beta_hyper_r{r}", torch.ones([V[r]], device=args.device, dtype=torch.float64) * args.coef_beta, constraint=constraints.positive)
    else:
        alpha_hyper = torch.ones([K], device=args.device, dtype=torch.float64) * args.coef_alpha
        beta_hyper = [0] * R
        for r in pyro.plate("records1", R):
            beta_hyper[r] = torch.ones([V[r]], device=args.device, dtype=torch.float64) * args.coef_beta

    for r in pyro.plate("records2", R):
        with pyro.plate(f"topics_r{r}", K):
            phi = pyro.sample(f"phi_r{r}", dist.Dirichlet(beta_hyper[r]))

        with pyro.plate(f"documents_r{r}", D) as d:
            theta = pyro.sample(f"theta_r{r}", dist.Dirichlet(alpha_hyper))

            w_d = data[r][d] + args.eps
            w_d = w_d / torch.sum(w_d, dim=1, keepdim=True)

            p = torch.mm(theta, phi)  # [D, V]
            N = torch.sum(data[r][d], dim=1)  # [D]
            p = p * N[:, None] + 1  # Dirichlet(p)は最頻値(極大値)がp(w)になる分布
            w = pyro.sample(f"w_r{r}", dist.Dirichlet(p), obs=w_d)

    return phi, theta, w


def guide(data, args):
    R = len(data)
    D = data[0].shape[0]
    V = [data[r].shape[1] for r in range(R)]
    K = args.K

    alpha_model = pyro.param("alpha_model", torch.ones([D, K], device=args.device, dtype=torch.float64), constraint=constraints.positive)
    beta_model = [0] * R
    for r in pyro.plate("records1", R):
        beta_model[r] = pyro.param(f"beta_model_r{r}", torch.ones([K, V[r]], device=args.device, dtype=torch.float64), constraint=constraints.positive)

    for r in pyro.plate("records2", R):
        with pyro.plate(f"topics_r{r}", K) as k:
            phi = pyro.sample(f"phi_r{r}", dist.Dirichlet(beta_model[r][k]))
        with pyro.plate(f"documents_r{r}", D) as d:
            theta = pyro.sample(f"theta_r{r}", dist.Dirichlet(alpha_model[d]))


def summary(data, args, words, reviews, pathResultFolder=None, counts=None):
    R = len(data)
    D = data[0].shape[0]
    V = [data[r].shape[1] for r in range(R)]
    K = args.K

    alpha_model = pyro.param("alpha_model")
    alpha_model_ = alpha_model.cpu().detach().numpy()
    theta = dist.Dirichlet(alpha_model).independent(1).mean
    theta_ = theta.cpu().detach().numpy()

    beta_model = [0] * R
    beta_model_ = [0] * R
    phi = [0] * R
    phi_ = [0] * R
    for r in range(R):
        beta_model[r] = pyro.param(f"beta_model_r{r}")
        beta_model_[r] = beta_model[r].cpu().detach().numpy()
        phi[r] = dist.Dirichlet(beta_model[r]).independent(1).mean
        phi_[r] = phi[r].cpu().detach().numpy()

    if(args.autoHyperParam):
        alpha_hyper = pyro.param("alpha_hyper")
        alpha_hyper_ = alpha_hyper.cpu().detach().numpy()
        beta_hyper = [0] * R
        beta_hyper_ = [0] * R
        for r in range(R):
            beta_hyper[r] = pyro.param(f"beta_hyper_r{r}")
            beta_hyper_[r] = beta_hyper[r].cpu().detach().numpy()

    if(pathResultFolder is not None):
        print("saving models")

        # model_variables.pickle
        variables = {}
        args_dict = {k: args.__dict__[k] for k in args.__dict__ if not k.startswith("__")}

        variables["args"] = args_dict
        variables["words"] = words
        variables["reviews"] = reviews
        variables["beta_model"] = beta_model_
        variables["alpha_model"] = alpha_model_
        variables["phi"] = phi_
        variables["theta"] = theta_
        if(args.autoHyperParam):
            variables["beta_hyper"] = beta_hyper_
            variables["alpha_hyper"] = alpha_hyper_
        with open(pathResultFolder.joinpath("model_variables.pickle"), 'wb') as f:
            pickle.dump(variables, f)

        # result.xlsx
        wb = openpyxl.Workbook()
        tmp_ws = wb[wb.get_sheet_names()[0]]

        args_dict_str = args_dict.copy()
        for k in args_dict_str:
            if(not isinstance(args_dict_str[k], (int, float, complex, bool))):
                args_dict_str[k] = str(args_dict_str[k])
        ws = wb.create_sheet("args")
        writeVector(ws, list(args_dict_str.values()), axis="row", names=list(args_dict_str.keys()))

        if(args.autoHyperParam):
            ws = wb.create_sheet("alpha_hyper")
            writeVector(ws, alpha_hyper_, axis="row", names=[f"topic{d+1}" for d in range(K)],
                        addDataBar=True)

            ws = wb.create_sheet("beta_hyper")
            for r in range(R):
                writeVector(ws, beta_hyper_[r], column=r * 3 + 1,
                            axis="row", names=words[r],
                            addDataBar=True)

        ws = wb.create_sheet("theta")
        writeMatrix(ws, theta_, 1, 1,
                    row_names=[f"doc{d+1}" for d in range(D)],
                    column_names=[f"topic{k+1}" for k in range(K)],
                    addDataBar=True)

        for r in range(R):
            ws = wb.create_sheet(f"phi_r{r}")
            writeMatrix(ws, phi_[r].T, 1, 1,
                        row_names=words[r],
                        column_names=[f"topic{k+1}" for k in range(K)],
                        addDataBar=True)

            ws = wb.create_sheet(f"phi_r{r}_sorted")
            writeSortedMatrix(ws, phi_[r].T, axis=0, row=1, column=1,
                              row_names=words[r], column_names=[f"topic{k+1}" for k in range(K)],
                              maxwrite=100, order="higher")
            writeSortedMatrix(ws, phi_[r].T, axis=0, row=1, column=K + 3,
                              row_names=None, column_names=[f"topic{k+1}" for k in range(K)],
                              maxwrite=100, order="higher",
                              addDataBar=True)

        wb.remove_sheet(tmp_ws)
        wb.save(pathResultFolder.joinpath("result.xlsx"))
