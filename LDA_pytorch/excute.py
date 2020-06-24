import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from analysis.analyzeHealthCheck import habitLevels2
import LDA_pytorch.LDA as LDA
import LDA_pytorch.MCLDA as MCLDA


seed = 1
np.random.seed(seed)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def excuteLDA(pathDocs, pathResult, *,
              args=None, summary_args=None):

    args = args if args is not None else type("args", (object,), {})
    summary_args = summary_args if summary_args is not None else type("args", (object,), {})

    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        docs = json.load(f)

    model_class = LDA.LDA
    args.modelType = "LDA"
    args.num_steps = 200
    args.step_subsample = 100000
    args.K = 7
    args.auto_beta = False
    args.auto_alpha = False
    args.coef_beta = 1
    args.coef_alpha = 1

    summary_args.morphome_key = "morphomes"
    summary_args.full_docs = docs

    data = [doc[summary_args.morphome_key] for doc in docs]
    print(f"coef_beta:   {args.coef_beta} (auto: {args.auto_beta})")
    print(f"coef_alpha:  {args.coef_alpha} (auto: {args.auto_alpha})")

    _excute(model_class, args, data, pathResult, summary_args)


def excuteMCLDA(pathDocs, pathTensors, pathResult, *,
                args=None, summary_args=None):

    args = args if args is not None else type("args", (object,), {})
    summary_args = summary_args if summary_args is not None else type("args", (object,), {})

    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        documents = json.load(f)
    with open(str(pathTensors), 'rb') as f:
        tensors = pickle.load(f)

    morphomesKeys = tensors["tensor_keys"]
    measurementKeysHC = tensors["measurement_keys"]
    habitKeysHC = tensors["habit_keys"]
    medicineKeysHC = tensors["medicine_keys"]
    tensors["habit_levels"] = habitLevels2

    docs = [[] for _ in range(len(morphomesKeys))]
    measurements = [[] for _ in range(len(measurementKeysHC))]
    habits = [[] for _ in range(len(habitKeysHC))]

    for doc in documents:
        pid = int(doc["p_seq"])
        if(pid in tensors["ids"]):
            idx = tensors["ids"].index(pid)
            for rt, key in enumerate(morphomesKeys):
                docs[rt].append(doc[key + "_morphomes"])
            for rm, key in enumerate(measurementKeysHC):
                measurements[rm].append(tensors[key][idx])
            for rh, key in enumerate(habitKeysHC):
                habits[rh].append(tensors[key][idx])
    data = [docs, np.array(measurements), np.array(habits)]
    # data = [docs, np.array([]), np.array([])]
    # data = [[], np.array(measurements), np.array([])]
    # data = [[], np.array([]), np.array(habits)]

    model_class = MCLDA.MCLDA
    args.modelType = "MCLDA"
    args.num_steps = 200
    args.step_subsample = 10
    args.K = 10
    args.D = len(docs[0]) if len(docs) != 0 else (len(measurements[0]) if len(measurements) != 0 else (len(habits[0]) if len(habits) != 0 else 0))
    args.n_rh = [len(habitLevels2[rh]) for rh in range(len(habitKeysHC))]
    # args.n_rh = []
    args.auto_beta = False
    args.auto_alpha = False
    args.coef_beta = 1
    args.coef_alpha = 1

    summary_args.full_docs = docs
    summary_args.full_tensors = tensors
    print(f"D: {args.D}, K: {args.K}")
    print(f"coef_beta:   {args.coef_beta} (auto: {args.auto_beta})")
    print(f"coef_alpha:  {args.coef_alpha} (auto: {args.auto_alpha})")

    _excute(model_class, args, data, pathResult, summary_args)


def _excute(modelClass, args, data, pathResult, summary_args):

    print(f"Model: {args.modelType}  (to {pathResult})")

    pathResult.mkdir(exist_ok=True, parents=True)

    args.device = DEVICE
    summary_args.summary_path = pathResult

    model = modelClass(args, data)
    losses = []
    for n in range(args.num_steps):
        probability = model.step(args.step_subsample)
        losses.append(probability)
        print("i:{:<5d} loss:{:<f}".format(n + 1, probability))
        if(n == 46):
            pass

    plt.plot(losses)
    plt.savefig(pathResult.joinpath("probability.png"))
    plt.clf()
    model.summary(summary_args)
