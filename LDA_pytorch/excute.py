import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import openpyxl

from analysis.analyzeHealthCheck import habitLevels2, habitWorstLevels2
import LDA_pytorch.LDA as LDA
import LDA_pytorch.MCLDA as MCLDA
from utils.general_util import Args
from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def excuteLDA(pathDocs, pathResult, *,
              args=None, summary_args=None):

    args = args if args is not None else Args()
    summary_args = summary_args if summary_args is not None else Args()

    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        docs = json.load(f)

    model_class = LDA.LDA
    args.modelType = "LDA"
    args.random_seed = seed
    args.num_steps = 100
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
                args=None, summary_args=None,
                pathTestdocs=None, pathTesttensors=None):

    args = args if args is not None else Args()
    summary_args = summary_args if summary_args is not None else Args()

    documents, tensors, data = _make_data_1(pathDocs, pathTensors)
    testset = None
    if(pathTestdocs is not None and pathTesttensors is not None):
        _, __, testset = _make_data_1(pathTestdocs, pathTesttensors)

    model_class = MCLDA.MCLDA
    args.modelType = "MCLDA"
    args.random_seed = seed
    args.num_steps = 200
    args.step_subsample = 10
    args.K = 6
    args.D = len(data[0][0]) if len(data[0]) != 0 else (len(data[1][0]) if len(data[1]) != 0 else (len(data[2][0]) if len(data[2]) != 0 else 0))
    args.n_rh = [len(habitLevels2[rh]) for rh in range(len(tensors["habit_keys"]))]
    # args.n_rh = []
    args.auto_beta = False
    args.auto_alpha = False
    args.coef_beta = 1
    args.coef_alpha = 0.1
    args.nu_h = 1

    summary_args.full_docs = documents
    summary_args.full_tensors = tensors
    summary_args.full_tensors = tensors
    summary_args.habitWorstLevels = habitWorstLevels2
    print(f"D: {args.D}, K: {args.K}")
    print(f"coef_beta:   {args.coef_beta} (auto: {args.auto_beta})")
    print(f"coef_alpha:  {args.coef_alpha} (auto: {args.auto_alpha})")

    # _excute(model_class, args, data, pathResult, summary_args, testset=testset, from_pickle=True)
    _excute(model_class, args, data, pathResult, summary_args, from_pickle=False)


def excuteMCLDA_K_range(pathDocs, pathTensors, pathResult, *,
                        pathTestdocs=None, pathTesttensors=None):

    args = Args()
    summary_args = Args()

    documents, tensors, data = _make_data_1(pathDocs, pathTensors)
    testset = None
    if(pathTestdocs is not None and pathTesttensors is not None):
        _, __, testset = _make_data_1(pathTestdocs, pathTesttensors)

    model_class = MCLDA.MCLDA
    args.modelType = "MCLDA"
    args.num_steps = 200
    args.step_subsample = 10
    # args.K = 10
    args.D = len(data[0][0]) if len(data[0]) != 0 else (len(data[1][0]) if len(data[1]) != 0 else (len(data[2][0]) if len(data[2]) != 0 else 0))
    args.n_rh = [len(habitLevels2[rh]) for rh in range(len(tensors["habit_keys"]))]
    args.auto_beta = False
    args.auto_alpha = False
    args.coef_beta = 1
    args.coef_alpha = 1
    args.nu_h = 1

    summary_args.full_docs = documents
    summary_args.full_tensors = tensors

    # Ks = [1] + np.arange(10, 101, 10).tolist()
    Ks = np.arange(1, 21, 1).tolist()
    for K in Ks:
        args.K = K
        print(f"D: {args.D}, K: {args.K}")
        # _excute(model_class, args, data, pathResult.joinpath(f"K{K}"), summary_args, testset=testset)
        _excute(model_class, args, data, pathResult.joinpath(f"K{K}"), summary_args,
                testset=testset, from_pickle=False)

    accuracies_rt = []
    accuracies_rm = []
    accuracies_rh = []
    for K in Ks:
        with open(str(pathResult.joinpath(f"K{K}", "accuracy.json")), "r", encoding="utf_8_sig") as f:
            accuracy = json.load(f)
            accuracies_rt.append(accuracy["rt"])
            accuracies_rm.append(accuracy["rm"])
            accuracies_rh.append(accuracy["rh"])

    wb = openpyxl.Workbook()
    ws = wb.create_sheet("Rt")
    writeMatrix(ws, accuracies_rt, 1, 1,
                row_names=Ks,
                column_names=tensors["tensor_keys"],
                addDataBar=True)
    ws = wb.create_sheet("Rm")
    writeMatrix(ws, accuracies_rm, 1, 1,
                row_names=Ks,
                column_names=tensors["measurement_keys"],
                addDataBar=True)
    ws = wb.create_sheet("Rh")
    writeMatrix(ws, accuracies_rh, 1, 1,
                row_names=Ks,
                column_names=tensors["habit_keys"],
                addDataBar=True)
    wb.save(pathResult.joinpath("accuracy.xlsx"))


def _make_data_1(pathDocs, pathTensors):
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
            if(tensors["ＧＯＴ（ＡＳＴ）"][idx] > 200):
                continue
            for rt, key in enumerate(morphomesKeys):
                docs[rt].append(doc[key + "_morphomes"])
            for rm, key in enumerate(measurementKeysHC):
                # if(key == "ＨｂＡ１ｃ（ＮＧＳＰ）"):
                #     measurements[rm].append(tensors[key][idx] + np.random.randn() * 0.1)
                # else:
                measurements[rm].append(tensors[key][idx])
            for rh, key in enumerate(habitKeysHC):
                habits[rh].append(tensors[key][idx])
    data = [docs, np.array(measurements), np.array(habits)]
    # data = [docs, np.array([]), np.array([])]
    # data = [[], np.array(measurements), np.array([])]
    # data = [[], np.array([]), np.array(habits)]

    return documents, tensors, data


def _excute(modelClass, args, data, pathResult, summary_args,
            testset=None, from_pickle=False):

    print(f"Model: {args.modelType}  (to {pathResult}) (from pickle: {from_pickle})")

    args.device = DEVICE
    summary_args.summary_path = pathResult

    if(from_pickle):
        model = torch.load(pathResult.joinpath("model.pickle"), args.device)
    else:
        model = modelClass(args, data)
        losses = []
        for n in range(args.num_steps):
            probability = model.step(args.step_subsample)
            losses.append(probability)
            if((n + 1) % 10 == 0):
                print("i:{:<5d} loss:{:<f}".format(n + 1, probability))

        pathResult.mkdir(exist_ok=True, parents=True)
        plt.plot(losses)
        plt.savefig(pathResult.joinpath("probability.png"))
        plt.clf()
        torch.save(model, pathResult.joinpath("model.pickle"))
    model.summary(summary_args)

    if(testset is not None):
        print("calcurating accuracy")
        model.set_testset(testset)
        accuracy = model.calc_all_mean_accuracy_from_testset(args.step_subsample)
        # accuracy = model.calc_mean_accuracy_from_testset({"rt": [0]}, args.step_subsample, max_iter=5)

        with open(str(pathResult.joinpath("accuracy.json")), "w", encoding="utf_8_sig") as output:
            text = json.dumps(accuracy, ensure_ascii=False, indent=4)
            output.write(text)
