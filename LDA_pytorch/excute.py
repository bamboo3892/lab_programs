import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import openpyxl
import cv2

from analysis.analyzeHealthCheck import habitLevels2, habitWorstLevels2, medicineLevels, medicineWorstLevels
import LDA_pytorch.LDA as LDA
import LDA_pytorch.MCLDA as MCLDA
from utils.general_util import Args
from utils.openpyxl_util import writeMatrix, writeVector, writeSortedMatrix
from utils.graphic_util import makeColorMap, addImage, DEFAULT_COLOR_FUNC2


seed = 1
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
    args.num_steps = 500
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

    include_medicine = False
    documents, tensors, data = _make_data_1(pathDocs, pathTensors, include_medicine=include_medicine)
    testset = None
    if(pathTestdocs is not None and pathTesttensors is not None):
        _, __, testset = _make_data_1(pathTestdocs, pathTesttensors, include_medicine=include_medicine)

    model_class = MCLDA.MCLDA
    args.modelType = "MCLDA"
    args.random_seed = seed
    args.include_medicine = include_medicine
    args.num_steps = 500
    args.step_subsample = 10
    args.K = 10
    args.D = len(data[0][0]) if len(data[0]) != 0 else (len(data[1][0]) if len(data[1]) != 0 else (len(data[2][0]) if len(data[2]) != 0 else 0))
    args.n_rh = [len(tensors["habit_levels"][rh]) for rh in range(len(tensors["habit_keys"]))]
    # args.n_rh = []
    args.auto_beta = False
    args.auto_alpha = False
    args.coef_beta = 1
    args.coef_alpha = 1
    args.nu_h = 1

    summary_args.full_docs = documents
    summary_args.full_tensors = tensors
    summary_args.full_tensors = tensors
    summary_args.habitWorstLevels = habitWorstLevels2
    if(include_medicine):
        summary_args.habitWorstLevels += medicineWorstLevels
    print(f"D: {args.D}, K: {args.K}")
    print(f"coef_beta:   {args.coef_beta} (auto: {args.auto_beta})")
    print(f"coef_alpha:  {args.coef_alpha} (auto: {args.auto_alpha})")

    # _excute(model_class, args, data, pathResult, summary_args, testset=testset, from_pickle=False)
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
                rule="databar")
    ws = wb.create_sheet("Rm")
    writeMatrix(ws, accuracies_rm, 1, 1,
                row_names=Ks,
                column_names=tensors["measurement_keys"],
                rule="databar")
    ws = wb.create_sheet("Rh")
    writeMatrix(ws, accuracies_rh, 1, 1,
                row_names=Ks,
                column_names=tensors["habit_keys"],
                rule="databar")
    wb.save(pathResult.joinpath("accuracy.xlsx"))


def _make_data_1(pathDocs, pathTensors, include_medicine=False):
    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        documents = json.load(f)
    with open(str(pathTensors), 'rb') as f:
        tensors = pickle.load(f)

    morphomesKeys = tensors["tensor_keys"]
    measurementKeysHC = tensors["measurement_keys"]
    habitKeysHC = tensors["habit_keys"]
    tensors["habit_levels"] = habitLevels2
    if(include_medicine):
        habitKeysHC += tensors["medicine_keys"]
        tensors["habit_levels"] += medicineLevels

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
                measurements[rm].append(tensors[key][idx])
            for rh, key in enumerate(habitKeysHC):
                habits[rh].append(tensors[key][idx])
    data = [docs, np.array(measurements), np.array(habits)]
    # data = [docs, np.empty([0, 0]), np.empty([0, 0])]
    # data = [[], np.array(measurements), np.empty([0, 0])]
    # data = [[], np.empty([0, 0]), np.array(habits)]

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

        variables = None
        hist_conv = []
        rm_boundary = _func0(model)
        pathResult.joinpath("figs").mkdir(exist_ok=True, parents=True)
        out = cv2.VideoWriter(str(pathResult.joinpath("figs", "colormaps.mp4")), cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (500, 500))

        for n in range(args.num_steps):
            probability = model.step(args.step_subsample)
            losses.append(probability)
            if((n + 1) % 10 == 0):
                print("i:{:<5d} loss:{:<f}".format(n + 1, probability))
            variables, conv = _func1(model, summary_args.habitWorstLevels, variables, hist_conv)
            conv = [None, None, None]
            img = _func2(model, variables, conv, rm_boundary)
            out.write(img)

        out.release()
        print("Summarizing the result")
        pathResult.joinpath("figs").mkdir(exist_ok=True, parents=True)
        plt.plot(losses)
        plt.savefig(pathResult.joinpath("figs", "probability.png"))
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


def _func0(model):
    mean = torch.mean(model.x_rm, 1).cpu().detach().numpy().copy()
    std = torch.std(model.x_rm, 1).cpu().detach().numpy().copy()
    lower = np.repeat((mean - std)[None, :], model.K, axis=0)
    upper = np.repeat((mean + std)[None, :], model.K, axis=0)
    return [lower, upper]


def _func1(model, habitWorstLevels, past_variables, hist_conv):
    phi = mu = rho = None
    conv1 = [None, None, None]
    conv2 = [None, None, None]

    if(model.Rt != 0):
        phi = model.phi(0)[:, :10]
        if(past_variables is not None):
            conv1[0] = np.isclose(phi, past_variables[0], rtol=5e-02, atol=0e-08)
            if(len(hist_conv) >= 4):
                conv2[0] = conv1[0]
                for i in range(4):
                    conv2[0] &= hist_conv[-1 - i][0]

    if(model.Rm != 0):
        mu = np.array([model.mu(rm) for rm in range(model.Rm)]).T
        if(past_variables is not None):
            conv1[1] = np.isclose(mu, past_variables[1], rtol=5e-02, atol=0e-08)
            if(len(hist_conv) >= 4):
                conv2[1] = conv1[1]
                for i in range(4):
                    conv2[1] &= hist_conv[-1 - i][1]

    if(model.Rh != 0):
        rho = model.getWorstAnswerProbs(habitWorstLevels).T
        if(past_variables is not None):
            conv1[2] = np.isclose(rho, past_variables[2], rtol=5e-02, atol=0e-08)
            if(len(hist_conv) >= 4):
                conv2[2] = conv1[2]
                for i in range(4):
                    conv2[2] &= hist_conv[-1 - i][2]

    if(past_variables is not None):
        hist_conv.append(conv1)
    return [phi, mu, rho], conv2


def _func2(model, variables, convergences, rm_boundary):
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)

    if(model.Rt != 0):
        img_phi = makeColorMap(variables[0], 40, 10, color_func=DEFAULT_COLOR_FUNC2,
                               boundary=[0., 0.05], border_mask=convergences[0])  # K*40 x 100
        addImage(canvas, img_phi, 0, 0)

    if(model.Rm != 0):
        img_mu = makeColorMap(variables[1], 40, 10, color_func=DEFAULT_COLOR_FUNC2,
                              boundary=rm_boundary, border_mask=convergences[1])  # K*40 x Rm*10
        addImage(canvas, img_mu, 0, 120)

    if(model.Rh != 0):
        img_rho = makeColorMap(variables[2], 40, 10, color_func=DEFAULT_COLOR_FUNC2,
                               boundary=[0., 1.], border_mask=convergences[2])  # K*40 x Rh*10
        addImage(canvas, img_rho, 0, 140 + model.Rm * 10)

    canvas = np.transpose(canvas, [1, 0, 2])
    return canvas
