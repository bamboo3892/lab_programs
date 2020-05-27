import pickle
import json
import numpy as np
import torch

import LDA_pytorch.LDA as LDA


seed = 1
np.random.seed(seed)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def excuteFromData(modelType, docs,
                   pathResult, *,
                   summary_args=type("args", (object,), {}), args=type("args", (object,), {})):

    print(f"Model: {modelType}  (to {pathResult})")

    args.modelType = modelType
    args.device = DEVICE
    summary_args.path = pathResult

    if(modelType == "LDA"):
        args.num_steps = 200
        args.K = 7
        args.auto_beta = False
        args.auto_alpha = False
        args.coef_beta = 1
        args.coef_alpha = 1  # 大きいほうが同じようなトピックが形成される感じ?

        summary_args.morphome_key = "morphomes"
        data = [doc[summary_args.morphome_key] for doc in docs]
        summary_args.full_docs = docs
        model_class = LDA.LDA
        print(f"coef_beta:   {args.coef_beta} (auto: {args.auto_beta})")
        print(f"coef_alpha:  {args.coef_alpha} (auto: {args.auto_alpha})")

    pathResult.mkdir(exist_ok=True, parents=True)

    model = model_class(args, data)
    for n in range(args.num_steps):
        model.step()

    model.summary(summary_args)

    print("")



def excuteLDAFromPath(modelType, pathDocs, pathResult):

    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        docs = json.load(f)

    excuteFromData(modelType, docs, pathResult)
