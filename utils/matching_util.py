import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def nn_matching_by_ps(score_group1, score_group2, min_range=float("inf"), model_name="LR"):
    """
    nearest-neighbor matching by propensity score
    score_group1 : ndarray[n_sample1, n_feature]
    score_group2 : ndarray[n_sample2, n_feature]
    model_name: "LR" or "NN2"
    """
    if(model_name == "LR"):
        X = np.concatenate([score_group1, score_group2])
        y = np.zeros(X.shape[0], dtype=int)
        y[score_group1.shape[0]:] = 1
        model = LogisticRegression(max_iter=1000).fit(X, y)
        ps = model.predict_proba(X)[:, 0]
    elif(model_name == "NN2"):
        X = np.concatenate([score_group1, score_group2])
        y = np.zeros((X.shape[0], 2))
        y[:score_group1.shape[0], 0] = 1
        y[score_group1.shape[0]:, 1] = 1
        ps = regression2nn(X, y)

    idx1, idx2 = nn_matching(ps[:score_group1.shape[0]], ps[score_group1.shape[0]:], min_range=min_range)
    return idx1, idx2


def nn_matching(score_group1, score_group2, min_range=float("inf")):
    """
    nearest-neighbor matching
    score_group1 : ndarray[n_sample1]
    score_group2 : ndarray[n_sample2]
    """
    reverse = False
    if(len(score_group1) > len(score_group2)):
        tmp = score_group1
        score_group1 = score_group2
        score_group2 = tmp
        reverse = True

    indices1 = []
    indices2 = []
    matched = np.full_like(score_group2, False, dtype="bool")
    per = np.random.permutation(len(score_group1))
    for idx1 in per:
        idx_sorted = np.abs(score_group2 - score_group1[idx1]).argsort()
        for idx2 in idx_sorted:
            if(np.abs(score_group1[idx1] - score_group2[idx2]) > min_range):
                break
            if(not matched[idx2]):  # マッチング成功
                indices1.append(idx1)
                indices2.append(idx2)
                matched[idx2] = True
                break

    # # check
    # score1 = score_group1[indices1]
    # score2 = score_group2[indices2]
    # r = stats.ttest_ind(score1, score2)
    # print(r[1])

    if(not reverse):
        return indices1, indices2
    else:
        return indices2, indices1


class NN2(nn.Module):
    def __init__(self, n_features):
        super(NN2, self).__init__()
        self.layer1 = nn.Linear(n_features, n_features * 2)
        self.layer2 = nn.Linear(n_features * 2, n_features * 2)  # 全結合層
        self.layer3 = nn.Linear(n_features * 2, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=0)
        return x


def regression2nn(X, y):
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    model = NN2(X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for i in range(1000):
        data, target = Variable(X), Variable(y)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # outputs = model(Variable(torch.from_numpy(X).float()))
    # _, predicted = torch.max(outputs.data, 1)
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y.numpy(), axis=1)
    accuracy = np.sum(y_predicted == y_true) / len(y_predicted)
    print(accuracy)

    return outputs.detach().numpy()[:, 1]
