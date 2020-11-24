import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from sklearn.cross_decomposition import CCA
from openpyxl.styles import Font, Color
from utils.openpyxl_util import writeMatrix, writeVector, addColorScaleRules, addBorderToMaxCell


def CCA_between_thetas(path_theta1, path_theta2, pathResult):
    n = 7

    data1 = pd.read_csv(path_theta1)
    data2 = pd.read_csv(path_theta2)
    pseqs, idx1, idx2 = np.intersect1d(data1.iloc[:, 0].values, data2.iloc[:, 0].values,
                                       assume_unique=True, return_indices=True)
    theta1 = data1.iloc[idx1, 1:].values  # [n_samples, n_features1]
    theta2 = data2.iloc[idx2, 1:].values  # [n_samples, n_features2]
    names1 = data1.columns[1:]
    names2 = data2.columns[1:]
    corr = np.corrcoef(theta1.T, theta2.T)
    corr11 = corr[0:theta1.shape[1], 0:theta1.shape[1]]  # [n_features1, n_features1]
    corr12 = corr[0:theta1.shape[1], theta1.shape[1]:]  # [n_features1, n_features2]
    corr22 = corr[theta1.shape[1]:, theta1.shape[1]:]  # [n_features2, n_features2]
    type_names = [f"type{t+1}" for t in range(n)]
    colors = [Color('FF0000'), Color('FFFFFF'), Color('0000FF')]

    cca = CCA(n_components=n, scale=True)
    cca.fit(theta1, theta2)

    wb = openpyxl.Workbook()
    ws = wb[wb.get_sheet_names()[0]]
    ws.title = "構造相関係数"
    writeMatrix(ws, cca.x_weights_.T, 0 * n + 1, 1, row_names=type_names, column_names=names1, rule="colorscale", rule_color=colors, ruleBoundary=[-1., 0., 1.])
    writeMatrix(ws, cca.y_weights_.T, 1 * n + 3, 1, row_names=type_names, column_names=names2, rule="colorscale", rule_color=colors, ruleBoundary=[-1., 0., 1.])

    ws = wb.create_sheet("正準変数間の相関係数")
    aaa = np.corrcoef(cca.x_scores_.T, cca.y_scores_.T)
    c_corr = np.array([aaa[i, i + n] for i in range(n)])
    writeMatrix(ws, c_corr[:, None], 1, 1, row_names=type_names, rule="databar", ruleBoundary=[0, 1])
    # aaa = np.corrcoef(np.dot(theta1, cca.x_weights_).T, np.dot(theta2, cca.y_weights_).T)
    # c_corr = np.array([aaa[i, i + n] for i in range(n)])
    # writeMatrix(ws, c_corr[:, None], 1, 5, row_names=type_names, rule="databar", ruleBoundary=[0, 1])

    ws = wb.create_sheet("正準負荷量")
    # cx = (cca.x_loadings_ ** 2).sum(axis=0) / (theta1.var(axis=0).sum())
    # cy = (cca.y_loadings_ ** 2).sum(axis=0) / (theta2.var(axis=0).sum())
    cx = (cca.x_loadings_ ** 2).mean(axis=0)  # 内部で標準化されている
    cy = (cca.y_loadings_ ** 2).mean(axis=0)
    writeMatrix(ws, cx[:, None], 0 * n + 1, 1, row_names=type_names, column_names=["寄与率"])
    writeMatrix(ws, cca.x_loadings_.T, 0 * n + 1, 3, column_names=names1, rule="colorscale", rule_color=colors, ruleBoundary=[-1, 0, 1])
    writeMatrix(ws, cy[:, None], 1 * n + 3, 1, row_names=type_names, column_names=["寄与率"])
    writeMatrix(ws, cca.y_loadings_.T, 1 * n + 3, 3, column_names=names2, rule="colorscale", rule_color=colors, ruleBoundary=[-1, 0, 1])

    ws = wb.create_sheet("交差負荷量")
    x_cross = np.dot(cca.y_weights_.T, corr12.T).T
    y_cross = np.dot(cca.x_weights_.T, corr12).T
    rx = (x_cross ** 2).mean(axis=0)
    ry = (y_cross ** 2).mean(axis=0)
    writeMatrix(ws, rx[:, None], 0 * n + 1, 1, row_names=type_names, column_names=["冗長性係数"])
    writeMatrix(ws, x_cross.T, 0 * n + 1, 3, column_names=names1, rule="colorscale", rule_color=colors, ruleBoundary=[-1, 0, 1])
    writeMatrix(ws, ry[:, None], 1 * n + 3, 1, row_names=type_names, column_names=["冗長性係数"])
    writeMatrix(ws, y_cross.T, 1 * n + 3, 3, column_names=names2, rule="colorscale", rule_color=colors, ruleBoundary=[-1, 0, 1])

    ws = wb.create_sheet("rotation")
    writeMatrix(ws, cca.x_rotations_.T, 0 * n + 1, 1, row_names=type_names, column_names=names1, rule="colorscale", rule_color=colors, ruleBoundary=[-1, 0, 1])
    writeMatrix(ws, cca.y_rotations_.T, 1 * n + 3, 1, row_names=type_names, column_names=names2, rule="colorscale", rule_color=colors, ruleBoundary=[-1, 0, 1])

    pathResult.mkdir(exist_ok=True, parents=True)
    wb.save(pathResult.joinpath("coefs.xlsx"))

    # graph
    n_sample = len(cca.x_scores_[:, 0])
    ratio_high = 0.1
    num_high = int(n_sample * ratio_high)

    for t in range(n):
        score_xy = cca.x_scores_[:, t] * cca.y_scores_[:, t]
        idx_pos = np.array(np.where((score_xy > 0) & (cca.x_scores_[:, t] > 0))).ravel()  # 第１象限
        idx_high_group = idx_pos[np.argsort(score_xy[idx_pos])[-num_high:]]  # 第１象限でxyが大きいやつ

        idx_neg = np.array(np.where((score_xy > 0) & (cca.x_scores_[:, t] < 0))).ravel()  # 第３象限
        idx_low_group = idx_neg[np.argsort(score_xy[idx_neg])[-num_high:]]  # 第３象限でxyが大きいやつ

        idx_other = np.full_like(score_xy, True, dtype="bool")
        idx_other[idx_high_group] = False
        idx_other[idx_low_group] = False
        idx_middle_group = np.array(np.where(idx_other)).ravel()  # それ以外

        idx1, idx2 = nn_matching(cca.x_scores_[idx_high_group, t], cca.x_scores_[idx_middle_group, t], min_range=0.1)

        plt.figure(figsize=(16, 12))
        plt.scatter(cca.x_scores_[idx_high_group, t], cca.y_scores_[idx_high_group, t], c="tab:blue", marker="x")
        plt.scatter(cca.x_scores_[idx_low_group, t], cca.y_scores_[idx_low_group, t], c="tab:orange", marker="x")
        plt.scatter(cca.x_scores_[idx_middle_group, t], cca.y_scores_[idx_middle_group, t], c="tab:green", marker="x")

        plt.scatter(cca.x_scores_[idx_high_group[idx1], t], cca.y_scores_[idx_high_group[idx1], t], c="tab:blue", marker="o")
        plt.scatter(cca.x_scores_[idx_middle_group[idx2], t], cca.y_scores_[idx_middle_group[idx2], t], c="tab:green", marker="o")

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig(pathResult.joinpath(f"scatter_type{t}.png"))
        plt.clf()


def nn_matching(score_group1, score_group2, min_range=float("inf")):
    """
    nearest-neighbor matching
    score_group1 : ndarray[n_sample1]
    score_group2 : ndarray[n_sample2]
    """
    reversed = False
    if(len(score_group1) > len(score_group2)):
        tmp = score_group1
        score_group1 = score_group2
        score_group2 = tmp
        reversed = True

    indices1 = []
    indices2 = []
    matched = np.full_like(score_group2, False, dtype="bool")
    for idx1 in range(len(score_group1)):
        idx_sorted = np.abs(score_group2 - score_group1[idx1]).argsort()
        for idx2 in idx_sorted:
            if(np.abs(score_group1[idx1] - score_group2[idx2]) > min_range):
                break
            if(not matched[idx2]):  # マッチング成功
                indices1.append(idx1)
                indices2.append(idx2)
                break

    if(not reversed):
        return indices1, indices2
    else:
        return indices2, indices1


def getNearestIdx(lis, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param lis: データ配列
    @param num: 対象値
    @return 対象値に最も近い要素のインデックス
    """

    idx = np.abs(np.array(lis) - num).argmin()
    return idx
