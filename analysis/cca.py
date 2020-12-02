import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import openpyxl
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from openpyxl.styles import Font, Color
from utils.openpyxl_util import writeMatrix, writeVector, addColorScaleRules, addBorderToMaxCell, paintCells, writePaintedText, drawImage


def CCA_between_thetas(path_theta1, path_theta2, pathTensors, pathResult):
    n = 7

    data1 = pd.read_csv(path_theta1)
    data2 = pd.read_csv(path_theta2)

    with open(str(pathTensors), 'rb') as f:
        tensors = pickle.load(f)
    measurementKeysHC = tensors["measurement_keys"]
    habitKeysHC = tensors["habit_keys"]
    tensors_before = pd.DataFrame({key: tensors[key] for key in measurementKeysHC})
    tensors_after = pd.DataFrame({key: tensors[key + "_after"] for key in measurementKeysHC})
    tensors_outcome = tensors_after - tensors_before
    aaa = ~tensors_outcome.isnull().all(axis=1)
    tensors_before = tensors_before[aaa]
    tensors_after = tensors_after[aaa]
    tensors_outcome = tensors_outcome[aaa]

    pseqs, idx1, idx2 = np.intersect1d(data1.iloc[:, 0].values, data2.iloc[:, 0].values, assume_unique=True, return_indices=True)
    pseqs, idx3, idxToPID = np.intersect1d(pseqs, np.array(tensors["ids"])[aaa], assume_unique=True, return_indices=True)
    theta1 = data1.iloc[idx1[idx3], 1:].values  # [n_samples, n_features1]
    theta2 = data2.iloc[idx2[idx3], 1:].values  # [n_samples, n_features2]
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

    # matching
    n_sample = len(cca.x_scores_[:, 0])
    ratio_high = 0.1
    num_high = int(n_sample * ratio_high)
    pathResult.joinpath("figs").mkdir(exist_ok=True, parents=True)

    def _save_ws(ws, t, idx1, idx2):
        plt.figure(figsize=(6, 6))
        plt.scatter(cca.x_scores_[:, t], cca.y_scores_[:, t], c="tab:gray", marker=".")
        plt.scatter(cca.x_scores_[idx1, t], cca.y_scores_[idx1, t], c="tab:orange", marker=".")
        plt.scatter(cca.x_scores_[idx2, t], cca.y_scores_[idx2, t], c="tab:blue", marker=".")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig(pathResult.joinpath("figs", f"{ws.title}.png"))
        plt.close()

        writePaintedText(ws, 1, f"各指導タイプが測定項目にどのような影響を与えたのか(n={len(idx1)})", paintColor="000000")
        drawImage(ws, pathResult.joinpath("figs", f"{ws.title}.png"), 200, 200, "B3")
        row = 15
        writePaintedText(ws, row, "指導前の検診データの比較(保健指導から一年前以内)")
        _print_compare_two_group(ws, row + 1, tensors_before.iloc[idxToPID[idx1]], tensors_before.iloc[idxToPID[idx2]])
        row += 7
        writePaintedText(ws, row, "指導後の検診データの比較(保健指導から一年後以内)")
        _print_compare_two_group(ws, row + 1, tensors_after.iloc[idxToPID[idx1]], tensors_after.iloc[idxToPID[idx2]])
        row += 7
        writePaintedText(ws, row, "指導前後の検査データの変化量の比較")
        _print_compare_two_group(ws, row + 1, tensors_outcome.iloc[idxToPID[idx1]], tensors_outcome.iloc[idxToPID[idx2]])
        row += 7

    def _print_compare_two_group(ws, row, values1, values2):
        r = stats.ttest_ind(values1, values2)
        writeMatrix(ws, [values1.mean()], row, 1, row_names=["良さそうな指導"], column_names=measurementKeysHC)
        writeMatrix(ws, [values2.mean()], row + 2, 1, row_names=["悪そうな指導"])
        writeMatrix(ws, [values1.mean() - values2.mean()], row + 3, 1, row_names=["diff"])
        writeMatrix(ws, [r[1]], row + 4, 1, row_names=["pvalue"], rule="databar", ruleBoundary=[0, 1])

    for t in range(n):
        # grouping
        score_xy = cca.x_scores_[:, t] * cca.y_scores_[:, t]
        idx_pos = np.array(np.where((score_xy > 0) & (cca.x_scores_[:, t] > 0))).ravel()  # 第１象限
        idx_high_group = idx_pos[np.argsort(score_xy[idx_pos])[-num_high:]]  # 第１象限でxyが大きいやつ

        idx_neg = np.array(np.where((score_xy > 0) & (cca.x_scores_[:, t] < 0))).ravel()  # 第３象限
        idx_low_group = idx_neg[np.argsort(score_xy[idx_neg])[-num_high:]]  # 第３象限でxyが大きいやつ

        idx_other = np.full_like(score_xy, True, dtype="bool")
        idx_other[idx_high_group] = False
        idx_other[idx_low_group] = False
        idx_middle_group = np.array(np.where(idx_other)).ravel()  # それ以外

        plt.figure(figsize=(16, 12))
        plt.scatter(cca.x_scores_[idx_high_group, t], cca.y_scores_[idx_high_group, t], c="tab:blue", marker=".")
        plt.scatter(cca.x_scores_[idx_low_group, t], cca.y_scores_[idx_low_group, t], c="tab:orange", marker=".")
        plt.scatter(cca.x_scores_[idx_middle_group, t], cca.y_scores_[idx_middle_group, t], c="tab:green", marker=".")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig(pathResult.joinpath(f"scatter_type{t+1}.png"))
        plt.close()

        # matching(pos)
        # idx1, idx2 = nn_matching(cca.x_scores_[idx_high_group, t], cca.x_scores_[idx_middle_group, t], min_range=0.1)
        idx1, idx2 = nn_matching_by_ps(tensors_before.values[idx_high_group], tensors_before.values[idx_middle_group], min_range=0.001)
        _save_ws(wb.create_sheet(f"matching_type{t+1}_pos"), t, idx_high_group[idx1], idx_middle_group[idx2])
        # matching(neg)
        # idx1, idx2 = nn_matching(cca.x_scores_[idx_low_group, t], cca.x_scores_[idx_middle_group, t], min_range=0.1)
        idx1, idx2 = nn_matching_by_ps(tensors_before.values[idx_low_group], tensors_before.values[idx_middle_group], min_range=0.001)
        _save_ws(wb.create_sheet(f"matching_type{t+1}_neg"), t, idx_low_group[idx1], idx_middle_group[idx2])

    pathResult.mkdir(exist_ok=True, parents=True)
    wb.save(pathResult.joinpath("cca_result.xlsx"))


def nn_matching_by_ps(score_group1, score_group2, min_range=float("inf")):
    """
    nearest-neighbor matching by propensity score
    score_group1 : ndarray[n_sample1, n_feature]
    score_group2 : ndarray[n_sample2, n_feature]
    """
    X = np.concatenate([score_group1, score_group2])
    y = np.zeros(X.shape[0], dtype=int)
    y[score_group1.shape[0]:] = 1
    model = LogisticRegression(max_iter=1000).fit(X, y)
    ps = model.predict_proba(X)[:, 0]
    idx1, idx2 = nn_matching(ps[:score_group1.shape[0]], ps[score_group1.shape[0]:], min_range=min_range)
    return idx1, idx2


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
