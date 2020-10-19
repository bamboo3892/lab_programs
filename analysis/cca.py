import numpy as np
import pandas as pd
import openpyxl
from sklearn.cross_decomposition import CCA
from openpyxl.styles import Font, Color
from utils.openpyxl_util import writeMatrix, writeVector, addColorScaleRules, addBorderToMaxCell


def CCA_between_thetas(path_theta1, path_theta2, pathResult):
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

    n = 7
    cca = CCA(n_components=n, scale=True)
    cca.fit(theta1, theta2)
    type_names = [f"type{t+1}" for t in range(n)]
    colors = [Color('FF0000'), Color('FFFFFF'), Color('0000FF')]

    wb = openpyxl.Workbook()
    ws = wb[wb.get_sheet_names()[0]]
    ws.title = "構造相関係数"
    writeMatrix(ws, cca.x_weights_.T, 0 * n + 1, 1, row_names=type_names, column_names=names1, rule="colorscale", rule_color=colors)
    writeMatrix(ws, cca.y_weights_.T, 1 * n + 3, 1, row_names=type_names, column_names=names2, rule="colorscale", rule_color=colors)

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
    writeMatrix(ws, cca.x_loadings_.T, 0 * n + 1, 3, column_names=names1, rule="colorscale", rule_color=colors)
    writeMatrix(ws, cy[:, None], 1 * n + 3, 1, row_names=type_names, column_names=["寄与率"])
    writeMatrix(ws, cca.y_loadings_.T, 1 * n + 3, 3, column_names=names2, rule="colorscale", rule_color=colors)

    ws = wb.create_sheet("交差負荷量")
    x_cross = np.dot(cca.y_weights_.T, corr12.T).T
    y_cross = np.dot(cca.x_weights_.T, corr12).T
    rx = (x_cross ** 2).mean(axis=0)
    ry = (y_cross ** 2).mean(axis=0)
    writeMatrix(ws, rx[:, None], 0 * n + 1, 1, row_names=type_names, column_names=["冗長性係数"])
    writeMatrix(ws, x_cross.T, 0 * n + 1, 3, column_names=names1, rule="colorscale", rule_color=colors)
    writeMatrix(ws, ry[:, None], 1 * n + 3, 1, row_names=type_names, column_names=["冗長性係数"])
    writeMatrix(ws, y_cross.T, 1 * n + 3, 3, column_names=names2, rule="colorscale", rule_color=colors)

    pathResult.mkdir(exist_ok=True, parents=True)
    wb.save(pathResult.joinpath("coefs.xlsx"))

    pass
