import numpy as np
import pandas as pd
import openpyxl
from sklearn.cross_decomposition import CCA
from utils.openpyxl_util import writeMatrix, writeVector, addColorScaleRules, addBorderToMaxCell


def CCA_between_thetas(path_theta1, path_theta2, pathResult):
    data1 = pd.read_csv(path_theta1)
    data2 = pd.read_csv(path_theta2)
    pseqs, idx1, idx2 = np.intersect1d(data1.iloc[:, 0].values, data2.iloc[:, 0].values,
                                       assume_unique=True, return_indices=True)
    theta1 = data1.iloc[idx1, 1:].values
    theta2 = data2.iloc[idx2, 1:].values
    names1 = data1.columns[1:]
    names2 = data2.columns[1:]

    n = 5
    cca = CCA(n_components=n)
    cca.fit(theta1, theta2)

    wb = openpyxl.Workbook()
    ws = wb[wb.get_sheet_names()[0]]
    writeMatrix(ws, cca.x_weights_.T, 0 * n + 1, 1, column_names=names1)
    writeMatrix(ws, cca.y_weights_.T, 1 * n + 3, 1, column_names=names2)
    writeMatrix(ws, cca.x_loadings_.T, 2 * n + 5, 1, column_names=names1)
    writeMatrix(ws, cca.y_loadings_.T, 3 * n + 7, 1, column_names=names2)

    pathResult.mkdir(exist_ok=True, parents=True)
    wb.save(pathResult.joinpath("coefs.xlsx"))

    pass
