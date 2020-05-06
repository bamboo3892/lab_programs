import numpy as np
import openpyxl
from openpyxl.styles import Font, Color
from openpyxl.formatting.rule import ColorScale, FormatObject, Rule


def writeVector(ws, vector, row=1, column=1, axis="column", names=None,
                matrix_font=None,
                name_font=Font(color="006400")):
    if(axis == "column"):
        writeMatrix(ws, [vector], row=row, column=column, column_names=names,
                    matrix_font=matrix_font, column_name_font=name_font)
    elif(axis == "row"):
        writeMatrix(ws, np.array(vector)[:, None].tolist(), row=row, column=column, row_names=names,
                    matrix_font=matrix_font, row_name_font=name_font)
    else:
        raise Exception("irregal parameter")


def writeMatrix(ws, matrix, row=1, column=1, row_names=None, column_names=None,
                matrix_font=None,
                row_name_font=Font(color="006400"),
                column_name_font=Font(color="006400")):

    offset_row = row if column_names is None else row + 1
    offset_column = column if row_names is None else column + 1

    if(row_names is not None):
        for i in range(len(row_names)):
            ws.cell(offset_row + i, column, row_names[i]).font = row_name_font

    if(column_names is not None):
        for i in range(len(column_names)):
            ws.cell(row, offset_column + i, column_names[i]).font = column_name_font

    for i in range(len(matrix)):
        vec = matrix[i]
        for j in range(len(vec)):
            ws.cell(offset_row + i, offset_column + j, vec[j]).font = matrix_font


def writeSortedMatrix(ws, matrix, axis=0, row=1, column=1, row_names=None, column_names=None,
                      maxwrite=None, order="higher",
                      matrix_font=None,
                      name_font=Font(color="006400")):
    """
    names is none: write sorted matrix values
    names is not none: write sorted names
    """

    idx = np.argsort(matrix, axis)
    if(order == "higher"):
        if(axis == 0):
            idx = idx[::-1, :]
        elif(axis == 1):
            idx = idx[:, ::-1]

    if(maxwrite is not None):
        if(axis == 0):
            idx = idx[:min([idx.shape[0], maxwrite]), :]
        elif(axis == 1):
            idx = idx[:, :min([idx.shape[1], maxwrite])]

    if(axis == 0):
        if(row_names is None):
            data = np.zeros_like(idx, dtype=matrix.dtype)
            for i in range(data.shape[1]):
                data[:, i] = matrix[idx[:, i], i]
        else:
            data = np.array(row_names)[idx].tolist()
    elif(axis == 1):
        if(column_names is None):
            data = np.zeros_like(idx, dtype=matrix.dtype)
            for i in range(data.shape[0]):
                data[i, :] = matrix[i, idx[i, :]]
        else:
            data = np.array(column_names)[idx].tolist()

    if(axis == 0):
        writeMatrix(ws, data, row=row, column=column, column_names=column_names,
                    matrix_font=matrix_font,
                    row_name_font=name_font)
    elif(axis == 1):
        writeMatrix(ws, data, row=row, column=column, row_names=row_names,
                    matrix_font=matrix_font,
                    row_name_font=name_font)
