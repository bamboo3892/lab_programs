import numpy as np
import openpyxl
from openpyxl.styles import Font, Color
from openpyxl.formatting.rule import ColorScale, DataBar, FormatObject, Rule
from openpyxl.utils.cell import get_column_letter


def writeVector(ws, vector, row=1, column=1, axis="column", names=None,
                matrix_font=None,
                name_font=Font(color="006400"),
                addDataBar=False):
    if(axis == "column"):
        writeMatrix(ws, [vector], row=row, column=column, column_names=names,
                    matrix_font=matrix_font, column_name_font=name_font, addDataBar=addDataBar)
    elif(axis == "row"):
        writeMatrix(ws, [[a] for a in vector], row=row, column=column, row_names=names,
                    matrix_font=matrix_font, row_name_font=name_font, addDataBar=addDataBar)
    else:
        raise Exception("irregal parameter")


def writeMatrix(ws, matrix, row=1, column=1, row_names=None, column_names=None,
                matrix_font=None,
                row_name_font=Font(color="006400"),
                column_name_font=Font(color="006400"),
                addDataBar=False):

    offset_row = row if column_names is None else row + 1
    offset_column = column if row_names is None else column + 1

    if(row_names is not None):
        for i in range(len(row_names)):
            ws.cell(offset_row + i, column, row_names[i]).font = row_name_font

    if(column_names is not None):
        for i in range(len(column_names)):
            ws.cell(row, offset_column + i, column_names[i]).font = column_name_font

    maxVecLength = 0
    for i in range(len(matrix)):
        vec = matrix[i]
        maxVecLength = len(vec) if len(vec) > maxVecLength else maxVecLength
        for j in range(len(vec)):
            ws.cell(offset_row + i, offset_column + j, vec[j]).font = matrix_font

    if(addDataBar):
        addDataBarRules(ws, offset_row, offset_row + len(matrix), offset_column, offset_column + maxVecLength)


def writeSortedMatrix(ws, matrix, axis=0, row=1, column=1, row_names=None, column_names=None,
                      maxwrite=None, order="higher",
                      matrix_font=None,
                      name_font=Font(color="006400"),
                      addDataBar=False):
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
                    row_name_font=name_font, addDataBar=addDataBar)
    elif(axis == 1):
        writeMatrix(ws, data, row=row, column=column, row_names=row_names,
                    matrix_font=matrix_font,
                    row_name_font=name_font, addDataBar=addDataBar)


def addDataBarRules(ws, row1, row2, column1, column2, color="00bfff"):
    if(row1 > row2):
        a = row1
        row1 = row2
        row2 = a
    if(column1 > column2):
        a = column1
        column1 = column2
        column2 = a

    area = getAreaLatter(row1, row2, column1, column2)
    data_bar = DataBar(cfvo=[FormatObject(type='min'), FormatObject(type='max')], color=color, showValue=None, minLength=0, maxLength=100)
    ws.conditional_formatting.add(area, Rule(type='dataBar', dataBar=data_bar))


def getAreaLatter(row1, row2, column1, column2):
    return get_column_letter(column1) + str(row1) + ":" + get_column_letter(column2) + str(row2)
