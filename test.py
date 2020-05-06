# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
import matplotlib
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import xlwt
import openpyxl
import json
import pandas as pd
import gensim
import math

import sepReviewsToMorphomes
import LLDA_GIBBS.example
import analyzeResult

# fonts = fm.findSystemFonts()
# print([[fm.FontProperties(fname=font).get_name()] for font in fonts])

font = {'family': 'TakaoPGothic'}
matplotlib.rc('font', **font)
matplotlib.font_manager._rebuild()

# path = Path(r"\\10.249.250.242\prj_qol\福間研共同研究\特定保健指導\data\result\sompo\LLDA_gibbs7")
path = Path(r"G:\lab\sompo")
pathTemplate = Path(r"G:\lab\takeda\data\template.xlsm")

DATA_FOLDER = Path(__file__).absolute().parent.parent.joinpath("data")
DICTIONARY = DATA_FOLDER.joinpath("dictionarys")

# LLDA_GIBBS.example.testLLDA(path.joinpath("2018_12_25", "LLDA17_4", "documents.json"),
#                             path.joinpath("2018_12_25", "LLDA_gibbs9", "documents.json"),
#                             path.joinpath("LLDAS30_100", "LLDAS30_0"),
#                             w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))),
#                             fromPickle=True)
# LLDA_GIBBS.example.testLLDAs(path.joinpath("label19", "LLDA19_100", "LLDA19_0", "documents.json"),
#                              path.joinpath("label19", "LLDA19_100", "LLDA19_0", "documents.json"),
#                              path.joinpath("label18", "LLDAS19飲料習慣"), "LLDAS19_",
#                              w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))),
#                              fromPickle=False)


# path = Path(r"G:\lab\sompo\LLDAS30_100\LLDAS30_0")
# analyzeResult.analyze(path, path, pathTemplate)

# label = ["疾病管理", "運動", "食材・バランス", "飲酒", "生活リズム", "タバコ"]
# counts = [0.991393199, 0.740418045, 0.532816077, 0.391694225, 0.509094378, 0.049923684]
# label = ["食事", "設備", "距離", "価格", "スタッフ", "戦略", "フェアウェイ"]
# counts = [0.19832, 0.22598, 0.19072, 0.33264, 0.24192, 0.22366, 0.21106]
# label = ["モニタリング", "正しい病識", "適正受診", "適正服薬", "疾病の自己管理", "活動量", "飲料習慣", "間食習慣", "栄養バランス", "飽和脂肪酸の量", "主菜バランス", "野菜,海藻,きのこ類の量", "塩分量", "飲酒習慣", "夕食の時間と量", "食事リズム", "喫煙"]
# counts = [0.988001357, 0.002289494, 0.196642076, 0.010111931, 0.007886034, 0.740418045, 0.239506487, 0.324662936, 0.024421267, 0.011977444, 0.003773425, 0.028682269, 0.011405071, 0.391694225, 0.496459764, 0.019312304, 0.049923684]

# fig = plt.figure(figsize=(8, 6))
# plt.bar(np.arange(len(label)), counts, tick_label=label)
# plt.xticks(rotation='vertical')
# plt.subplots_adjust(bottom=0.28)
# pp = PdfPages(str(DATA_FOLDER.joinpath("fkr_category_j.pdf")))
# pp.savefig(fig)
# pp.close()
# plt.show()


# sys.exit(0)


llda10 = [[0.123, 0.123, 0.123, 0.123, 0.123, 0.061, 0.110, 0.073, 0.123, 0.078],
          [0.102, 0.082, 0.079, 0.044, 0.078, 0.076, 0.096, 0.105, 0.078, 0.044],
          [0.083, 0.071, 0.068, 0.069, 0.090, 0.083, 0.068, 0.074, 0.125, 0.062],
          [0.026, 0.000, 0.028, 0.000, 0.000, 0.026, 0.026, 0.000, 0.000, 0.026],
          [0.082, 0.082, 0.080, 0.082, 0.082, 0.100, 0.082, 0.082, 0.082, 0.080],
          [0.015, 0.015, 0.015, 0.015, 0.031, 0.037, 0.037, 0.037, 0.037, 0.037],
          [0.163, 0.091, 0.091, 0.091, 0.091, 0.163, 0.091, 0.091, 0.091, 0.091]]
model10 = [[0.129, 0.094, 0.129, 0.094, 0.129, 0.094, 0.129, 0.129, 0.094, 0.094],
           [0.222, 0.148, 0.211, 0.127, 0.222, 0.222, 0.172, 0.152, 0.193, 0.180],
           [0.098, 0.108, 0.098, 0.116, 0.142, 0.080, 0.054, 0.098, 0.078, 0.110],
           [0.039, 0.039, 0.025, 0.039, 0.039, 0.039, 0.039, 0.039, 0.039, 0.039],
           [0.089, 0.067, 0.121, 0.072, 0.087, 0.074, 0.087, 0.086, 0.087, 0.089],
           [-0.002, -0.002, 0.012, 0.041, 0.041, 0.012, 0.023, 0.041, -0.002, 0.012],
           [0.347, 0.091, 0.091, 0.091, 0.347, 0.347, 0.091, 0.091, 0.347, 0.091]]
label = ["食事", "設備", "距離", "価格", "スタッフ", "戦略", "フェアウェイ"]

# fig = plt.figure(figsize=(8, 6))
# l = []
# n = []
# for k in range(7):
#     l.append(llda10[k])
#     l.append(model10[k])
#     n.append("LLDA, {}".format(label[k]))
#     n.append("Setting2, {}".format(label[k]))
# l.append(np.average(llda10, axis=0))
# l.append(np.average(model10, axis=0))
# n.append("LLDA, Topic avg.")
# n.append("Setting2, Topic avg.")
# plt.boxplot(l, showfliers=False)
# plt.xlabel("Model name, label name corresponding each topic")
# plt.ylabel("coherence")
# plt.xticks([i + 1 for i in range(16)], n)
# plt.xticks(rotation='vertical')
# plt.subplots_adjust(bottom=0.28)
# pp = PdfPages(str(DATA_FOLDER.joinpath("golf_coherence10.pdf")))
# pp.savefig(fig)
# pp.close()
# plt.show()


def plot_polar(labels, values1, values2):
    # angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, len(labels) + 1, endpoint=True)
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(labels) + 1, endpoint=True)
    for i in range(len(labels) + 1):
        angles[i] -= math.floor(angles[i] / np.pi / 2) * 2 * np.pi
    values1 = np.concatenate((values1, [values1[0]]))  # 閉じた多角形にする
    values2 = np.concatenate((values2, [values2[0]]))  # 閉じた多角形にする
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)  # 軸ラベル
    ax.set_rlim(0, 0.25)
    ax.plot(angles, values1, '-')  # 外枠
    ax.fill(angles, values1, alpha=0.25)  # 塗りつぶし
    ax.plot(angles, values2, '-')  # 外枠
    ax.fill(angles, values2, alpha=0.25)  # 塗りつぶし
    ax.legend(["LLDA", "Setting2"], loc='upper right', bbox_to_anchor=(0.1, 0.1))
    pp = PdfPages(str(DATA_FOLDER.joinpath("golf_coherence_radar.pdf")))
    pp.savefig(fig)
    pp.close()
    # plt.show(fig)
    # plt.close(fig)


labels = ["Topic avg.", "食事", "設備", "距離", "価格", "スタッフ", "戦略", "フェアウェイ"]
values1 = np.concatenate(([np.average(llda10)], np.average(llda10, axis=1)))
values2 = np.concatenate(([np.average(model10)], np.average(model10, axis=1)))
plot_polar(labels, values1, values2)

pass
