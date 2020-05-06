# -*- coding: utf-8 -*-
"""
Created on Nov 12 2018

@author: takeda masaki
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.style.use('ggplot')

# file paths
DATA_FOLDER = Path("/mnt/nas/prj_qol/福間研共同研究/特定保健指導/data")
ORIGINAL = DATA_FOLDER.joinpath("original")
FORMED = DATA_FOLDER.joinpath("formed")
GRAPH = DATA_FOLDER.joinpath("graph")
RESULT = DATA_FOLDER.joinpath("result")
FILE0 = ORIGINAL.joinpath("Original_data_SOMPO_20180927.csv")
FILE1 = ORIGINAL.joinpath("furikaeri_k_id_50117.csv")


# df = pd.read_csv(FILE0)
# print(df["p_r_tgtset_explan"].head(30))

# df2 = df[["m_id", "c_id", "p_num", "m_bday", "m_sex",
#           "p_r_waist", "p_r_waist_gu", "p_tgt_waist", "p_r_weight", "p_r_weight_gu", "p_tgt_weight",
#           "p_r_smoking", "p_r_exer_score", "p_r_food_score", "p_r_life_score", "p_r_other_score",
#           "p_r_fkr_seqs", "p_r_fkr_scores"]]

# df2 = df2.sort_values(['m_id', 'p_num'], ascending=True)
# df2.to_csv(GRAPH.joinpath("summary.csv"), encoding='utf-8')

# plt.plot(df["p_r_waist_gu"], df["p_tgt_waist"], "o")
# plt.plot([60, 140], [60, 140])
# plt.xlabel("p_r_waist_gu")
# plt.ylabel("p_tgt_waist")
# plt.savefig(str(GRAPH.joinpath("p_r_waist_gu2.png")))
# plt.show()


# with open(str(FORMED.joinpath("sompo.json")), "r", encoding="utf_8_sig") as f0:
#     reviews = json.load(f0)
# reviews2 = []
# for i, review in enumerate(reviews):
#     if(int(review["p_num"]) == 1):
#         review2 = {}
#         if("p_r_tgtset_explan_seqs_id" in review.keys()):
#             review2["p_r_tgtset_explan_seqs_id"] = review["p_r_tgtset_explan_seqs_id"]
#             review2["p_r_tgtset_explan_seqs_text"] = review["p_r_tgtset_explan_seqs_text"]
#         review2["p_r_tgtset_explan"] = review["p_r_tgtset_explan"]
#         reviews2.append(review2)
#         if(len(reviews2) > 100):
#             break
# with open(str(GRAPH.joinpath("p_r_tgtset_explan.json")), "w", encoding="utf_8_sig") as output:
#     json.dump(reviews2, output, ensure_ascii=False, indent=2)


# ID = [1300, 1400, 1500, 1600, 1100, 1200, 1700, 1710]
# LABEL = ["Exercise", "Drink", "Snack", "Alcohol", "Consultation", "Monitaring", "Life rhythm", "Life rhythm"]
# LABEL = ["運動", "飲料", "間食", "飲酒", "適正受診", "モニタリング", "生活リズム", "生活リズム"]
# df = pd.read_csv(FILE1)
# df.dropna(subset=["fkr_category"], axis=0, inplace=True)
# aaa = np.array(df["fkr_category"].astype(int))
# # bbb = aaa[aaa in ID]
# bbb = []
# for i in aaa:
#     if(i in ID):
#         bbb.append(LABEL[ID.index(i)])
# # label, counts = np.unique(aaa, return_counts=True)
# label, counts = np.unique(bbb, return_counts=True)
