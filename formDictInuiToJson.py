# -*- coding: utf-8 -*-
"""
Created on SEP 20 2018\n
Form SentimentDictionary made by Inui\n
Original file can be downloaded from http://www.cl.ecei.tohoku.ac.jp/index.php?Open%20Resources%2FJapanese%20Sentiment%20Polarity%20Dictionary\n
@author: takeda masaki
"""

import json


def formDictInui(pathDict_YOGEN, pathDict_MEISI, pathOutput):

    print("Start forming dict_inui")

    dictionary = {}
    with open(pathDict_YOGEN, "r", encoding="utf_8_sig") as f0:
        text = f0.read()
        words = text.split("\t")
        for word in words:
            l0 = word.split(" ")
            if(len(l0) == 2 and ("ポジ" in l0[1] or "ネガ" in l0[1])):  # 活用してるものは無視
                dictionary[l0[0]] = 1 if "ポジ" in l0[1] else -1

    with open(pathDict_MEISI, "r", encoding="utf_8_sig") as f1:
        text = f1.read()
        words = text.split("\n")
        for word in words:
            l0 = word.split("\t")
            if(len(l0) == 3 and l0[1] in ["p", "n"]):
                dictionary[l0[0]] = 1 if l0[1] == "p" else -1

    with open(pathOutput, "w", encoding="utf_8_sig") as output:
        json.dump(dictionary, output, ensure_ascii=False, indent=2)

    print("Finish forming dict_inui")
