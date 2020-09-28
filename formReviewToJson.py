# -*- coding: utf-8 -*-
"""
Created on SEP 19 2018

@author: takeda masaki
"""

import sys
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def formHotelReview(pathOriginal, pathFormed, *,
                    nDocument=-1, startIndex=0):
    """
    Form original review file to json file\n
    nDocument:  Number of documents you want to read.\n
                If negative, try to read all documents.\n
                (deprecated)\n
    startIndex: Start index of documents to read\n
                Disabled if nDocument is negative\n
                (deprecated)\n
    """

    print("Start forming hotel reviews to json")
    if(nDocument < 0):
        print("Try to read all the documents")

    reviews = []
    with open(pathOriginal, "r", encoding="utf_8_sig") as original:
        for index, line in enumerate(original):
            if(nDocument < 0 or (startIndex <= index and index < startIndex + nDocument)):
                l0 = line.split("\t")
                d0 = {}
                d0["hotelID"] = int(l0[0])
                d0["date"] = l0[1]
                d0["reviewID"] = int(l0[3])
                d0["review"] = l0[2]
                d0["reply"] = l0[9]
                reviews.append(d0)

                if(nDocument < 0):
                    sys.stdout.write("\rDocument No. {:>6}".format(index + 1))
                else:
                    sys.stdout.write("\rLoaded {:>6}/{:>6}".format(index - startIndex + 1, nDocument))
                sys.stdout.flush()

    print("")
    print("{} documents loaded".format(nDocument if nDocument >= 0 else index - startIndex + 1))

    with open(pathFormed, "w", encoding="utf_8_sig") as output:
        json.dump(reviews, output, ensure_ascii=False, indent=2)

    print("Finish forming hotel reviews to json")


def formGolfReview(pathOriginal, pathFormed):

    print("Start forming golf reviews to json")

    l0 = ["rating", "rating_cost", "rating_staff", "rating_strategy", "rating_meal", "rating_facility", "rating_fairway", "rating_length"]
    l1 = [[] for i in range(len(l0))]

    reviews = []
    with open(pathOriginal, "r", encoding="utf_8_sig") as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            review = {}
            if(row[18].strip() == "" or row[20].strip() == ""):
                continue
            for i in range(len(l0)):
                review[l0[i]] = int(row[9 + i])
                l1[i].append(int(row[9 + i]))
            review["title"] = row[17]
            review["review"] = row[18].replace("\\n", "。")
            review["prefecture"] = row[3]
            review["age"] = row[4]
            review["date"] = row[20]
            reviews.append(review)
            if(len(reviews) % 100 == 0):
                sys.stdout.write("\rDocument No. {:>6}".format(len(reviews)))
    print("")
    print("{} documents loaded".format(len(reviews)))

    for i in range(len(l0)):
        plt.figure()
        plt.hist(l1[i], bins=5, rwidth=0.8)
        plt.title(l0[i])
        plt.savefig(str(pathFormed.parent.joinpath("{}.png".format(l0[i]))))

    with open(pathFormed, "w", encoding="utf_8_sig") as output:
        json.dump(reviews, output, ensure_ascii=False, indent=2)
    print("Finish forming golf reviews to json")


def formSOMPO(pathOriginal, pathFurikaeri, pathNewProgram, pathFormed,
              textLabel="p_r_tgtset_explan", firstGuidanceOnly=False):
    """
    Remove new program person and add "p_r_fkr_seqs_id" to corresponding "fkr_category" id.
    Added elements:
        p_r_fkr_seqs_id     : target category ids "1000,1100,1200"
        p_r_fkr_seqs_text   : targets text "AAA,BBB,CCC"
    """
    print("Start forming SOMPO reviews to json")

    furikaeri = pd.read_csv(pathFurikaeri)
    furikaeri = furikaeri.dropna(subset=['fkr_category'])
    newProgram = list(pd.read_csv(pathNewProgram)["お客様コード"].values)
    nNew = 0

    reviews = []
    with open(str(pathOriginal), "r", encoding="utf_8_sig") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        label = csvfile.readline().split(",")
        a0 = label.index("p_r_fkr_seqs")
        a1 = label.index("m_id")
        a2 = label.index(textLabel)
        a3 = label.index("p_num")
        for n, row in enumerate(spamreader):
            # if(n > 100):
            #     break

            # 新プログラムの人を除く
            n0 = int(row[a1])
            if(n0 in newProgram or row[a2] is None or row[a2] == ""):
                nNew += 1
                continue

            review = {}
            for i in range(len(label)):
                if(label[i] != "" and len(row[i]) != 0):
                    review[label[i]] = row[i]
                if(i == a0):
                    s0 = row[i]
                    s0 = s0[1:len(s0) - 1]
                    if(s0 == ""):
                        continue
                    s0 = s0.split(",")
                    s1 = ""
                    s2 = ""
                    for seq in s0:
                        seq = int(seq)
                        aaa = furikaeri[furikaeri['fkr_seq'] == seq]
                        if(aaa.empty):
                            continue
                        s1 += str(int(np.abs(aaa["fkr_category"]))) + ","
                        s2 += aaa["fkr_text"].values[0] + ","
                    s1 = s1[:len(s1) - 1]
                    s2 = s2[:len(s2) - 1]
                    review["p_r_fkr_seqs_id"] = s1
                    review["p_r_fkr_seqs_text"] = s2
            reviews.append(review)
            sys.stdout.write("\rDocument No. {:>6}".format(n + 1))
    print("")
    print("{} documents loaded".format(len(reviews)))
    print("Detected {} documents recorded by new program, which have been deleted".format(nNew))

    # add p_r_tgtset_explan_seqs_id, p_r_tgtset_explan_seqs_text
    print("Adding p_r_tgtset_explan_seqs, p_r_tgtset_explan_text")
    targets = {}
    for review in reviews:
        if(review["m_id"] not in targets.keys()):
            targets[review["m_id"]] = []
        targets[review["m_id"]].append(review)
    for target in targets.values():
        targetSorted = []
        for p_num in range(1, len(target) + 1):
            for review in target:
                if(int(review["p_num"]) == p_num):
                    targetSorted.append(review)
                    break
        for i in range(len(targetSorted) - 1):
            if("p_r_fkr_seqs_id" in targetSorted[i + 1].keys()):
                targetSorted[i]["p_r_tgtset_explan_seqs_id"] = targetSorted[i + 1]["p_r_fkr_seqs_id"]
                targetSorted[i]["p_r_tgtset_explan_seqs_text"] = targetSorted[i + 1]["p_r_fkr_seqs_text"]

    if(firstGuidanceOnly):
        # 初回以外を除く
        reviews2 = []
        for review in reviews:
            if(review["p_num"] == "1"):
                reviews2.append(review)
        reviews = reviews2
        print("{} documents finally loaded".format(len(reviews)))

    with open(str(pathFormed), "w", encoding="utf_8_sig") as output:
        json.dump(reviews, output, ensure_ascii=False, indent=2)
    print("Finish forming SOMPO reviews to json")
