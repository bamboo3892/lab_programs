# -*- coding: utf-8 -*-
"""
Created on NOV 15 2019

@author: takeda masaki
"""
import json
import sys
import numpy as np
import pandas as pd
import pickle
import datetime


def makeTensor2(pathMorphomes, pathTensor, docType=None, *, morphomesKey="morphomes"):

    with open(str(pathMorphomes), "r", encoding="utf_8_sig") as f:
        reviews = json.load(f)

    words = []
    count = []
    for n, review in enumerate(reviews):
        for l in review[morphomesKey]:
            for w in l:
                if(w not in words):
                    words.append(w)
                    count.append(1)
                idx = words.index(w)
                count[idx] += 1
    idx = np.argsort(count)[::-1]
    words = np.array(words)[idx].tolist()

    t = np.zeros((len(reviews), len(words)), dtype="int8")
    for n, review in enumerate(reviews):
        if((n + 1) % 100 == 0):
            sys.stdout.write("\rDocument No. {:>6}/{:>6}".format(n + 1, len(reviews)))
        for l in review[morphomesKey]:
            for w in l:
                t[n, words.index(w)] += 1
    sys.stdout.write("\rDocument No. {:>6}/{:>6}".format(n + 1, len(reviews)))
    print("")

    if(docType == "golf"):
        print("Adding season tags and score labels")
        labeled_scores = [1, 2, 5]
        labels = ["rating_cost", "rating_staff", "rating_strategy", "rating_meal", "rating_facility", "rating_fairway", "rating_length"]

        for i, review in enumerate(reviews):
            season = _dateToSeason(review["date"])
            review["season"] = season

            l0 = []
            for j, label in enumerate(labels):
                score = int(review[label])
                if(score in labeled_scores):
                    l0.append(j)
            review["labels"] = l0

    print("Saving documents")
    with open(str(pathTensor.joinpath("matrix.pickle")), 'wb') as f:
        pickle.dump(t, f)
    with open(str(pathTensor.joinpath("words.dat")), "w", encoding="utf_8_sig") as f:
        f.write("\t".join(words))
    # with open(str(pathTensor.joinpath("documents.pickle")), 'wb') as f:
    #     pickle.dump(reviews, f)
    with open(str(pathTensor.joinpath("documents.json")), "w", encoding="utf_8_sig") as output:
        text = json.dumps(reviews, ensure_ascii=False)
        text = text.replace("},", "},\n")
        output.write(text)


def makeTensor3_sompo(pathMorphomes, pathWordcount, pathTensor, *, morphomesKey="morphomes"):

    words = pd.read_csv(pathWordcount)
    words = words[words["valid"]]["word"]
    words = words.reset_index()["word"]
    mIDs = []

    with open(str(pathMorphomes), "r", encoding="utf_8_sig") as f:
        reviews = json.load(f)

    t = np.zeros((len(reviews), len(words), 4), dtype="int8")
    for i, review in enumerate(reviews):
        sys.stdout.write("\rDocument No. {:>6}/{:>6}".format(i + 1, len(reviews)))
        mid = int(review["m_id"])
        if(mid not in mIDs):
            mIDs.append(mid)
        n = mIDs.index(mid)
        pnum = int(review["p_num"])
        pnum = pnum if pnum < 4 else 4
        pnum -= 1
        for l in review[morphomesKey]:
            for w in l:
                t[n, words == w, pnum] += 1
    print("")
    t = t[0:len(mIDs), :, :]

    with open(str(pathTensor.joinpath("tensor.pickle")), 'wb') as f:
        pickle.dump(t, f)
    with open(str(pathTensor.joinpath("words.dat")), "w", encoding="utf_8_sig") as f:
        f.write("\t".join(words.tolist()))
    with open(str(pathTensor.joinpath("mIDs.dat")), "w", encoding="utf_8_sig") as f:
        f.write("\t".join(map(str, mIDs)))


def makeTensor3_golf(pathMorphomes, pathTensor, *, morphomesKey="morphomes"):

    with open(str(pathMorphomes), "r", encoding="utf_8_sig") as f:
        reviews = json.load(f)

    words = []
    count = []
    for n, review in enumerate(reviews):
        for l in review[morphomesKey]:
            for w in l:
                if(w not in words):
                    words.append(w)
                    count.append(1)
                idx = words.index(w)
                count[idx] += 1
    idx = np.argsort(count)[::-1]
    words = np.array(words)[idx].tolist()

    t = np.zeros((len(reviews), len(words), 4), dtype="int8")
    for i, review in enumerate(reviews):
        if((i + 1) % 100 == 0):
            sys.stdout.write("\rDocument No. {:>6}/{:>6}".format(i + 1, len(reviews)))
        season = _dateToSeason(review["date"])
        review["season"] = season
        for l in review[morphomesKey]:
            for w in l:
                t[i, words.index(w), season] += 1
    sys.stdout.write("\rDocument No. {:>6}/{:>6}".format(i + 1, len(reviews)))
    print("")

    with open(str(pathTensor.joinpath("tensor.pickle")), 'wb') as f:
        pickle.dump(t, f)
    with open(str(pathTensor.joinpath("words.dat")), "w", encoding="utf_8_sig") as f:
        f.write("\t".join(words))
    # with open(str(pathTensor.joinpath("documents.pickle")), 'wb') as f:
    #     pickle.dump(reviews, f)
    with open(str(pathTensor.joinpath("documents.json")), "w", encoding="utf_8_sig") as output:
        text = json.dumps(reviews, ensure_ascii=False)
        text = text.replace("},", "},\n")
        output.write(text)


def makeTensorForMultiChannel(pathMorphomes, pathHealthCheckRefined, pathTensorPickle,
                              morphomesKeys,
                              measurementKeysHC, habitKeysHC, medicineKeysHC):

    print("Start making tensors for multi channel")
    print('("{}"  to  "{}")'.format(str(pathMorphomes), str(pathTensorPickle)))
    with open(str(pathMorphomes), "r", encoding="utf_8_sig") as f:
        reviews = json.load(f)
    df = pd.read_pickle(pathHealthCheckRefined)

    tensors = {}
    tensors["tensor_keys"] = morphomesKeys
    tensors["measurement_keys"] = measurementKeysHC
    tensors["habit_keys"] = habitKeysHC
    tensors["medicine_keys"] = medicineKeysHC

    pids = []
    HCbefore = []
    HCafter = []
    year1 = datetime.timedelta(days=365)
    row_nan = df.iloc[0].copy()
    row_nan.values[:] = None
    for n, review in enumerate(reviews):
        pid = int(review["p_seq"])
        mid = int(review["m_id"])

        # 対応する検診データをみつける
        df1 = df[df["m_id"] == mid]
        if(len(df1) > 0):
            # この対象者の保健指導から一年前以内の検診データがあるかどうか
            p_date = datetime.datetime.strptime(review["p_s_date"], "%Y-%m-%d")
            df2 = df1[df1["健診受診日"] < p_date]  # 保健指導日より前
            df2 = df2[df2["健診受診日"] > p_date - year1]  # 保健指導日から一年前以内
            if(len(df2) > 0):
                pids.append(pid)
                HCbefore.append(df2.sort_values("健診受診日", ascending=False).iloc[0])

                # この保健指導から一年後以内の検診データがあったら指導後のデータとして追加
                df3 = df1[df1["健診受診日"] > p_date]  # 保健指導日より後
                df3 = df3[df3["健診受診日"] < p_date + year1]  # 保健指導日から一年前以内
                if(len(df3) > 0):
                    HCafter.append(df3.sort_values("健診受診日", ascending=False).iloc[0])
                else:
                    HCafter.append(row_nan)

    tensors["ids"] = pids
    HCbefore = pd.DataFrame(HCbefore)
    HCafter = pd.DataFrame(HCafter)
    print("{} -> {} docs. ({} docs with hc result)".format(len(reviews), len(pids), len(HCafter.dropna(how="all"))))

    for morphomesKey in morphomesKeys:
        words = []
        count = []
        for n, review in enumerate(reviews):
            for l in review[morphomesKey + "_morphomes"]:
                for w in l:
                    if(w not in words):
                        words.append(w)
                        count.append(1)
                    count[words.index(w)] += 1
        idx = np.argsort(count)[::-1]
        words = np.array(words)[idx].tolist()
        count = np.array(count)[idx]

        t = np.zeros((len(pids), len(words)), dtype="int8")
        for review in reviews:
            if(int(review["p_seq"]) in pids):
                n = pids.index(int(review["p_seq"]))
                for l in review[morphomesKey + "_morphomes"]:
                    for w in l:
                        t[n, words.index(w)] += 1

        tensors[morphomesKey] = t
        tensors[morphomesKey + "_words"] = words
        tensors[morphomesKey + "_counts"] = count

        print('\rKey "{}" finished'.format(morphomesKey))

    for key in measurementKeysHC:
        tensors[key] = HCbefore[key].values
        tensors[key + "_after"] = HCafter[key].values
    for key in habitKeysHC:
        tensors[key] = HCbefore[key + "_level"].values
        tensors[key + "_after"] = HCafter[key + "_level"].values
    for key in medicineKeysHC:
        tensors[key] = HCbefore[key + "_level"].values
        tensors[key + "_after"] = HCafter[key + "_level"].values

    print("Saving documents")
    with open(str(pathTensorPickle), 'wb') as f:
        pickle.dump(tensors, f)


def _dateToSeason(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d")
    if(d.month <= 3):
        return 3  # winter
    elif(d.month <= 6):
        return 0  # spring
    elif(d.month <= 3):
        return 1  # summer
    else:
        return 2  # fall
