# -*- coding: utf-8 -*-
"""
Created on SEP 19 2018\n
Install MeCab to use this module\n
@author: takeda masaki
"""

import MeCab
import json
import re
import sys
import numpy as np
import csv


SEP_CHARS = ["。", "!", "！", "．", "."]


def sepReviews(pathFormed, pathMorphomes, *,
               inputTextKey,
               outputTextKey,
               sepChars=SEP_CHARS,
               blacklist=None, whitelist=None,
               blackWord=[], whiteWord=[],
               removeRateFromBtm=-1, removeRateFromTop=-1,
               nDocument=-1, startIndex=0, minWordsInSentence=1):
    """
    Separate reviews in json file to morphomes\n
    "wordcounts.csv" will be created in the same directory.\n
    blacklist:  Ban words by black list(regex)\n
    whitelist:  Ban words by white list(regex)\n
    removeRateFromBtm: The Rate at which words will be removed from lowest frequency\n
    removeRateFromTop: The Rate at which words will be removed from highest frequency\n
    nDocument:  Number of documents you want to read.\n
                If negative, try to read all documents.\n
    startIndex: Start index of documents to read\n
                Disabled if nDocument is negative
    minWordsInSentence: Restrict too short sencentes
    """

    print("Start separating reviews to morphomes")

    mecab = MeCab.Tagger("-Ochasen")
    if(blacklist is not None):
        black = re.compile(blacklist)
    if(whitelist is not None):
        white = re.compile(whitelist)

    with open(str(pathFormed), "r", encoding="utf_8_sig") as file0:
        reviews = json.load(file0)
        documents = []
        wordcounts = {}
        totalWords0 = 0

        # load documents
        for index, review in enumerate(reviews):
            if(nDocument < 0 or (startIndex <= index and index < startIndex + nDocument)):
                document = []
                sentence = []
                if(inputTextKey not in review):
                    review[inputTextKey] = ""
                words = mecab.parse(review[inputTextKey]).split("\n")
                for word in words:
                    if(len(word) >= 4):
                        morphome = word.split("\t")[2]
                        type0 = word.split("\t")[3]
                        if(morphome in sepChars):
                            if(len(sentence) >= minWordsInSentence):
                                document.append(sentence)
                                sentence = []
                        elif((blacklist is not None) and (black.match(type0) is not None)):
                            continue
                        elif((whitelist is not None) and (white.match(type0) is not None)):
                            sentence.append(morphome)
                            totalWords0 += 1
                            if(morphome not in wordcounts):
                                wordcounts[morphome] = 1
                            else:
                                wordcounts[morphome] += 1
                if(len(sentence) >= minWordsInSentence):
                    document.append(sentence)
                if(len(document) != 0):
                    review[outputTextKey] = document
                    documents.append(review)

                if((index + 1) % 1000 == 0):
                    if(nDocument < 0):
                        sys.stdout.write("\rDocument No. {:>6}".format(index + 1))
                    else:
                        sys.stdout.write("\rLoaded {:>6}/{:>6}".format(index - startIndex + 1, nDocument))
                    sys.stdout.flush()
        if(nDocument < 0):
            sys.stdout.write("\rDocument No. {:>6}".format(index + 1))
        else:
            sys.stdout.write("\rLoaded {:>6}/{:>6}".format(index - startIndex + 1, nDocument))
        sys.stdout.flush()
        print("")
        print("{} documents loaded ({} words)".format(len(documents), totalWords0))

        # restrict words by frequency
        documents2 = documents
        if(removeRateFromBtm > 0 or removeRateFromTop > 0):
            print("Restrict words by frequency")
            documents2 = []
            totalWords1 = 0
            f0 = 0
            f1 = 0
            black = []
            print("Removed from top-----------------------------------------")
            for key, value in sorted(wordcounts.items(), key=lambda x: -x[1]):
                f0 += value / totalWords0
                if(f0 < removeRateFromTop):
                    black.append(key)
                    sys.stdout.write(key + ", ")
                else:
                    print()
                    print("Removed words appearing more than {} times.".format(value))
                    break
            print("Removing from bottom--------------------------------------")
            for key, value in sorted(wordcounts.items(), key=lambda x: x[1]):
                f1 += value / totalWords0
                if(f1 < removeRateFromBtm):
                    black.append(key)
                    # sys.stdout.write(key + ", ")
                else:
                    print()
                    print("Removing words appearing less than {} times.".format(value))
                    break
            white = whiteWord.copy()
            for key in wordcounts.keys():
                if((key not in black) and (key not in blackWord) and (key not in whiteWord)):
                    white.append(key)
            for d, document in enumerate(documents):
                if((d + 1) % 1000 == 0):
                    sys.stdout.write("\rProcessing {:>6}/{:>6}".format(d + 1, len(documents)))
                    sys.stdout.flush()
                d0 = []
                for sentence in document[outputTextKey]:
                    s0 = []
                    for word in sentence:
                        if(word in white):
                            s0.append(word)
                            totalWords1 += 1
                    if(len(s0) >= minWordsInSentence):
                        d0.append(s0)
                if(len(d0) > 0):
                    document[outputTextKey] = d0
                    documents2.append(document)
            sys.stdout.write("\rProcessing {:>6}/{:>6}".format(d + 1, len(documents)))
            sys.stdout.flush()
            print("")
            print("{} documents loaded ({} words, {} wordtypes)".format(len(documents2), totalWords1, len(white)))

    # save json
    with open(str(pathMorphomes), "w", encoding="utf_8_sig") as output:
        text = json.dumps(documents2, ensure_ascii=False)
        text = text.replace("},", "},\n")
        output.write(text)

    # # save word counts
    # with open(str(pathMorphomes.parent.joinpath("wordcounts.csv")), "w", encoding="utf_8_sig") as f0:
    #     writer = csv.writer(f0, lineterminator='\n')
    #     writer.writerow(["word", "count", "valid"])
    #     for k, v in sorted(wordcounts.items(), key=lambda x: -x[1]):
    #         writer.writerow([k, v, k in white])

    print("Finish separating reviews to morphomes")
