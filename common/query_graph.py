'''
만든이:이하은
코드 개요: Query를 위한 Query graph 생성
'''
import sys
import numpy as np
import pandas as pd
import torch
import csv

import networkx as nx
import matplotlib.pyplot as plt
# from gensim.models import FastText
from tqdm import tqdm
import time
import json
from collections import Counter
import pickle
# import nltk
# from nltk.corpus import conll2000


#1028 10개 노드를 갖는 쿼리그래프 10개와 쿼리 그래프의 서브 그래프 생성 및 시간 측정
import time
from common.subgraph import make_subgraph

qg0 = nx.complete_graph(list(range(10)))
qg1 = nx.complete_graph(list(range(10)))
qg2 = nx.complete_graph(list(range(10)))
qg3 = nx.complete_graph(list(range(10)))
qg4 = nx.complete_graph(list(range(10)))
qg5 = nx.complete_graph(list(range(10)))
qg6 = nx.complete_graph(list(range(10)))
qg7 = nx.complete_graph(list(range(10)))
qg8 = nx.complete_graph(list(range(10)))
qg9 = nx.complete_graph(list(range(10)))

q1DList = ['A','B','C']
q1D = {string : i for i,string in enumerate(q1DList)}


qg0List = ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', ]
qg1List = ['window', 'building', 'car', 'man', 'sign', 'sidewalk', 'street', 'truck', 'leaf', 'tree', ]
qg2List = ['chair', 'rock', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'person', 'tree', ]
qg3List = ['table', 'light', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper', ]
qg4List = ['keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair']
qg5List = ['shirt', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'jacket', 'leg']
qg6List = ['person', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'rock', 'leaf', 'lamp']
qg7List = ['window', 'table', 'light', 'book', 'desk', 'pillow', 'letter', 'cup', 'bottle', 'ceiling', ]
qg8List = ['table', 'light', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper', ]
qg9List = ['cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair', 'keyboard', 'monitor', ]




for i in range(10) :
    globals()['q{}D'.format(i)] =  {i : string for i,string in enumerate(globals()['qg{}List'.format(i)])}

for idx in range(10):  # Embedding 값은 [3,]인데, 원소 각각을 특징으로 node에 할당
    nx.set_node_attributes(globals()['qg{}'.format(idx)], globals()['q{}D'.format(idx)], "name")
qgList = []
[qgList.append( globals()['qg{}'.format(i)]) for i in range(10)]


[print( globals()['qg{}'.format(i)]) for i in range(10)]
# with open("common/data/query_sub_1028.pickle", "wb") as fw:
#     pickle.dump(qgList, fw)

start = time.time()


db = []
db_idx = []
max_node = 5
R_BFS = True
ver = 2
# Make subgraph from scene graph of Visual Genome
query_number = 5002
for i in range(len(qgList)):
    if query_number == i:
        continue
    subs = make_subgraph(qgList[i], max_node, False, R_BFS)
    db.extend(subs)
    db_idx.extend([i]*len(subs))

print("저장 전 time : ", time.time() - start)
with open("common/data/query_subs_max5.pickle", "wb") as fw:
    pickle.dump(db, fw)
with open("common/data/query_subs_max5_idx.pickle", "wb") as fw:
    pickle.dump(db_idx, fw)

print("저장 후 time : ", time.time() - start)


print(db_idx)
print(type(db))
print(db[0])


sys.exit()



def vGphShow(nexG):
    plt.figure(figsize=[15, 7])
    nx.draw(nexG, with_labels=True)
    plt.show()

# with open("data/networkx_ver2.pickle", "rb") as fr:
#     ver2G = pickle.load(fr)

nodeNames = ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', 'road',
             'window', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'truck', 'leaf', 'tree',
             'chair', 'rock', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'person', 'tree', 'chair',
             'person', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'rock', 'leaf', 'lamp', 'keyboard',
             'monitor', 'plate', 'car', 'car', 'road', 'light',
             'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair',
             'keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair', 'wall',
             'window', 'table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling',
             'table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper',
             'man', 'shirt', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'wall', 'person',
             'shirt', 'bag', 'water', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'jacket']


nodeNames = list(set(nodeNames))


# embDict = {}
# for i in range(len(net200)):
#     names = [row[1] for row in net200[i].nodes(data='name')]
#     f0 = [row[1] for row in net200[i].nodes(data='f0')]
#     f1 = [row[1] for row in net200[i].nodes(data='f1')]
#     f2 = [row[1] for row in net200[i].nodes(data='f2')]

#     for i in range(len(names)):
#         features = [f0[i], f1[i], f2[i]]
#         embDict[names[i]] = features



nodeNameList = [[['light', 'car', 'car', 'tr'],
                ['car', 'road', 'tire']],
                [['tag', 'car', 'car'],
                 ['car', 'plate', 'road']]]


gList = []
for i in range(len(nodeNameList)):
    objNodeName = nodeNameList[i][0]
    subNodeName = nodeNameList[i][1]

    df = pd.DataFrame({"objNodeName": objNodeName,
                      "subNodeName": subNodeName, })
    gI = nx.from_pandas_edgelist(
        df, source='objNodeName', target='subNodeName')

    nodesList = objNodeName + subNodeName

    for index, row in df.iterrows():
        gI.nodes[row['objNodeName']]['name'] = row["objNodeName"]  # name attr
        gI.nodes[row['subNodeName']]['name'] = row['subNodeName']  # name attr

    for i in range(len(nodesList)):  # nodeId
        name = nodesList[i]
        emb = embDict[name]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
        for j in range(3):  # Embedding 값은 [3,]인데, 원소 각각을 특징으로 node에 할당
            nx.set_node_attributes(gI, {name: float(emb[j])}, "f" + str(j))

    dictIdx = {nodeId: idx for idx, nodeId in enumerate(nodesList)}
    gI = nx.relabel_nodes(gI, dictIdx)
    gList.append(gI)


with open("common/data/query_sub_1028.pickle", "wb") as fw:
    pickle.dump(gList, fw)

# with open("./data/query01.pickle", "rb") as fr:
#     gList = pickle.load(fr)

# vGphShow(gList[0])
