import pickle
import networkx as nx
import torch
import numpy as np
import pandas
import pandas as pd
import os, sys
import pyarrow.parquet as pq 

pd.set_option('display.max_rows', None)

# with open('data/QgfeaturesRaw1029.pkl'.format(vNum), 'rb') as f:
# with open('cbir_subsg/extract_query_feature/QgfeaturesRaw1029.pickle', 'rb') as f:
with open('cbir_subsg/extract_query_feature/QgfeaturesRaw1030.pickle', 'rb') as f:
    data = pickle.load(f)

gIdList = []
subIdList = []
features = []
for i in range(len(data)) : # #query graph
  for idx in range(len(data[i])) :   # query_subgrapahs - [gId, subIdx, [feature - list[:64]]]
  # for idx in range(len(data[i])) :   # query_subgrapahs - [gId, subIdx, [feature - tensor 64]]
    gIdList.append(data[i][idx][0])
    subIdList.append(data[i][idx][1])
    # features.append(((data[i][idx][2].tolist())[0]))
    features.append((data[i][idx][2]))

# fDf = pd.DataFrame(features)
# idDf = pd.DataFrame({"gId" : gIdList, "subIdList" : subIdList})
# totalDf = pd.concat([idDf, fDf], axis = 1)
totalDf = pd.DataFrame({"gId" : gIdList, "subgId " : subIdList, "features" : features})
print(totalDf.head())


# with open('cbir_subsg/extract_query_feature/QgfeaturesDF.pickle', 'wb') as f:
#     f.dump(totalDf)

totalDf.to_parquet('cbir_subsg/extract_query_feature/QgfeaturesDF.parquet', engine='pyarrow', index=False) 