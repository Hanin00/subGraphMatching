from common import utils
from common import models
from common import subgraph
from cbir_subsg.config import parse_encoder

from collections import Counter

import os,sys
import torch
import argparse
import pickle
import time

import sys
import networkx as nx

# cd ../work_space/CBIR/CBIR-SubSG
# python3 -m cbir_subsg.extract_mkQgLst1
# python3 -m cbir_subsg.extract_mkQgLst1_parallel

#쿼리 그래프 "1개"에 대해 서브그래프를 생성하고, 특징을 추출하는 코드
#쿼리 그래프를 구성하는 노드의 names을 리스트로 받고, 

def feature_extract(args, nodeList):
    ''' Extract feature from subgraphs
    It extracts all subgraphs feature using a trained model.
    and then, it compares DB subgraphs and query subgraphs and finds
    5 candidate DB subgraphs with similar query subgraphs.
    Finally, it counts all candidate DB subgraphs and finds The highest counted image.
    '''
   

    with open("data/totalEmbDictV3_x100.pickle", "rb") as fr:  # 불러오는 데 약 0.02sec
        embDict= pickle.load(fr)

    mkQueryStart = time.time()
    max_node = 3
    R_BFS = True
    ver = 2

    # qg0 = nx.complete_graph(list(range(10)))
    qg0 = nx.dense_gnm_random_graph(10, 45 )# #node, #edgenum
    # qg0List = ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', ]
    qg0List = nodeList
    
    q0D =  {i : string for i,string in enumerate(qg0List)}
    q0D2 =  {string : i for i,string in enumerate(qg0List)}
    nx.set_node_attributes(qg0, q0D, "name")

    qg0= nx.relabel_nodes(qg0, q0D)  #name올림, id 아직 str
    nameList = list(qg0.nodes())
    names = [nameList[i] for i in range(len(nameList))]
    dictionary = { name : float(embDict[name]) for name in names }
    nx.set_node_attributes(qg0, dictionary, "f0")
    qg0 = nx.relabel_nodes(qg0, q0D2)
    nodesList = qg0.nodes()
    querys = subgraph.make_subgraph(qg0, max_node, False, False)
    # print("len(querys : ",len(querys))
    query_number = 0
    mkQueryEnd = time.time()




    # model load
    modelLoadStart = time.time()
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    model = models.GnnEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    if args.model_path:
        model.load_state_dict(torch.load(
            args.model_path, map_location=utils.get_device()))
    else:
        return print("model does not exist")
    model.eval()
    modelLoadEnd = time.time()

    extFeatureStart = time.time()
    
    temp = []
    results = []
    candidate_imgs = []

    with torch.no_grad():        
        features_1query = []
        for  idx, i in enumerate(querys): 
    
            query = []
            query.append(i)
            query = utils.batch_nx_graphs(query, None)
            query = query.to(utils.get_device())
            emb_query_data = model.emb_model(query).tolist()
            features_1query.append([idx, *emb_query_data])

    extFeatureEnd = time.time()

    print(f"len(querys) - modelLoad - mkQuerySubGraph - extFeature")
    print(f"{len(querys)} - {modelLoadEnd  - modelLoadStart:.5f} sec - {mkQueryEnd - mkQueryStart:.5f} sec - {extFeatureEnd - extFeatureStart:.5f} sec")

    # print(f"extFeature : {extFeatureEnd - extFeatureStart:.5f} sec")
    return features_1query



def main():
    start = time.time()

    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()



    global qg0List 
    qg0List= ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', ]
    global qg1List 
    qg1List = ['window', 'building', 'car', 'man', 'sign', 'sidewalk', 'street', 'truck', 'leaf', 'tree', ]
    global qg2List 
    qg2List = ['chair', 'rock', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'person', 'tree', ]
    global qg3List
    qg3List = ['table', 'light', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper', ]
    global qg4List
    qg4List = ['keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair']
    global qg5List
    qg5List = ['shirt', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'jacket', 'leg']
    global qg6List
    qg6List = ['person', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'rock', 'leaf', 'lamp']
    global qg7List
    qg7List = ['window', 'table', 'light', 'book', 'desk', 'pillow', 'letter', 'cup', 'bottle', 'ceiling', ]
    global qg8List
    qg8List = ['table', 'light', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper', ]
    global qg9List 
    qg9List = ['cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair', 'keyboard', 'monitor', ]

    gList = []
    
    [ gList.append( globals()['qg{}List'.format(i)] ) for i in range(10) ]

    extracts = []
    for qgId in range(len(gList)) : 
            features_query = feature_extract(args, gList[qgId])
            extracts.append([[qgId, *fqrow] for fqrow in features_query ])
    
    end = time.time()
    print(f"{end - start:.5f} sec")

    with open("cbir_subsg/extract_query_feature/QgfeaturesRaw1030.pickle", "wb") as fr:
        pickle.dump(extracts, fr)

    

if __name__ == "__main__":
    main()
