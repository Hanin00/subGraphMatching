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


''''
    completegraph / 


'''

def feature_extract(args, qgId , nodeList):
    ''' Extract feature from subgraphs
    It extracts all subgraphs feature using a trained model.
    and then, it compares DB subgraphs and query subgraphs and finds
    5 candidate DB subgraphs with similar query subgraphs.
    Finally, it counts all candidate DB subgraphs and finds The highest counted image.
    '''
    max_node = 3
    R_BFS = True
    ver = 2

    with open("data/totalEmbDictV3_x100.pickle", "rb") as fr:
        embDict= pickle.load(fr)

    qg0 = nx.complete_graph(list(range(10)))    
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
    query_number = 0

    # model load
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    model = models.GnnEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    if args.model_path:
        model.load_state_dict(torch.load(
            args.model_path, map_location=utils.get_device()))
    else:
        return print("model does not exist")

    temp = []
    results = []
    candidate_imgs = []
    model.eval()
    with torch.no_grad():        
        features_1query = []
        for  idx, i in enumerate(querys):  # idx : subgraph idx of query graph 
            query = []
            query.append(i)
            query = utils.batch_nx_graphs(query, None)
            query = query.to(utils.get_device())
            emb_query_data = model.emb_model(query)
            features_1query.append([qgId, idx, emb_query_data])
    return features_1query

def main():
    start = time.time()

    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    qg0List= ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', ]
    gList = []
    
    [ gList.append( globals()['qg{}List'.format(i)] ) for i in range(10) ]

    totalQueryFeature = []
    for qGId, qGList in enumerate(gList) :  # qGId, qG node name list
        totalQueryFeature.append(feature_extract(args, qGId, qGList))

    with open("cbir_subsg/extract_query_feature/QgfeaturesRaw1029.pickle", "wb") as fr:
        pickle.dump(totalQueryFeature, fr)
    
    end = time.time()
    print(f"{end - start:.5f} sec")

if __name__ == "__main__":
    main()