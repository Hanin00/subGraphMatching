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
import queue

import sys
import networkx as nx

# import torch.multiprocessing as mp
import multiprocessing as mp
from multiprocessing import Queue


#쿼리 그래프 "1개"에 대해 서브그래프를 생성하고, 특징을 추출하는 코드


# def extract(querys, model) : 
#     features_1query = []
#     for  idx, i in enumerate(querys): 
#             query = []
#             query.append(i)
#             query = utils.batch_nx_graphs(query, None)
#             query = query.to(utils.get_device())
#             emb_query_data = model.emb_model(query)
#             features_1query.append([idx, emb_query_data])
#     return features_1query

        
def extract(query, model,idx) : 
    query = utils.batch_nx_graphs2(query, None)
    query = query.to(utils.get_device())
    emb_query_data = model.emb_model(query)
    result = [idx, emb_query_data]


def feature_extract(args):
    ''' Extract feature from subgraphs
    It extracts all subgraphs feature using a trained model.
    and then, it compares DB subgraphs and query subgraphs and finds
    5 candidate DB subgraphs with similar query subgraphs.
    Finally, it counts all candidate DB subgraphs and finds The highest counted image.
    '''
    max_node = 3
    R_BFS = True
    ver = 2

    mkQueryStart = time.time()
    with open("data/totalEmbDictV3_x100.pickle", "rb") as fr:
        embDict= pickle.load(fr)

    # qg0 = nx.complete_graph(list(range(10)))
    qg0 = nx.dense_gnm_random_graph(10, 20)
    # qg0List = ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', ]
    qg0List = ['window', 'building', 'car', 'man', 'sign', 'sidewalk', 'street', 'truck', 'leaf', 'tree', ]
    
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

    mp.set_start_method('spawn')
    model.eval()
    modelLoadEnd = time.time()


# feature extract
    extFeatureStart = time.time()
    with torch.no_grad():        
        num_processes = 4
        model.share_memory()
        processes = []
        print(len(querys))
        
        for  idx in range(0, len(querys)-num_processes, num_processes):
            # q = Queue()
            for rank in range(num_processes):
                query = querys[idx+rank]
                p = mp.Process(target=extract, args=(query, model,idx+rank))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()


    extFeatureEnd = time.time()

    print(f"len(querys) - modelLoad - mkQuerySubGraph - extFeature")
    print(f"{len(querys)} - {modelLoadEnd  - modelLoadStart:.5f} sec - {mkQueryEnd - mkQueryStart:.5f} sec - {extFeatureEnd - extFeatureStart:.5f} sec")


def main():
    start = time.time()

    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)
    
    end = time.time()
    print(f"{end - start:.5f} sec")
    

if __name__ == "__main__":
    main()
