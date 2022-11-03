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
    # dataset, db_idx, querys, query_idx = load_dataset(max_node, R_BFS)
    # db_data = utils.batch_nx_graphs(dataset, None)
    # db_data = db_data.to(utils.get_device())

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

    with open("data/query_1028.pickle", "rb") as q:
            querys = pickle.load(q)
            query = subgraph.make_subgraph(querys[0], max_node, False, False)
            # query_number = 1
            query_number = 0

    # db_check = [{i[1] for i in d.nodes(data="name")}for d in dataset]
    temp = []
    results = []
    candidate_imgs = []
    model.eval()
    with torch.no_grad():
        # emb_db_data = model.emb_model(db_data)
        features_1query = []
        print("len(querys) :",len(querys)) #query_subgrp List
        for  idx, i in enumerate(querys): 
        # for i in querys:
            # query = temp.copy() # 왜 이렇게 하지? 더 빠른가?지우는 시간 없어서? 근데 그게 그거 아닌가...?
            query = []
            query.append(i)
            query = utils.batch_nx_graphs(query, None)
            query = query.to(utils.get_device())
            emb_query_data = model.emb_model(query)
            features_1query.append([idx, i, emb_query_data])
            print("features_1query[idx] : ",features_1query[idx])
            continue
    return features_1query 
        # print("len(features_1query) : ",len(features_1query)) # 쿼리 그래프의 서브 그래프 개수
        # print("len(features_1query) : ",len(features_1query[0])) # 로우 하나의 개수 총 3 개 여야 함
        
        # print("q_graph idx : ",(features_1query[0][0])) #query graph 0번의 Idx
        # print("q_graph networkx: ",(features_1query[0][1])) # query graph 0번의 query graph_sub 객체 - 노드 개수 확인
        # print("len q_graph networkx: ",len(features_1query[0][1])) # query graph 0번의 query graph_sub 객체 - 노드 개수 확인
        # print("q_graph feature: ",(features_1query[0][2])) #query graph 하나의 feature
        # print("len q_graph feature: ",len(features_1query[0][2])) #query graph 하나의 feature # 64개여야 함
        # print("len q_graph feature: ",len(features_1query[0][2][0])) #query graph 하나의 feature # 64개여야 함
        # with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_dense.pickle", "wb") as fr:
        #     pickle.dump(results, fr)


            # retreival_start_time = time.time()
            # e = torch.sum(torch.abs(emb_query_data - emb_db_data), dim=1)
            # rank = [(i, d) for i, d in enumerate(e)]
            # rank.sort(key=lambda x: x[1])
            # q_check = {n[1] for n in i.nodes(data="name")}
            # print("Q graph nodes :", q_check)
            # print("number of DB subgraph", e.shape)


    #         # result = [(query_idx+1, i)]
    #         result = []
    #         for n, d in rank[:5]:
    #             print("DB img id :", db_idx[n]+1)
    #             print("similarity : {:.5f}".format(d.item()))
    #             print("DB graph nodes :", db_check[n])
    #             result.append((db_idx[n]+1, dataset[n]))

    #             candidate_imgs.append(db_idx[n]+1)

    #         results.append(result)
    #         retreival_time = time.time() - retreival_start_time
    #         print("@@@@@@@@@@@@@@@@@retreival_time@@@@@@@@@@@@@@@@@ :", retreival_time)

    #         # Check similar/same class count with subgraph in DB
    #         checking_in_db = [len(q_check) - len(q_check - i)
    #                           for i in db_check]
    #         checking_result = Counter(checking_in_db)
    #         print(checking_result)

    #         # Check similar/same class with subgraph in DB
    #         value_checking_in_db = [
    #             str(q_check - (q_check - i)) for i in db_check]
    #         value_checking_result = Counter(value_checking_in_db)
    #         print(value_checking_result)
    #         print("==="*20)
    # # Final image rank
    # imgs = Counter(candidate_imgs)
    # print(imgs)

    # # Store result
    # # if R_BFS:
    # #     with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_RBFS.pickle", "wb") as fr:
    # #         pickle.dump(results, fr)
    # # else:
    # #     with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_dense.pickle", "wb") as fr:
    # #         pickle.dump(results, fr)


def load_dataset(max_node, R_BFS):
    ''' Load subgraphs
    Load Scene Graph and then, Creat subgraphs from Scene Graphs.
    First, it reads scene graphs of Visual Genome and then, it makes subgraphs
    Second, it selects query image and then, it makes subgraphs
    ps) It can use user-defined query images

    max_node: When subgraphs create, It configures subgraph size.
    R_BFS: When subgraphs create, Whether it`s R_BFS mothod or not.

    Return
    db: Subgraphs in database
    db_idx: Index image of subgraphs
    query: Query subgraphs/subgraph
    query_number: Query subgraph number
    '''
    with open("data/DatasetVer3/v3_x1000.pickle", "rb") as fr:
        # with open("data/networkx_ver2_10000.pickle", "rb") as fr:
        datas = pickle.load(fr)

    db = []
    db_idx = []

    # Make subgraph from scene graph of Visual Genome
    query_number = 5002
    for i in range(len(datas)):
        if query_number == i:
            continue
        subs = subgraph.make_subgraph(datas[i], max_node, False, R_BFS)
        db.extend(subs)
        db_idx.extend([i]*len(subs))

    # print(db[0])
    # print(db_idx[0])
    
    # print(len(db))
    # print(len(db_idx))
    # sys.exit()
    

    # db = subgraph
    # db_idx = metadata

    # Select query image
    # query = subgraph.make_subgraph(datas[query_number], max_node, False, True)

    # user-defined query images
    with open("data/query_1028.pickle", "rb") as q:
        querys = pickle.load(q)
        query = subgraph.make_subgraph(querys[0], max_node, False, False)
        # query_number = 1
        query_number = 0
    return db, db_idx, query, query_number


def main():
    start = time.time()

    parser = argparse.ArgumentParser(description='embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)

    end = time.time()

    print(f"변환 시간 : {end - start:.5f} sec")

if __name__ == "__main__":
    main()
