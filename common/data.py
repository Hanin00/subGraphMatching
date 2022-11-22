import os
import pickle
import random
import sys

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset, Generator
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch.utils.data as data
from torch_geometric.datasets import TUDataset, PPI, QM9, Planetoid
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import queue
import scipy.stats as stats

from common import utils

from astar_ged.src.distance import ged, normalized_ged

import time
import sys
import multiprocessing as mp

from multiprocessing import Pool, Process, Manager, Array
 

def mkDatasetInFile(data, s, works) :  
    # dataPath = path + flist[q2.get()+idx] # datapath
    dataset = [[], [], []]
    # dataset = [[], [], []]
    # s = q2.get()
    e = s + works
    print("s : ", s)
    print("e : ", e)
    for i in range(s, e):
        print("i : ", i)
        try :
            dataset[0].append(data[0][i])
            dataset[1].append(data[1][i])
            dataset[2].append(data[2][i])
        except : 
            break
        # print("dataset : ")
        # print(dataset[:10])
        # print(len(dataset))
    return dataset

def mkDataset2(path,flist, q, max_row_per_worker) : 
    for idx in range(max_row_per_worker) :     
        dataPath = path + flist[q.get()+idx] # datapath
        print("dataPath : ",dataPath)
        with open(dataPath, "rb") as fr:
            tmp = pickle.load(fr)
        # totalNum = len(tmp) 
        q2 = mp.Queue()
        workerNum = 40  # 프로세서 개수
        works = 64      # 프로세서 하나 당 처리해야 하는 데이터 수
        
        for i in range(0, len(tmp), works) : 
            q2.put(i)

        workers2 = []
        for i in range(workerNum) : 
            worker2 = mp.Process(target = mkDatasetInFile, args = (data, q2, works))
            workers2.append(worker2)
            worker2.start()
        for worker2 in workers2 : 
            worker2.join()
        return dataset

def mkDataset(path,flist, q, max_row_per_worker) : 
    dataset = [[], [], []]
    for idx in range(max_row_per_worker) :     
        dataPath = path + flist[q.get()+idx] # datapath
        print("dataPath : ",dataPath)
        with open(dataPath, "rb") as fr:
            tmp = pickle.load(fr)
            for i in range(0, len(tmp[0]), 64):
                dataset[0].append(tmp[0][i])
                dataset[1].append(tmp[1][i])
                dataset[2].append(tmp[2][i])
    return dataset

def load_dataset(name):
    """ Load real-world datasets, available in PyTorch Geometric.

    Used as a helper for DiskDataSource.
    """
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    elif name == "proteins":
        dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name="COX2")
        # dataset = TUDataset(root="/tmp/cox2", name="COX2", use_node_attr=True)
    elif name == "aids":
        dataset = TUDataset(root="/tmp/AIDS", name="AIDS")
    elif name == "reddit-binary":
        dataset = TUDataset(root="/tmp/REDDIT-BINARY", name="REDDIT-BINARY")
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name="IMDB-BINARY")
    elif name == "firstmm_db":
        dataset = TUDataset(root="/tmp/FIRSTMM_DB", name="FIRSTMM_DB")
    elif name == "dblp":
        dataset = TUDataset(root="/tmp/DBLP_v1", name="DBLP_v1")
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
    elif name == "qm9":
        dataset = QM9(root="/tmp/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]
    # 테스트 데이터 셋
    # elif name == "scene":
    #     dataset = [[], [], []]
    #     for filename in os.listdir('common/data/DB_dataset_ver3_100000/test'):
    #         with open("common/data/DB_dataset_ver3_100000/test/"+filename, "rb") as fr:
    #             tmp = pickle.load(fr)
    #             dataset[0].extend(tmp[0])
    #             dataset[1].extend(tmp[1])
    #             dataset[2].extend(tmp[2])

    #     return dataset

    # 학습 데이터 셋
    # elif name == "scene":
    #     dataset = [[], [], []]
    #     for foldername in os.listdir('common/data/DB_dataset_ver3_100000/'):
    #         for filename in os.listdir('common/data/DB_dataset_ver3_100000/'+foldername):
    #             with open("common/data/DB_dataset_ver3_100000/"+foldername+"/"+filename, "rb") as fr:
    #                 tmp = pickle.load(fr)
    #                 dataset[0].extend(tmp[0])
    #                 dataset[1].extend(tmp[1])
    #                 dataset[2].extend(tmp[2])

    #     return dataset

    # 검증 데이터 셋
    # elif name == "scene":
    #     dataset = [[], [], []]
    #     for foldername in os.listdir('common/data/DB_dataset_ver3_100000/'):
    #         for filename in os.listdir('common/data/DB_dataset_ver3_100000/'+foldername):
    #             with open("common/data/DB_dataset_ver3_100000/"+foldername+"/"+filename, "rb") as fr:
    #                 tmp = pickle.load(fr)
    #                 for i in range(0, len(tmp[0]), 64):
    #                     dataset[0].append(tmp[0][i])
    #                     dataset[1].append(tmp[1][i])
    #                     dataset[2].append(tmp[2][i])
    #     return dataset

    # 1번 또는 7번에서 인코딩 문제 발생하고 있음
    # elif name == "scene":
    #     dataset = [[], [], []]
    #     loadstart_data = time.time()
    #     for filename in os.listdir('common/data/trainDataset/v3_x1003'):
    #         try : 
    #             with open("common/data/trainDataset/v3_x1003/"+filename, "rb", ) as fr:
    #                 tmp = pickle.load(fr)
    #                 print(filename)
    #                 for i in range(0, len(tmp[0]), 64):
    #                     print(i)
    #                     dataset[0].append(tmp[0][i])
    #                     dataset[1].append(tmp[1][i])
    #                     dataset[2].append(tmp[2][i])
    #         except :  
    #             print("filename : ", filename)
    #             print("filename : ", filename)
    #             print("filename : ", filename)
    #             print("filename : ", filename)
    #             with open("common/data/trainDataset/v3_x1003/"+filename, "r", encoding='utf-8') as fr:
    #                 tmp = pickle.load(fr)
    #                 print("except : ", filename)
    #                 for i in range(0, len(tmp[0]), 64):
    #                     print(i)
    #                     dataset[0].append(tmp[0][i])
    #                     dataset[1].append(tmp[1][i])
    #                     dataset[2].append(tmp[2][i])
    

    # elif name == "scene":
    #     dataset = [[], [], []]
    #     loadstart_data = time.time()
    #     for foldername in os.listdir('common/data/trainDataset/'):
    #         print("foldername : ",foldername)
    #         for filename in os.listdir('common/data/trainDataset/'+foldername):
    #             try : 
    #                 with open("common/data/trainDataset/"+foldername+"/"+filename, "rb", ) as fr:
    #                     tmp = pickle.load(fr)
    #                     print(filename)
    #                     for i in range(0, len(tmp[0]), 64):
    #                         print(i)
    #                         dataset[0].append(tmp[0][i])
    #                         dataset[1].append(tmp[1][i])
    #                         dataset[2].append(tmp[2][i])

    #             except : 
    #                 print(foldername, " : ", filename)
    #                 print(foldername, " : ", filename)
    #                 print(foldername, " : ", filename)
    #                 continue


                    # with open("common/data/trainDataset/"+foldername+"/"+filename, "rb", encoding='utf') as fr:
                    #     tmp = pickle.load(fr)
                    #     print("except : ", filename)
                    #     for i in range(0, len(tmp[0]), 64):
                    #         print(i)
                    #         dataset[0].append(tmp[0][i])
                    #         dataset[1].append(tmp[1][i])
                    #         dataset[2].append(tmp[2][i])
    #일단 6만번대 데이터dptj 100개 데이터에 대해서만 gpu 프로세싱 붙여보기
    #이 부분 불러들이는 것도 멀티 프로세싱..
    elif name == "scene" :

        loadstart_data = time.time()
        flist = os.listdir('common/data/merge/')
        #단일 파일 내에서.
        # flist 내에서 구간을 나눠서 처리할 수 있도록
        mp.set_start_method('spawn')
        q = mp.Queue()
        # manager = Manager()
        # datasetGlb = manager.list()
        datasetGlb = Array('i',range(3))
        works_per_worker = 64      # Number of datasets created by one subgraph
        number_of_worker = 20     # Number of process -> 


        path = "common/data/merge/"+flist[0]
        # path = "common/data/trainDataset/v3_x1003/0_64.pickle"
        print("dataPath : ",path)
        with open(path, "rb") as fr:
            tmp = pickle.load(fr)
            print("len(tmp) : ", len(tmp))

        dataset = [[],[],[]]
        for i in range(0, len(tmp[0]), works_per_worker):
            q.put(i)
        workers = []
        p = Pool(number_of_worker)
        for i in range(number_of_worker) : 
            s = q.get()
            ret = p.apply_async(mkDatasetInFile,(tmp, s, works_per_worker))

            print("여기 : ",ret.get())
            dataset = list(map(list.__add__, dataset, ret.get()))
        p.close()
        p.join()
        
        loadend_data = time.time()                

        print("load time _data.py : ", loadend_data - loadstart_data)
            # worker = mp.Process(target=mkDatasetInFile, args=(tmp, q, works_per_worker, datasetGlb))
            # # worker = mp.Process(target=mkDataset, args=(path, flist, q, works_per_worker))
            # workers.append(worker)
            # worker.start()
            # worker.join()
        # 병렬로 불러 들임

        # for worker in workers:
        #     worker.join()
        
        return dataset
    # elif name == "scene":
    #     dataset = [[], [], []]
    #     loadstart_data = time.time()
    #     flist = os.listdir('common/data/merge/')

    #     flist = flist[:10]

    #     # flist 내에서 구간을 나눠서 처리할 수 있도록
    #     mp.set_start_method('spawn')
    #     q = mp.Queue()
    #     works_per_worker = 64      # Number of datasets created by one subgraph
    #     number_of_worker = 20     # Number of process -> 

    #     for i in range(0, len(flist), works_per_worker):
    #         q.put(i)
    #     workers = []

    #     path = "common/data/merge/"
    #     for i in range(number_of_worker) : 
    #         worker = mp.Process(target=mkDataset2, args=(path, flist, q, works_per_worker))
    #         # worker = mp.Process(target=mkDataset, args=(path, flist, q, works_per_worker))
    #         workers.append(worker)
    #         worker.start()
    #     # 병렬로 불러 들임

    #     for worker in workers:
    #         worker.join()
        
    #     print(dataset[:10])
    #     print(len(dataset))
    #     sys.exit()

    #     return dataset



    # elif name == "scene":
    #     dataset = [[], [], []]
    #     loadstart_data = time.time()
    #     for filename in os.listdir('common/data/v3_x1006/'):
    #         print("foldername : ",filename)
    #         with open("common/data/v3_x1006/"+"/"+filename, "rb") as fr:
    #             tmp = pickle.load(fr)
    #             print(filename)
    #             for i in range(0, len(tmp[0]), 64):
    #                 print(i)
    #                 dataset[0].append(tmp[0][i])
    #                 dataset[1].append(tmp[1][i])
    #                 dataset[2].append(tmp[2][i])
    #     loadend_data = time.time()                

    #     print("load time _data.py : ", loadend_data - loadstart_data)

    #     return dataset

    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name:
                    del graph.name
                x_f = graph.x
                graph = pyg_utils.to_networkx(graph).to_undirected()
                if name != "scene":
                    for j in range(3):
                        nx.set_node_attributes(
                            graph, {idx: f.item() for idx, f in enumerate(x_f[:, j])}, "f"+str(j))

            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task


class DataSource:
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError


class SceneDataSource(DataSource):
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)

    def gen_data_loaders(self, batch_sizes, train=True):
        n = batch_sizes
        l1, l2, l3 = [], [], []
        for i in range(len(self.dataset[0])//batch_sizes):
            l1.append(self.dataset[0][i:i+batch_sizes])
            l2.append(self.dataset[1][i:i+batch_sizes])
            l3.append(self.dataset[2][i:i+batch_sizes])
            print("i : ",i)


        return [[a, b, c] for a, b, c in zip(l1, l2, l3)]

    def gen_batch(self, datas, train):

        pos_d = datas[2]
        pos_a = utils.batch_nx_graphs(datas[0])
        for i in range(len(datas[1])):
            if len(datas[1][i].edges()) == 0:
                datas[1][i] = datas[0][i]
                datas[2][i] = 0.0
        pos_b = utils.batch_nx_graphs(datas[1])
        return pos_a, pos_b, pos_d

        # else:
        #     if len(self.g1)-b > batch_size:
        #         s = b
        #         e = b + batch_size
        #     else:
        #         s = b
        #         e = len(self.g1)
        #     print(len(self.g1))
        #     print(s)
        #     print(e)
        #     pos_a = self.g1[s:e//2]
        #     pos_b = self.g2[s:e//2]
        #     pos_d = self.ged[s:e//2]
        #     neg_a = self.g1[e//2:e]
        #     neg_b = self.g2[e//2:e]
        #     neg_d = self.ged[e//2:e]
        #     print(self.g1[s:e//2])
        #     print(len(pos_a))
        #     pos_a = utils.batch_nx_graphs(pos_a)
        #     pos_b = utils.batch_nx_graphs(pos_b)
        #     neg_a = utils.batch_nx_graphs(neg_a)
        #     neg_b = utils.batch_nx_graphs(neg_b)

        #     return pos_a, pos_b, neg_a, neg_b, pos_d, neg_d


class DiskDataSource(DataSource):
    """ Uses a set of graphs saved in a dataset file to train the model.

    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.

    See the load_dataset function for supported datasets.
    """

    def __init__(self, dataset_name, node_anchored=False, min_size=5,
                 max_size=29):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
                         use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=10, min_size=2, seed=None,
                  filter_negs=False, sample_method="tree-pair"):
        batch_size = a
        train_set, test_set, task = self.dataset
        graphs = train_set if train else test_set

        if seed is not None:
            random.seed(seed)

        pos_a, pos_b, pos_label = [], [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size // 2):
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                # graph : 원 그래프, a : neigh node
                graph, a = utils.sample_neigh(graphs, size)
                b = a[:random.randint(min_size, len(a) - 1)]
            elif sample_method == "subgraph-tree":
                graph = None
                while graph is None or len(graph) < min_size + 1:
                    graph = random.choice(graphs)
                a = graph.nodes
                _, b = utils.sample_neigh([graph], random.randint(min_size,
                                                                  len(graph) - 1))

            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)

            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)

            # 신 버전(GED)
            neigh_a.graph['gid'] = 0
            neigh_b.graph['gid'] = 1
            # d = ged(neigh_a, neigh_b, 'astar', debug=False, timeit=False)
            # d = normalized_ged(d, neigh_a, neigh_b)
            d = 1

            # 구 버전(GED)
            # p_tmp = nx.optimize_graph_edit_distance(neigh_a, neigh_b)
            # for p_i in p_tmp:
            #     p_label = p_i

            pos_a.append(neigh_a)
            pos_b.append(neigh_b)
            pos_label.append(d)

        # print("pos_data finish")
        neg_a, neg_b, neg_label = [], [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph_a, a = utils.sample_neigh(graphs, size)
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                                                                       size - 1))
            elif sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < min_size + 1:
                    graph_a = random.choice(graphs)
                a = graph_a.nodes
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                                                                       len(graph_a) - 1))
            if self.node_anchored:
                neg_a_anchors.append(list(graph_a.nodes)[0])
                neg_b_anchors.append(list(graph_b.nodes)[0])
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
            if filter_negs:
                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    neigh_a, neigh_b)
                if matcher.subgraph_is_isomorphic():  # a <= b (b is subgraph of a)
                    continue

            # 신 버전(GED)
            neigh_a.graph['gid'] = 0
            neigh_b.graph['gid'] = 1
            # d = ged(neigh_a, neigh_b, 'astar', debug=False, timeit=False)
            # d = normalized_ged(d, neigh_a, neigh_b)
            d = 0

            # 구 버전(GED)
            # n_tmp = nx.optimize_graph_edit_distance(neigh_a, neigh_b)
            # for n_i in n_tmp:
            #     n_label = n_i

            neg_a.append(neigh_a)
            neg_b.append(neigh_b)
            neg_label.append(d)

        # print("neg_data finish")

        pos_a = utils.batch_nx_graphs(pos_a, anchors=pos_a_anchors if
                                      self.node_anchored else None)
        pos_b = utils.batch_nx_graphs(pos_b, anchors=pos_b_anchors if
                                      self.node_anchored else None)
        neg_a = utils.batch_nx_graphs(neg_a, anchors=neg_a_anchors if
                                      self.node_anchored else None)
        neg_b = utils.batch_nx_graphs(neg_b, anchors=neg_b_anchors if
                                      self.node_anchored else None)

        return pos_a, pos_b, neg_a, neg_b, pos_label, neg_label


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    for name in ["enzymes", "reddit-binary", "cox2"]:
        data_source = DiskDataSource(name)
        train, test, _ = data_source.dataset
        i = 11
        neighs = [utils.sample_neigh(train, i) for j in range(10000)]
        clustering = [nx.average_clustering(graph.subgraph(nodes)) for graph,
                      nodes in neighs]
        path_length = [nx.average_shortest_path_length(graph.subgraph(nodes))
                       for graph, nodes in neighs]
        #plt.subplot(1, 2, i-9)
        plt.scatter(clustering, path_length, s=10, label=name)
    plt.legend()
    plt.savefig("plots/clustering-vs-path-length.png")
