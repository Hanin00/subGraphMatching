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
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, random_split
import torch.utils.data as data
from torch_geometric.datasets import TUDataset, PPI, QM9, Planetoid
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import queue
import scipy.stats as stats

from common import ha_utils as utils

from astar_ged.src.distance import ged, normalized_ged

import time
import sys
# from multiprocessing import Pool, Process, Manager, Array
from torch.multiprocessing import Pool, Process, Manager, Array
 



def mkDataset(fName) :  
    # dataPath = patha + flist[q2.get()+idx] # datapath
    dataPath = 'common/data/merge/'+fName
    with open(dataPath, "rb") as fr:
        tmp = pickle.load(fr)
    dataset = [[], [], []]
    for i in range(0, len(tmp[0]), 64):
        dataset[0].append(tmp[0][i])
        dataset[1].append(tmp[1][i])
        dataset[2].append(tmp[2][i])

    return dataset



def mkDatasetInFile(data, s, works) :  
    # dataPath = patha + flist[q2.get()+idx] # datapath
    dataset = [[], [], []]
    e = s + works
    print("s : ", s)
    print("e : ", e)
    for i in range(s, e):
        # print("i : ", i)
        try :
            dataset[0].append(data[0][i])
            dataset[1].append(dat[1][i])
            dataset[2].append(data[2][i])
        except : 
            break
    return dataset

def mkDataset2(path,flist, s, max_row_per_worker) : 
    #파일명 구간을 나타내는 idx를 기준으로 개수에 맞춰서 pool 생성하고, 
    #파일 불러서 Infile 으로 불러들이기
    for idx in range(max_row_per_worker) : 
        dataPath2 = path + flist[s + idx] # datapath
        print("dataPath : ",dataPath2)
        with open(dataPath2, "rb") as fr:
            tmp2 = pickle.load(fr)
        # totalNum = len(tmp) 
        q2 = mp.Queue()
        workerNum = 1  # 프로세서 개수
        works = 1      # 프로세서 하나 당 처리해야 하는 데이터 수

        dataset2 = [[],[],[]]
        for i in range(0, len(tmp2[0]), workerNum):
            q2.put(i)
        workers2 = []
        p2 = Pool(workerNum)
        for i in range(workerNum) : 
            print(s2)
            s2 = q2.get()
            ret2 = p2.apply_async(mkDatasetInFile,(tmp2, s2, workerNum))
             # print("여기 : ",ret.get())
            dataset2 = list(map(list.__add__, dataset2, ret2.get()))
        p2.close()
        p2.join()
        loadend_data2 = time.time()                

        print("load time _data.py : ", loadend_data2 - loadstart_data2)

        return dataset

# def mkDataset(path,flist, s, max_row_per_worker, seed=None) : 
#     dataset = [[], [], []]
#     # dataset = list(dataset)
#     # random.shuffle(dataset)
#     # if seed is not None:
#     # random.seed(seed)
#     for idx in range(max_row_per_worker) :     
#         # dataPath = path + flist[q.get()+idx] # datapath
#         dataPath = path + flist[s+idx] # datapath
#         print("dataPath : ",dataPath)
#         with open(dataPath, "rb") as fr:
#             tmp = pickle.load(fr)
#             for i in range(0, len(tmp[0]), 64):
#                 dataset[0].append(tmp[0][i])
#                 dataset[1].append(tmp[1][i])
#                 dataset[2].append(tmp[2][i])
#     return dataset

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
    

    elif name == "_scene" : 
        start = time.time()
        flist = os.listdir('common/data/merge/')
        dataset = [[],[],[]]   
        flist = flist[0]
        path = "common/data/merge/"
        dataPath = path+flist
        with open(dataPath, "rb") as fr:
            tmp = pickle.load(fr)
            for i in range(0, len(tmp[0]), 64):
                dataset[0].append(tmp[0][i])
                dataset[1].append(tmp[1][i])
                dataset[2].append(tmp[2][i])
        # for fName in flist : 
        #     path = "common/data/merge/"
        #     dataPath = path+fName
        #     with open(dataPath, "rb") as fr:
        #         tmp = pickle.load(fr)
        #         for i in range(0, len(tmp[0]), 64):
        #             dataset[0].append(tmp[0][i])
        #             dataset[1].append(tmp[1][i])
        #             dataset[2].append(tmp[2][i])

        end = time.time()
        print("load time : ", end-start)    
        return dataset
            
    elif name == "scene" :
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=parkjy76&logNo=221089918474
        loadstart_data = time.time()
        flist = os.listdir('common/data/merge/')
        flist = flist[:3]
        path = 'common/data/merge/'
        # flist = flist[:1]  # 건당 15-16초 정도 걸리는데, 늦어지는 건, multiprocessing 중간에 오류 있어서 그런 듯

        # flist 내에서 구간을 나눠서 처리할 수 있도록
        mp.set_start_method('spawn')
        pool = mp.Pool(processes = 4)
        dataset = pool.map(mkDataset, flist) #폴더 내 여러 파일에 대해 병렬로 읽는 작업 O
        # -> 한 파일 내 병렬로 쪼개서 읽는 것 X
        def collect_result(result):
            results.append(result)

        # map_async
        pool.map_async(worker, jobs, callback=collect_result)




        pool.close()
        pool.join()
        print(len(dataset))
        print(dataset[0][0])
        print(dataset[1][0])
        print(dataset[2][0])

        print(len(dataset[0][0]))
        print(len(dataset[1][0]))
        print(len(dataset[2][0]))

        sys.exit()





        
        # dataset = [[],[],[]]    
        # workers = [] 
        # p = Pool(number_of_worker)sfd
        # path = "common/data/merge/"
        # dataPath = path+flist[0]
        # with open(dataPath, "rb") as fr:
        #     tmp = pickle.load(fr)
        
        # for i in range(number_of_worker) : 
        #     s = q.get()
        #     # ret = p.apply_async(mkDataset,(path, flist, s, works_per_worker))
        #     ret = p.apply_async(mkDatasetInFile,(tmp, s, len(tmp[0])//number_of_worker))
        #     dataset = list(map(list.__add__, dataset, ret.get()))
        # p.close()
        # p.join()

        return dataset

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


'''
    대용량 데이터 dataset - dataloader 구현
    - inputdata는 getitem 함수 호출 시 해당 index 의 Input Tensor를 읽어 학습데이터를 동적으로 생성해 Returne
'''
class SceneDataset(Dataset) : 
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.dataset = load_dataset(dataset_name)
    def __len__(self) : 
        return len(self.img_labels)  #
        








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
