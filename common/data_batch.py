from common.subgraph import make_subgraph
from astar_ged.src.distance import ged, normalized_ged

import multiprocessing as mp
import pickle
import random

import time
import csv
import sys



def make_pkl(dataset, queue, train_num_per_row, max_row_per_worker, train, vNum):
    '''Make pickle file(create train dataset)
    Format of one train data is graph1, graph2, ged of graph1 and graph2.
    This process is to create a train dataset from subgraphs.
    It creates `max_row_per_worker` train dataset per subgraph.
    The train dataset have positive and negative dataset at the same rate.
    The positive dataset is created by generating a new subgraph that removes
    particular node from the original subgraph.
    The negative dataset is created by using other subgraphs.
    And then, it stores the processed train dataset per worker in pickle file

    dataset: Subgraphs generated from scene graph
    queue: Counter for checking processed subgraphs (rule flag)
    train_num_per_row: Number of datasets created by one subgraph
    max_row_per_worker: Number of subgraphs processed by one processor
    train: Whether it's learning or not (True or False)

    return: list of subgraph1, list of subgraph2, list of ged
    '''
    g1_list = []
    g2_list = []
    ged_list = []
    cnt = 0
    length = len(dataset)
    while True:
        if queue.empty():
            break
        num = queue.get()
        if length-num > max_row_per_worker:  
            s = num
            e = num + max_row_per_worker
        else:
            s = num
            e = len(dataset)
        for i in range(s, e):
            if train:
                for _ in range(train_num_per_row):
                    dataset[i].graph['gid'] = 0
                    # print(i, dataset[i])
                    if cnt > (train_num_per_row//2):
                        # print(a, b)
                        l = list(dataset[i].nodes())
                        l.remove(random.choice(l))
                        graph2 = dataset[i].subgraph(l)
                        # print(1, r)
                    else:
                        r = random.randrange(length)
                        graph2 = dataset[r]
                    graph2.graph['gid'] = 1
                    d = ged(dataset[i], graph2, 'astar',
                            debug=False, timeit=False)
                    d = normalized_ged(d, dataset[i], graph2)
                    g1_list.append(dataset[i])
                    g2_list.append(graph2)
                    ged_list.append(d)
                    cnt += 1
                cnt = 0
            else:
                dataset[i].graph['gid'] = 0
                r = random.randrange(length)
                dataset[r].graph['gid'] = 1
                d = ged(dataset[i], dataset[r], 'astar',
                        debug=False, timeit=False)
                d = normalized_ged(d, dataset[i], dataset[r])
                g1_list.append(dataset[i])
                g2_list.append(dataset[r])
                ged_list.append(d)
        print(len(g1_list))
        print(len(g2_list))
        print(len(ged_list))    # 1일 때 1088
        sys.exit()
        with open("common/data/v3_x100{}/{}_{}.pickle".format(vNum,s, e), "wb") as fw:
            pickle.dump([g1_list, g2_list, ged_list], fw)
        g1_list = []
        g2_list = []
        ged_list = []


def main(train):

    times = []
    start = time.strftime('%Y.%m.%d - %H:%M:%S')
    times.append("start : "+start)

    mp.set_start_method('spawn')
    q = mp.Queue()
    train_num_per_row = 64      # Number of datasets created by one subgraph
    max_row_per_worker = 64     # Number of Subgraphs processed by one processor
    number_of_worker = 80       # Number of processor

    # vNumList = [3,2,0,8,1,4,9,7,6] # gpu 7에서 2,0번 시작함 gpu8번에서 3, gpu 6번에서 8
    vNumList = [6]
    for vNum in vNumList : 
        with open("data/DatasetVer3/v3_x100{}.pickle".format(vNum), "rb") as fr:
        # with open("data/query_sub_1028.pickle".format(vNum), "rb") as fr:
            dataset = pickle.load(fr)
            # dataset = dataset[:1]

        total = []
        # total_class = set()
        # idx2 = []
        for i in range(len(dataset)):
            if train:
                subs = make_subgraph(dataset[i], 4, False, False)
            else:
                subs = make_subgraph(dataset[i], 3, False, False)
                subs = subs[:2]
            # idx2.append(len(subs))
            # total_class |= {v for idx, v in dataset[i].nodes(data='name')}
            total.extend(subs)
            
        # with open("common/data/query_sub/query_sub_1028.pkl".format(vNum), 'wb') as f:
        #     pickle.dump(total, f, pickle.HIGHEST_PROTOCOL)   
        # with open("common/data/v3_x100{}/subs.pkl".format(vNum), 'wb') as f:
        #     pickle.dump(total, f, pickle.HIGHEST_PROTOCOL)   
        # with open('common/data/v3_x100{}/subs.pkl'.format(vNum), 'rb') as f:
        #     data = pickle.load(f)
        #     print(len(data))


        # # print("class 수 :",len(total_class))

        # # print("각 이미지에 대한 subgraph 수 :", idx2)
        # # print("max", max(idx2), "min", min(idx2))
        # # print("10개 이상 :", len([i for i in idx2 if 10 < i]))
        # # print("20개 이상 :", len([i for i in idx2 if 20 < i]))
        # # print("30개 이상 :", len([i for i in idx2 if 30 < i]))
        # # print("50개 이상 :", len([i for i in idx2 if 50 < i]))
        # # print("100개 이상 :", len([i for i in idx2 if 100 < i]))
        # # print("총 subgraph 수 :", len(total))
        # # exit()

        for i in range(0, len(total), max_row_per_worker):
            q.put(i)

        workers = []
        for i in range(number_of_worker):
            worker = mp.Process(target=make_pkl, args=(
                total, q, train_num_per_row, max_row_per_worker, train,vNum))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()
        
        end = time.strftime('%Y.%m.%d - %H:%M:%S')
        times.append("end : "+end)

        with open("common/data/v3_x100{}/.csv".format(vNum), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(times)
        

if __name__ == "__main__":
    main(True)
