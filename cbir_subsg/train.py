from cbir_subsg.config import parse_encoder
from cbir_subsg.test import validation
from common import utils
from common import models
from common import data
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import os
import argparse
import sys, time, pickle
import multiprocessing as mp
from multiprocessing import Pool, Process, Manager, Array






def load_dataset() :
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
    return dataset

import torch.multiprocessing as mp
def build_model(args):
    if args.method_type == "gnn":
        model = models.GnnEmbedder(1, args.hidden_dim, args)
    # elif args.method_type == "mlp":
    #     model = models.BaselineMLP(1, args.hidden_dim, args)
    device = utils.get_device()
    NGPU = torch.cuda.device_count()
    if NGPU > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(NGPU)))
    # torch.multiprocessing.set_start_method('spawn')
    model.to(device)

    # model.to(utils.get_device())
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path,
                                         map_location=utils.get_device()))
    return model


def make_data_source(args):
    if args.dataset == "scene":
        data_source = data.SceneDataSource("scene")
    # data_source = load_dataset()
    
    return data_source

def train(args, model, dataset, data_source):
    """Train the embedding model.
    args: Commandline arguments
    dataset: Dataset of batch size
    data_source: DataSource class
    """
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    if args.method_type == "gnn":
        # clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
        clf_opt = optim.Adam(model.module.clf_model.parameters(), lr=args.lr)

    model.train()   # dorpout 및 batchnomalization 활성화
    model.zero_grad()   # 학습하기위한 Grad 저장할 변수 초기화
    pos_a, pos_b, pos_label = data_source.gen_batch(
        dataset, True)

    # emb_as, emb_bs = model.emb_model(pos_a), model.emb_model(pos_b)
    emb_as, emb_bs = model.module.emb_model(pos_a), model.module.emb_model(pos_b)
    labels = torch.tensor(pos_label).to(utils.get_device())

    intersect_embs = None
    pred = model(emb_as, emb_bs)
    # loss = model.criterion(pred, intersect_embs, labels)
    loss = model.module.criterion(pred, intersect_embs, labels)
    print("loss", loss)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
    opt.step()
    if scheduler:
        scheduler.step()

    # 분류하기 위해서
    if args.method_type == "gnn":
        with torch.no_grad():
            pred = model.module.predict(pred)  # 해당 부분은 학습에 반영하지 않겠다
            model.module.clf_model.zero_grad()
            pred = model.module.clf_model(pred.unsqueeze(1)).view(-1)
        # with torch.no_grad():
        #     pred = model.predict(pred)  # 해당 부분은 학습에 반영하지 않겠다
        # model.clf_model.zero_grad()
        # pred = model.clf_model(pred.unsqueeze(1)).view(-1)
        criterion = nn.MSELoss()
        clf_loss = criterion(pred.float(), labels.float())
        clf_loss.requires_grad_(True) #https://yjs-program.tistory.com/210
        clf_loss.backward()
        clf_opt.step()

    # acc = torch.mean((pred == labels).type(torch.float))

    return pred, labels, loss.item()


def train_loop(args):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    model = build_model(args)

    data_source = make_data_source(args)
    loaders = data_source.gen_data_loaders(
        args.batch_size, train=False)

    val = []
    batch_n = 0
    epoch = 10
    for e in range(epoch):
        for dataset in loaders:
            if args.test:
                mae = validation(args, model, dataset, data_source)
                val.append(mae)
            else:
                pred, labels, loss = train(
                    args, model, dataset, data_source)

                if batch_n % 100 == 0:
                    print(pred, pred.shape, sep='\n')
                    print(labels, labels.shape, sep='\n')
                    print("epoch :", e, "batch :", batch_n,
                          "loss :", loss)

                batch_n += 1

        if not args.test:
            print("Saving {}".format(args.model_path[:-5]+"_e"+str(e+1)+".pt"))
            torch.save(model.state_dict(),  args.model_path[:-5]+"_e"+str(e+1)+".pt")
        else:
            print(len(loaders))
            print(sum(val)/len(loaders))


def main(force_test=False):
    parser = argparse.ArgumentParser(description='Embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    train_loop(args)


if __name__ == '__main__':
    main()
