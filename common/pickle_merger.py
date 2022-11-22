import os
import sys
import pickle
import networkx as nx
import pandas as pd
import numpy as np

#한 번에 다 하려니까 OOM 나서 3개로 자름
#-> map으로 하면 메모리 덜 먹긴 하는데 6만번대거 돌리는데 OOM 뜸


# 1 3 7 -> 40개의 파일로 나눠서 했을 때, 3만번대거 하다가 꺼짐


print(os.listdir('common/data/trainDataset/'))

#v3_x1001 다시 해야함 1,3,7만 번대 데이터는 60개
FolderList = ['v3_x1001' ]
# for foldername in os.listdir('common/data/trainDataset/'):
for foldername in FolderList:
    fileNum = 60
    fNlist = os.listdir('common/data/trainDataset/'+foldername)
    a = len(fNlist)//fileNum
    for idx in range(fileNum):
        print("foldername : ",foldername)    
        total = [[], [], []]-
        s = a*idx
        e = a*(idx+1)
        try : 
            if idx == (fileNum-1) : 
                for filename in fNlist[s :] :
                    with open("common/data/trainDataset/"+foldername+"/"+filename, "rb") as fr:
                        tmp = pickle.load(fr)
                        filenameList.append(filename)#meta data 확인 시 혹시 필요할까봐
                        total = list(map(list.__add__, total, tmp))

            else : 
                for filename in fNlist[s : e] :
                    with open("common/data/trainDataset/"+foldername+"/"+filename, "rb") as fr:
                        tmp = pickle.load(fr)
                        filenameList.append(filename)#meta data 확인 시 혹시 필요할까봐
                        total = list(map(list.__add__, total, tmp))

        except : 
            continue
        

        with open("common/data/merge/merge_{}_{}.pickle".format(foldername,idx), "wb") as fw:
            pickle.dump(total, fw)
        with open("common/data/merge/merge_{}_{}fnamelist.pickle".format(foldername, idx), "wb") as fw:
            pickle.dump(filenameList, fw)

