from scipy.io import loadmat
import numpy as np
import torch
from sklearn.model_selection import KFold
torch.manual_seed(1206)


#  归一化的邻接矩阵
def Regularization(adj):
    row = torch.zeros(1373)
    col = torch.zeros(173)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                row[i] += 1
                col[j] += 1

    row = torch.sqrt(row)
    col = torch.sqrt(col)
    a = torch.Tensor([1])
    ADJ = torch.zeros(size=(1373, 173))
    for m in range(adj.shape[0]):
        for n in range(adj.shape[1]):
            if adj[m][n] == 1:
                temp = row[m] * col[n]
                ADJ[m][n] = torch.div(a, temp)

    return ADJ


# @ tensor 版本的shuffle 按维度0
def tensor_shuffle(ts, dim=0):
    return ts[torch.randperm(ts.shape[dim])]


Drug_similarity = torch.from_numpy(np.loadtxt("data/drugsimilarity.txt"))
Microbe_similarity = torch.from_numpy(np.loadtxt("data/microbe_microbe_similarity.txt"))

DM = torch.from_numpy(loadmat("data/net1.mat")['interaction'])
pos_index = DM.nonzero()  # 所有正例坐标 torch.Size([2470, 2])
neg_index = tensor_shuffle((DM == 0).nonzero(), dim=0)  # 所有负例坐标 torch.Size([235059, 2])
rand_num_4940 = torch.randperm(4940)

neg_index, rest_neg_index = neg_index[0: len(pos_index)], neg_index[len(pos_index):]  # 打乱后负例
pos_neg_index = torch.cat((pos_index, neg_index), dim=0)[rand_num_4940]

kflod = KFold(n_splits=5, shuffle=False)

train_index = []
test_index = []
double_DM_masked = []
embedding = []
print(pos_neg_index.shape)
for fold, (train_xy_idx, test_xy_idx) in enumerate(kflod.split(pos_neg_index)):
    print(f'第{fold + 1}折')
    # train_index.append(train_xy_idx)
    train_index.append(pos_neg_index[train_xy_idx,])  # 每折的训练集坐标
    test = pos_neg_index[test_xy_idx]
    test_all = torch.cat([test, rest_neg_index], dim=0)  # 每折的测试机坐标
    test_index.append(test_all)
    DM_i = DM.clone()
    for index in test:  # 遮掩测试机
        if DM[index[0]][index[1]] == 1:
            DM_i[index[0]][index[1]] = 0

    O2 = torch.zeros(size=(173, 173))
    O1 = torch.zeros(size=(1373, 1373))
    DM_i = Regularization(DM_i)
    row1 = torch.cat([O1, DM_i], dim=1)
    row2 = torch.cat([DM_i.T, O2], dim=1)
    # [0    DM]
    # [DM.T  0]
    double_DM = torch.cat([row1, row2], dim=0)  # 拼接双层邻接矩阵
    double_DM_masked.append(double_DM)
    # [DD   DM]
    # # [DM.T MM]
    DD_DM = torch.cat([Drug_similarity, DM_i], dim=1)
    DM_MM = torch.cat([DM_i.T, Microbe_similarity], dim=1)
    embed = torch.cat([DD_DM, DM_MM], dim=0)  # 生成embedding
    embedding.append(embed)

torch.save([embedding, train_index, test_index, double_DM_masked, DM, Drug_similarity, Microbe_similarity],
           './embed_index_adj_final_2.pth')
