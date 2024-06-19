import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from early_stopping import EarlyStopping
from model_all import final_model

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def train(model, train_set, test_set, embed, epoch, learn_rate, cross, adj_tri, D_adj, M_adj):
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    cost = nn.CrossEntropyLoss()

    embeds = embed.float().cuda()
    adj_tri = adj_tri.float().cuda()
    D_adj = D_adj.float().cuda()
    M_adj = M_adj.float().cuda()

    early_stopping = EarlyStopping(patience=20, verbose=True, save_path='best_parameter')

    for i in range(epoch):
        model.train()
        LOSS = 0
        for x1, x2, y in train_set:
            x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
            out = model(x1, x2, embeds, adj_tri, D_adj, M_adj)
            loss = cost(out, y)
            LOSS += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Cross: %d  Epoch: %d / %d Loss: %0.7f" % (cross + 1, i + 1, epoch, LOSS))

        early_stopping(LOSS, model)
        if early_stopping.early_stop:
            print(f'early_stopping!')

            test(model, test_set, cross, embeds, adj_tri, D_adj, M_adj)
            break
        # 如果到最后一轮了，保存测试结果
        if i + 1 == epoch:
            test(model, test_set, cross, embeds, adj_tri, D_adj, M_adj)


def test(model, test_set, cross, embeds, adj, D_adj, M_adj):
    correct = 0
    total = 0

    predall, yall = torch.tensor([]), torch.tensor([])
    model.eval()  # 使Dropout失效

    model.load_state_dict(torch.load('best_parameter/best_network.pth'))

    for x1, x2, y in test_set:
        x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
        with torch.no_grad():
            pred = model(x1, x2, embeds, adj, D_adj, M_adj)
        a = torch.max(pred, 1)[1]
        total += y.size(0)
        correct += (a == y).sum()
        predall = torch.cat([predall, torch.as_tensor(pred, device='cpu')], dim=0)
        yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])

    torch.save((predall, yall), 'result/fold_%d' % cross)
    print('Test_acc: ' + str((correct / total).item()))


class MyDataset(Dataset):
    def __init__(self, tri, ld):
        self.tri = tri
        self.ld = ld

    def __getitem__(self, idx):
        x, y = self.tri[idx, :]
        label = self.ld[x][y]
        return x, y, label

    def __len__(self):
        return self.tri.shape[0]


if __name__ == "__main__":

    learn_rate = 0.0005
    epoch = 80
    batch = 32
    embed, train_index, test_index, masked_DM, DM, D, M = torch.load('embed_index_adj_final_2.pth')

    for i in range(5):
        net = final_model().to(device)
        train_set = DataLoader(MyDataset(train_index[i], DM), batch, shuffle=True)
        test_set = DataLoader(MyDataset(test_index[i], DM), batch, shuffle=False)
        train(net, train_set, test_set, embed[i], epoch, learn_rate, i, masked_DM[i], D, M)
