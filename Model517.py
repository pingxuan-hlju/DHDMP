import torch as torch
from torch import nn
import torch.nn.functional as F
from parameters import args

from einops import rearrange, repeat

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=(2, 10), stride=1, padding=0)
        self.s1 = nn.MaxPool2d(kernel_size=(1, 10))
        self.c2 = nn.Conv2d(16, 32, kernel_size=(1, 10), stride=1, padding=0)
        self.s2 = nn.MaxPool2d(kernel_size=(1, 10))
        self.leakyrelu = nn.LeakyReLU()
        self.mlp = nn.Sequential(nn.Linear(27 * 32, 300),
                                 nn.LeakyReLU(),
                                 nn.Linear(300, 2),
                                 )

    def forward(self, x):
        # x 32 1 2 2826
        x = self.s1(self.leakyrelu(self.c1(x)))  # 32 1 1 402
        x = self.s2(self.leakyrelu(self.c2(x)))  # 32 1 1 56
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)

        return x


class FFN_homo_drug(nn.Module):
    def __init__(self):
        super(FFN_homo_drug, self).__init__()
        self.L1 = nn.Linear(1536, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        return x


class FFN_homo_mic(nn.Module):
    def __init__(self):
        super(FFN_homo_mic, self).__init__()
        self.L1 = nn.Linear(1536, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        return x


class FFN_hete(nn.Module):
    def __init__(self):
        super(FFN_hete, self).__init__()
        self.L1 = nn.Linear(1536, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        return x


# Graph Convolutional Networks with Node Attention Propagation Mechanism
class NAPGCN(nn.Module):
    def __init__(self):
        super(NAPGCN, self).__init__()

        self.wd1 = nn.Parameter(torch.randn(1546, 1024))
        self.wd2 = nn.Parameter(torch.randn(1024, 512))
        self.wm1 = nn.Parameter(torch.randn(1546, 1024))
        self.wm2 = nn.Parameter(torch.randn(1024, 512))
        self.dm1 = nn.Parameter(torch.randn(2570, 1024))
        self.dm2 = nn.Parameter(torch.randn(1536, 512))
        self.att = nn.Parameter(torch.randn(1546, 1))

        self.ffn_d = FFN_homo_drug()
        self.ffn_m = FFN_homo_mic()
        self.ffn_dm = FFN_hete()

        self.act = nn.LeakyReLU()

    def forward(self, adj_DM, adj_D, adj_M, drg_embed, mic_embed, mix_embed):
        # AXW1 1373x1373  1373x1546  1546x1024
        # AXW2 1373x1373  1373x1024  1024x512
        drg1hop = self.act(adj_D @ drg_embed @ self.wd1)  # 1373 x 1024
        drg2hop = self.act(adj_D @ drg1hop @ self.wd2)  # 1373 x 512
        # AXW1 173x173  173x1546  1546x1024
        # AXW2 173x173  173x1024  1024x512
        mic1hop = self.act(adj_M @ mic_embed @ self.wm1)  # 173 x 1024
        mic2hop = self.act(adj_M @ mic1hop @ self.wm2)  # 173 x 512

        embed1 = torch.cat([drg1hop, mic1hop], dim=0)  # 1546 x 1024
        embed1 = self.att * embed1
        mix_embed = torch.cat([mix_embed, embed1], dim=1)  # 1546 x 2570
        # AXW 1546x1546  1546x2570  2570x1024
        dm1hop = self.act(adj_DM @ mix_embed @ self.dm1)  # 1546 x 1024

        embed2 = torch.cat([drg2hop, mic2hop], dim=0)  # 1546 x 512
        embed2 = self.att * embed2
        dm1hop1 = torch.cat([dm1hop, embed2], dim=1)  # 1546 x 1536
        # AXW 1546x1546  1546x1536  1536x512
        dm2hop = self.act(adj_DM @ dm1hop1 @ self.dm2)  # 1546 x 512

        #   1373x1024 1373x512 173x1024  173x512  1546x1024 1546x512
        return drg1hop, drg2hop, mic1hop, mic2hop, dm1hop, dm2hop


class Neighbor_info_integration(nn.Module):
    def __init__(self):
        super(Neighbor_info_integration, self).__init__()
        self.ffn_d = FFN_homo_drug()
        self.ffn_m = FFN_homo_mic()
        self.ffn_dm = FFN_hete()
        self.act = nn.LeakyReLU()

    def forward(self, drg1hop, drg2hop, mic1hop, mic2hop, dm1hop, dm2hop, x1, x2):
        drg_embed = torch.cat([drg1hop, drg2hop], dim=1)  # 1373 x 1536
        mic_embed = torch.cat([mic1hop, mic2hop], dim=1)  # 173 x 1536
        dm_embed = torch.cat([dm1hop, dm2hop], dim=1)  # 1546 x 1536

        drg_embed = self.ffn_d(drg_embed)  # 1373 x 128
        mic_embed = self.ffn_m(mic_embed)  # 173 x 128
        dm_embed = self.ffn_dm(dm_embed)  # 1546 x 128

        embed_hete_pair = torch.cat([dm_embed[x1][:, None, None, :], dm_embed[x2 + 1373][:, None, None, :]],
                                    dim=2)  # 32 1 2 128
        embed_homo_pair = torch.cat([drg_embed[x1][:, None, None, :], mic_embed[x2][:, None, None, :]],
                                    dim=2)  # 32 1 2 128

        x = torch.cat([embed_homo_pair, embed_hete_pair], dim=3)  # 32 1 2 256
        return x


# 返回原始特征和经过超图的特征
class HGNN(nn.Module):
    def __init__(self):
        super(HGNN, self).__init__()
        self.act = nn.LeakyReLU()
        self.Hyper = nn.Parameter(init(torch.empty(args.latdim, args.hyperNum)))  # 超边矩阵

        self.Dnode_embed = nn.Parameter(torch.randn(1, args.node_type_dim))  # 1 32 节点类型特征
        self.Mnode_embed = nn.Parameter(torch.randn(1, args.node_type_dim))  # 1 32
        self.Wd = nn.Parameter(torch.randn(args.drug_num, args.drug_num))
        self.Wm = nn.Parameter(torch.randn(args.microbe_num, args.microbe_num))

        self.Db = nn.Parameter(torch.rand(1546, 1))
        self.De = nn.Parameter(torch.rand(32, 1))
        self.embed_weight_HGNN = nn.Parameter(init(torch.empty(1546, 1)))

        self.linear = nn.Linear(1578, 512)

    def forward(self, x1, x2, embeds):
        Hyper = embeds @ self.Hyper  # 1546 32

        Dnode_embed = repeat(self.Dnode_embed, '() e -> n e', n=args.drug_num)  # 1373  32
        Mnode_embed = repeat(self.Mnode_embed, '() e -> n e', n=args.microbe_num)  # 173 32

        Dnode_embed = self.Wd @ Dnode_embed  # 1373 32
        Mnode_embed = self.Wm @ Mnode_embed  # 173 32

        node_type = torch.cat([Dnode_embed, Mnode_embed], dim=0)
        embeds = torch.cat([embeds, node_type], dim=1)  # 附带节点特征的embeds

        hyper_embed = self.act((self.Db * Hyper) @ (self.De * Hyper.T) * self.Db @ embeds)  # 32 1578
        # hyper_embed1 = self.act(Hyper @ Hyper.T @ hyper_embed)  # 32 1578

        # 加权重
        hyper_embed = self.embed_weight_HGNN * hyper_embed

        x_drug = hyper_embed[x1][:, None, None, :]  # 32 1 1 1578
        x_microbe = hyper_embed[x2 + 1373][:, None, None, :]  # 32 1 1 1578

        x_microbe = self.act(self.linear(x_microbe))
        x_drug = self.act(self.linear(x_drug))
        x_hyperembed = torch.cat([x_drug, x_microbe], dim=2)

        return x_hyperembed  # 32 1 2 512
