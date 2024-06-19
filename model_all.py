import torch
import torch.nn as nn
from parameters import args
from Transformer import InterT
import Model517


class final_model(nn.Module):
    def __init__(self):
        super(final_model, self).__init__()

        self.NAPGCN = Model517.NAPGCN()
        self.FFN_homo_drug = Model517.FFN_homo_drug()
        self.FFN_homo_mic = Model517.FFN_homo_mic()
        self.FFN_hete = Model517.FFN_hete()
        self.transformer = InterT()
        self.cnn = Model517.CNN()
        self.HGNN = Model517.HGNN()
        self.neighbor_info_integration = Model517.Neighbor_info_integration()

    def forward(self, x1, x2, embeds, adj_DM, adj_D, adj_M):
        x3 = x2 + 1373
        x_transformer = self.transformer(x1, x2, embeds)  # 32 1 2 512
        x_HGNN = self.HGNN(x1, x2, embeds)  # 32 1 2 512

        hete_embed = embeds
        drug_homo_embed = embeds[:1373, :]
        mic_homo_embed = embeds[1373:1546, :]

        drg1hop, drg2hop, mic1hop, mic2hop, dm1hop, dm2hop = self.NAPGCN(adj_DM, adj_D, adj_M, drug_homo_embed, mic_homo_embed, hete_embed)

        x_neighbor = self.neighbor_info_integration(drg1hop, drg2hop, mic1hop, mic2hop, dm1hop, dm2hop, x1, x2)  # 32 1 2 512
        xd_original = embeds[x1][:, None, None, :]
        xm_original = embeds[x3][:, None, None, :]  # 32 1 1 1546
        x_original = torch.cat([xd_original, xm_original], dim=2)  # 32 1 2 1546

        x = torch.cat([x_HGNN, x_transformer, x_neighbor, x_original], dim=3)

        x = self.cnn(x)

        return x
