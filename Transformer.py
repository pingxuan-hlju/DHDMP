import torch
import torch.nn as nn
from torch import einsum
from parameters import args
from einops import rearrange, repeat




class interact_Block(nn.Module):
    def __init__(self):
        super(interact_Block, self).__init__()
        self.num_attention_heads = args.attention_heads  # 8
        self.attention_head_size = args.head_dim  # 50
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 8 x 50

        self.q = nn.Linear(args.X_dim, self.all_head_size)
        self.k = nn.Linear(args.X_dim, self.all_head_size)
        self.v = nn.Linear(args.X_dim, self.all_head_size)

        self.norm = nn.LayerNorm(args.X_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.attention_dropout)
        self.scale = args.X_dim ** (-0.5)
        self.out = nn.Linear(self.all_head_size, args.X_dim)

    def forward(self, x1, x2):
        X_Drug, X_Microbe = x1, x2  # 32 31 50
        q_Drug, q_Microbe = self.q(X_Drug), self.q(X_Microbe)
        k_Drug, k_Microbe = self.k(X_Drug), self.k(X_Microbe)
        v_Drug, v_Microbe = self.v(X_Drug), self.v(X_Microbe)  # 32x31x(8x50)
        # 32 x 31 x (8x50) -> 32x8x31x50
        q_Drug = rearrange(q_Drug, 'b n (h d) -> b h n d', d=args.head_dim)
        q_Microbe = rearrange(q_Microbe, 'b n (h d) -> b h n d', d=args.head_dim)
        k_Drug = rearrange(k_Drug, 'b n (h d) -> b h n d', d=args.head_dim)
        k_Microbe = rearrange(k_Microbe, 'b n (h d) -> b h n d', d=args.head_dim)
        v_Drug = rearrange(v_Drug, 'b n (h d) -> b h n d', d=args.head_dim)
        v_Microbe = rearrange(v_Microbe, 'b n (h d) -> b h n d', d=args.head_dim)
        # print('q_drug.shape:', q_Drug.shape) # torch.Size([32, 8, 31, 50])
        m1 = einsum('b h i d, b h j d -> b h i j', q_Drug, k_Drug) * self.scale
        m2 = einsum('b h i d, b h j d -> b h i j', q_Drug, k_Microbe) * self.scale
        m3 = einsum('b h i d, b h j d -> b h i j', q_Microbe, k_Microbe) * self.scale
        m4 = einsum('b h i d, b h j d -> b h i j', q_Microbe, k_Drug) * self.scale

        m1, m2, m3, m4 = self.softmax(m1), self.softmax(m2), self.softmax(m3), self.softmax(m4)
        m1, m2, m3, m4 = self.dropout(m1), self.dropout(m2), self.dropout(m3), self.dropout(m4)

        out1 = einsum('b h i j, b h j d -> b h i d', m1, v_Drug)
        out2 = einsum('b h i j, b h j d -> b h i d', m2, v_Microbe)
        out3 = einsum('b h i j, b h j d -> b h i d', m3, v_Microbe)
        out4 = einsum('b h i j, b h j d -> b h i d', m4, v_Drug)
        # 32x8x31x50 -> 32x32x(50x8)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out3 = rearrange(out3, 'b h n d -> b n (h d)')
        out4 = rearrange(out4, 'b h n d -> b n (h d)')
        out1 = self.out(out1)
        out2 = self.out(out2)
        out3 = self.out(out3)
        out4 = self.out(out4)  # 32 31 50

        X_Drug, X_Microbe = (out1 + out2) / 2, (out3 + out4) / 2  # 32 31 50

        return X_Drug, X_Microbe


# Interact Transformer
class InterT(nn.Module):

    def __init__(self):
        super(InterT, self).__init__()

        # 每个节点的embed有多少个patch
        self.patchs = int(args.embed_size / args.patch_size)
        self.dropout = nn.Dropout(args.embed_dropout)
        self.transformer_interact = interact_Block()
        self.linear = nn.Linear(1550, 512)
        self.act = nn.LeakyReLU()

    def to_patch_embed(self, x):
        x = rearrange(x, 'b c h (w p) -> b (h w) (p c)', p=self.patchs)
        x = rearrange(x, 'b p n -> b n p')
        fc = nn.Linear(args.patch_size, args.X_dim).cuda()
        x = fc(x)
        return x

    def forward(self, x1, x2, embeds):
        x3 = x2 + 1373
        # 补0  [1546 1578] -> [1546 1550]
        zero = torch.zeros(size=(args.latdim, 4)).cuda()
        embed = torch.cat([embeds, zero], dim=1)

        x_drug = embed[x1][:, None, None, :]  # 32 1 1 1550
        x_microbe = embed[x3][:, None, None, :]  # 32 1 1 1550

        x_drug = self.to_patch_embed(x_drug)
        x_microbe = self.to_patch_embed(x_microbe)  # 32x31x50

        x_drug = self.dropout(x_drug)
        x_microbe = self.dropout(x_microbe)  # 32x31x50
        for i in range(args.depth_interact_attention):
            x_drug, x_microbe = self.transformer_interact(x_drug, x_microbe)  # 32 1550

        x_1 = rearrange(x_drug, 'b n p -> b (n p)')[:, None, None, :]  # 32 31 50  -> 32 1 1 1550
        x_2 = rearrange(x_microbe, 'b n p -> b (n p)')[:, None, None, :]
        x_Transformer = torch.cat([x_1, x_2], dim=2)  # 32 1 2 1550
        x_Transformer = self.act(self.linear(x_Transformer))
        # x_original = torch.cat([embeds[x1][:, None, None, :], embeds[x3][:, None, None, :]], dim=2)  # 32 1 2 1546
        # x = torch.cat([x_Transformer, x_original], dim=3)  # 32 1 2 3096
        # x = self.cnn(x)

        return x_Transformer  # 32 1 2 512
