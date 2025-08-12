import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_

class Flatten_Head(nn.Module):
    def __init__(self, individual, enc_in, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = enc_in

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        # x = einops.rearrange(x, 'b c n -> b n c')
        x = x.permute(0, 2, 1) # b c n → b n c

        x = self.norm(x)

        # return einops.rearrange(x, 'b n c -> b c n')
        return x.permute(0, 2, 1) # from [B, N, C] → [B, C, N]

class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop, local_kernel_size = None):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):

        _, _, N = x.size()
        x = einops.rearrange(x, 'b c n -> b n c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b n c -> b c n')
        return x



class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop, local_kernel_size):
        super(TransformerMLPWithConv,self).__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv1d(self.dim1, self.dim2, 1, 1, 0),

        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Sequential(
            nn.Conv1d(self.dim2, self.dim1, 1, 1, 0),
        )
        self.drop2 = nn.Dropout(drop, inplace=True)

        self.dwc = nn.Conv1d(self.dim2, self.dim2, local_kernel_size, 1, local_kernel_size//2, groups=self.dim2)
    def forward(self, x):

        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x