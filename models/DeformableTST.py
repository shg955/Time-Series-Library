import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from layers.StandardNorm import Normalize

from layers.Global_Attn import DAttentionBaseline
from layers.DeformableTST_Enc import Flatten_Head, TransformerMLPWithConv, TransformerMLP, LayerNormProxy

from timm.models.layers import DropPath

class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1))
        else:
            return x * self.weight.view(-1, 1)


class Stage(nn.Module):

    def __init__(self,
                 fmap_size,
                 dim_embed, depths,
                 drop_path_rate, layer_scale_value,
                 use_pe,
                 use_lpu,local_kernel_size,
                 expansion, drop, use_dwc_mlp,
                 heads, attn_drop, proj_drop,
                 stage_spec,
                 ksize, stride,
                 n_groups, offset_range_factor, no_off,
                 dwc_pe, fixed_pe, log_cpb,
                 ):


        super(Stage,self).__init__()
        fmap_size = fmap_size
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu


        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for d in range(2 * depths)]
        )


        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv1d(dim_embed, dim_embed, kernel_size=local_kernel_size, stride=1, padding=local_kernel_size//2,
                          groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )


        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()

        for i in range(depths):

            if stage_spec == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, ksize, log_cpb)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')


            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())


        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop, local_kernel_size) for _ in range(depths)
            ]
        )

        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )

    def forward(self, x):


        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec == 'D':

                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0

                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class DeformableTST(nn.Module):
    def __init__(self,
                 enc_in, rev,
                 stem_ratio,
                 down_ratio,
                 fmap_size,
                 dims, depths,
                 drop_path_rate, layer_scale_value,
                 use_pe,
                 use_lpu,
                 local_kernel_size,
                 expansion, drop, use_dwc_mlp,
                 heads, attn_drop, proj_drop,
                 stage_spec,
                 ksize, stride,
                 n_groups, offset_range_factor, no_off,
                 dwc_pe, fixed_pe, log_cpb,
                 seq_len, pred_len, head_dropout,
                 head_type, use_head_norm,
                 ):

        super(DeformableTST, self).__init__()

        self.rev = rev
        if self.rev:
            self.revin = Normalize(enc_in, affine=False, subtract_last=False) 


        patch_stride = stem_ratio
        patch_size = stem_ratio
        downsample_ratio = down_ratio
        self.downsample_layers = nn.ModuleList()
        if stem_ratio > 1:
            stem = nn.Sequential(
                nn.Conv1d(1, dims[0]//2, kernel_size = 3, stride = 2, padding= 3//2),
                LayerNormProxy(dims[0]//2),
                nn.GELU(),
                nn.Conv1d(dims[0]//2, dims[0], kernel_size=patch_size // 2, stride=patch_stride // 2),
                LayerNormProxy(dims[0])
            )
        else:
            stem = nn.Sequential(
                nn.Conv1d(1, dims[0], kernel_size = 1, stride = 1),
                LayerNormProxy(dims[0]))

        self.downsample_layers.append(stem)

        self.num_stage = len(depths)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                    LayerNormProxy(dims[i+1]),
                )
                self.downsample_layers.append(downsample_layer)



        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        fmap_size = fmap_size // stem_ratio

        self.stages = nn.ModuleList()
        for i in range(self.num_stage):
            self.stages.append(
                Stage(fmap_size,
                     dims[i], depths[i],
                     dpr[sum(depths[:i]):sum(depths[:i + 1])],
                     layer_scale_value[i],
                     use_pe[i],

                     use_lpu[i],
                     local_kernel_size[i],

                     expansion, drop, use_dwc_mlp[i],

                     heads[i], attn_drop, proj_drop,

                     stage_spec[i],

                     ksize[i], stride[i],
                     n_groups[i], offset_range_factor[i], no_off[i],
                     dwc_pe[i], fixed_pe[i], log_cpb[i],
                                )
            )
            fmap_size = fmap_size // down_ratio

        self.use_head_norm = use_head_norm
        if use_head_norm:
            self.head_norm = LayerNormProxy(dims[self.num_stage -1])
        if head_type == 'Flatten':
            L = seq_len
            S = 1
            for i in range(self.num_stage):
                if i == 0:
                    S = S * stem_ratio
                else:
                    S = S *down_ratio

            if L % S == 0:
                N = L // S
            else:
                N = L // S + 1
            self.head = Flatten_Head(individual=False, enc_in=enc_in, nf=N*dims[self.num_stage -1], target_window= pred_len, head_dropout= head_dropout)

        self.head_type = head_type

    def forward(self,x):

        if self.rev:
            x = self.revin(x,'norm')

        B,L,M = x.shape
        x = x.permute(0,2,1).reshape(B*M,L).unsqueeze(1)
        x = self.downsample_layers[0](x)


        for i in range(self.num_stage):
            x = self.stages[i](x)
            if i < (self.num_stage -1):
                x = self.downsample_layers[i+1](x)

        if self.head_type == 'Flatten':
            _,D,N = x.shape
            if self.use_head_norm:
                x = self.head_norm(x)
            x = x.reshape(B,M,D,N)
            x = self.head(x)
            x = x.permute(0,2,1)

        if self.rev:
            x = self.revin(x,'denorm')

        return x

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()

        enc_in = configs.enc_in
        rev = 1 # True 1 False 0

        down_ratio = 2

        if configs.seq_len == 96:
            stem_ratio = 1
        elif configs.seq_len == 384:
            stem_ratio = 4
        elif configs.seq_len == 768:
            stem_ratio = 8

        fmap_size = configs.seq_len

        dims = [16, 32, 64, 128]
        depths = [1, 1, 1, 1]

        drop_path_rate = 0.3
        layer_scale_value = [-1, -1, -1, -1]

        use_pe = [1, 1, 1, 1]
        use_lpu = [1, 1, 1, 1]
        local_kernel_size = [3, 3, 3, 3]

        expansion = 4
        drop = 0.0
        use_dwc_mlp = [1, 1, 1, 1] # use FFN with a DWConv; True 1 False 0

        heads = [4, 8, 16, 32]
        attn_drop = 0.0
        proj_drop = 0.0

        stage_spec = ['D','D','D','D']

        ksize = [9, 7, 5, 3]
        stride = [8, 8, 8, 8]
        n_groups = [2, 4, 8, 16]
        offset_range_factor = [-1, -1, -1, -1]
        no_off = [0, 0, 0, 0]

        dwc_pe = [0, 0, 0, 0]
        fixed_pe = [0, 0, 0, 0]
        log_cpb = [0, 0, 0, 0]

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        head_dropout = 0.1
        head_type = 'Flatten'
        use_head_norm = 1 # use final LN layer; True 1 False 0


        self.model = DeformableTST(
            enc_in, rev,
            stem_ratio,
            down_ratio,
            fmap_size,
            dims, depths,
            drop_path_rate, layer_scale_value,
            use_pe,
            use_lpu,
            local_kernel_size,
            expansion, drop, use_dwc_mlp,
            heads, attn_drop, proj_drop,
            stage_spec,
            ksize, stride,
            n_groups, offset_range_factor, no_off,
            dwc_pe, fixed_pe, log_cpb,
            seq_len, pred_len, head_dropout,
            head_type, use_head_norm,
        )

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None):

        return self.model(x)
