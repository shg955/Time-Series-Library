import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.total_features = configs.enc_in + len(configs.mark_in)
        self.pe_activation = configs.pe_weight_activation

        # positional embedding
        if configs.use_separate:
            if configs.position_embedding_emb:
                self.position_embedding_em = nn.Parameter(torch.randn(self.total_features, configs.d_model)) # (N, D)
            if configs.position_embedding_proj:
                self.position_embedding_pr = nn.Parameter(torch.randn(self.total_features, configs.d_model)) # (N, D)
        else:
            self.position_embedding_sh = nn.Parameter(torch.randn(self.total_features, configs.d_model)) # (N, D)

        # positional embedding weight
        if configs.position_embedding_weight:
            if configs.use_separate:
                if configs.each_weight:
                    self.pe_em_weight = nn.Parameter(torch.full((self.total_features, 1), configs.pe_weight)) # (total_features, 1)
                    # torch.full(size, value) : 지정한 크기의 텐서 만들고, 모든 원소를 같은 값으로 채움
                else:
                    self.pe_em_weight = nn.Parameter(torch.tensor(configs.pe_weight))

        # positional encoding
        if configs.position_encoding_emb or configs.position_encoding_proj:
            self.position_encoding = PositionalEmbedding(configs.d_model, self.total_features)

        # Embedding
        if configs.channelwise_embedding:
            self.enc_embedding = nn.ModuleList([DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
                                                for _ in range(self.total_features)
                                                ])
        else:
            self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'paper']:
            # 만약 recon이였다면
            if self.configs.reconstruction:
                #  seq_len + pred_len만큼 projection
                output_dim = configs.seq_len + configs.pred_len
            # recon이 아닌 기존 실험이라면
            else:
                # pred_len만큼 projection
                output_dim = configs.pred_len
            # channel-wise projection
            if self.configs.channelwise_projection:
                self.projection = nn.ModuleList([
                    nn.Linear(configs.d_model, output_dim, bias=True) for _ in range(configs.enc_in)
                ])
            else:
                self.projection = nn.Linear(configs.d_model, output_dim, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # x_enc.shape (batch size, seq_len, feature)
        _, _, N = x_enc.shape

        # Embedding
        if self.configs.channelwise_embedding:
            x_all = torch.cat([x_enc, x_mark_enc], dim=-1)
            embeds = []
            for i in range(self.total_features):
                x_i = x_all[:, :, i:i+1]
                # x_i.shape (batch size, seq, 1)
                emb_i = self.enc_embedding[i](x_i, None)
                # emb_i.shape (batch size, 1, d_model)
                embeds.append(emb_i)
            # len(embeds) feature+date
            enc_out = torch.cat(embeds, dim=1)
            # enc_out.shape (batch size, feature+date, d_model)
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out.shape (batch_size, feature+date, d_model)

        before_position_embedding_emb = enc_out.clone().detach()
        before_position_encoding_emb = enc_out.clone().detach()

        # position embedding embedding 이후
        if self.configs.position_embedding_emb:
            # position_embedding.unsqueeze(0).shape (1, feature+date, d_model)
            if self.configs.use_separate:
                pe = self.position_embedding_em
                if self.configs.position_embedding_weight:
                    pe = self.pe_activation(self.pe_em_weight) * pe  # scalar weight
            else:
                pe = self.position_embedding_sh
            enc_out = enc_out + pe.unsqueeze(0)
            # position_embedding.shape (feature+date, d_model)
            # enc_out.shape (batch_size, feature+date, d_model)
            after_position_embedding_emb = enc_out.clone().detach()
        else:
            after_position_embedding_emb = None

        # position embedding projection 이전
        if self.configs.position_encoding_emb:
            enc_out = enc_out + self.position_encoding(enc_out)
            after_position_encoding_emb = enc_out.clone().detach()
        else:
            after_position_encoding_emb = None
            
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # enc_out.shape (batch size, feature+date, d_model)

        before_position_embedding_proj = enc_out.clone().detach()
        before_position_encoding_proj = enc_out.clone().detach()

        # position embedding projection 이전
        if self.configs.position_embedding_proj:
            # enc_out.shape (batch_size, feature+date, d_model)
            # position_embedding.unsqueeze(0).shape (1, feature+date, d_model)
            if self.configs.use_separate:
                enc_out = enc_out + self.position_embedding_pr.unsqueeze(0)
            else:
                enc_out = enc_out + self.position_embedding_sh.unsqueeze(0)
            # enc_out.shape (batch_size, feature+date, d_model)
            after_position_embedding_proj = enc_out.clone().detach()
        else:   
            after_position_embedding_proj = None

        # position encoding projection 이전
        if self.configs.position_encoding_proj:
            enc_out = enc_out + self.position_encoding(enc_out)
            after_position_encoding_proj = enc_out.clone().detach()
        else:
            after_position_encoding_proj = None

        if self.configs.channelwise_projection:
            outputs = []
            for i in range(N):
                enc_out_i = enc_out[:, i, :] # [B, D]
                proj = self.projection[i](enc_out_i)  # [B, output_dim]
                outputs.append(proj.unsqueeze(-1))
            # len(outputs) : N(feature(dateX))
            dec_out = torch.cat(outputs, dim=-1)
            # dec_out.shape (batch size, (seq_len) + pred_len, feature(dateX))
        else:
            # 위에서 N -> date 정보를 제외한 feature수로 지정해줬었음
            dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
            # dec_out.shape (batch size, (seq_len) + pred_len, feature(dateX))

        # recon이면 seq_len + pred_len 만큼
        if self.configs.reconstruction:
            length = self.seq_len + self.pred_len
        # recon이 아니면 pred_len 만큼
        else :
            length = self.pred_len

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, length, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, length, 1))

        if self.configs.is_tsne_emb:
            if self.configs.position_encoding_emb or self.configs.position_encoding_proj:
                embeddings = (before_position_encoding_emb,
                        after_position_encoding_emb,
                        before_position_encoding_proj,
                        after_position_encoding_proj)
            else:
                embeddings = (before_position_embedding_emb,
                            after_position_embedding_emb,
                            before_position_embedding_proj,
                            after_position_embedding_proj)
        else:
            embeddings = None
        
        return dec_out, attns, embeddings

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'paper']:
            dec_out, attns, embeddings = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.configs.is_tsne_emb:
                return embeddings
            return dec_out, attns # [B, L, D]
        return None