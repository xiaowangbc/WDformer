import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RMS_Swi import RMSNorm
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DiffTransformerLayer, DiffAttention
from layers.Embed import DataEmbedding_wd
import numpy as np
import ptwt


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.wave_size = configs.wave_size
        # Embedding
        self.enc_embedding = DataEmbedding_wd(configs.wave_size,configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    DiffTransformerLayer(
                        DiffAttention(configs.d_model, configs.n_heads, False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, lambda_init=0.7 - 0.5 * math.exp(-0.3 * l)), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=RMSNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out)  # filter the covariates
        dec_out = self.wave_change(dec_out,self.wave_size)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def wave_change(self, x, wave_size):
        split_size = []
        num = 1
        for i in range(wave_size):
            num *= 2
            split_size.insert(0,self.pred_len//num)
        split_size.insert(0,self.pred_len//num)
        wave_list = torch.split(x,split_size_or_sections=split_size, dim=-1)
        dec_out = ptwt.waverec(wave_list, 'haar')
        return dec_out
