import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.blocks.lstm_attention_block import LSTMAttentionBlock


class Triple_Path_Block(nn.Module):
    def __init__(self, D, hidden_dim, n_head, dropout):
        super(Triple_Path_Block, self).__init__()

        self.intra_block = LSTMAttentionBlock(D, hidden_dim, n_head, dropout)
        self.inter_block = LSTMAttentionBlock(D, hidden_dim, n_head, dropout)
        self.inter_spk = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=n_head,
            dim_feedforward=D * 4,
            dropout=dropout,
            batch_first=True,
        )
        # self.inter_spk = nn.MultiheadAttention(embed_dim=D, num_heads=n_head, batch_first=True)
        self.norm = nn.LayerNorm(D)

    def forward(self, V):
        B, C, K, S, D = V.shape

        _input = V

        # intra
        V_intra = V.permute(0, 1, 3, 2, 4).contiguous()  # (B,C,S,K,D)
        V_intra = V_intra.view(B * C * S, K, D)
        V_intra = self.intra_block(V_intra)
        V_intra = V_intra.view(B, C, S, K, D).permute(0, 1, 3, 2, 4).contiguous()  #  (B, C, K, S, D)

        # inter
        V_inter = V_intra
        V_inter = V_inter.view(B * C * K, S, D)
        V_inter = self.inter_block(V_inter)
        V_inter = V_inter.view(B, C, K, S, D).contiguous()  #  (B, C, K, S, D)

        # inter speaker
        V_spk = V_inter.permute(0, 2, 3, 1, 4).contiguous()  # [B, K, S, C, D]
        # print("shape {}".format(V_spk.shape)) [16, 96, 84, 2, 128]
        V_spk = V_spk.view(B * K * S, C, D)
        # print("v spk dim {}".format(V_spk.shape)) [129024, 2, 128]
        V_spk = self.inter_spk(V_spk)
        V_spk = V_spk.view(B, K, S, C, D).permute(0, 3, 1, 2, 4).contiguous()  #  (B, C, K, S, D)

        # skip conn
        V_out = V_spk + _input
        V_out = self.norm(V_out)

        return V_out


class Triple_Path_Process(nn.Module):
    def __init__(self, num_block, hidden_dim, D, n_head, dropout):
        super(Triple_Path_Process, self).__init__()

        self.triple_path_list = nn.ModuleList(
            [Triple_Path_Block(D, hidden_dim, n_head, dropout) for i in range(num_block)]
        )

    def forward(self, V):
        outputs = []
        for i, layer in enumerate(self.triple_path_list):
            V = layer(V)
            outputs.append(V)

        return outputs
