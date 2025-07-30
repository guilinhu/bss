import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
import math
from src.models.blocks.lstm_attention_block import LSTMAttentionBlock
from src.models.blocks.TDA_module import TDAModule
from src.models.blocks.triple_path_process import Triple_Path_Process
from src.models.blocks.lstm_attention_block import RelativePositionBias
from types import SimpleNamespace


class Separator(nn.Module):
    def __init__(self, De, D, hidden_dim, K, M, n_head, dropout, N, max_spk):
        super(Separator, self).__init__()
        self.De = De  # Encoder output dim
        self.D = D  # Projected dim
        self.K = K  # Chunk size
        self.N = N

        self.proj = nn.Linear(De, D)

        config = SimpleNamespace(bidirectional=True, num_buckets=32, max_distance=128, num_heads=n_head)
        self.rel_pos_bias_intra = RelativePositionBias(config)
        self.rel_pos_bias_inter = RelativePositionBias(config)

        self.intra_block = LSTMAttentionBlock(D, hidden_dim, n_head, dropout, self.rel_pos_bias_intra)
        self.inter_block = LSTMAttentionBlock(D, hidden_dim, n_head, dropout, self.rel_pos_bias_inter)
        self.TDA = TDAModule(D, num_layers=M, nhead=8, dropout=0.1, max_spk=max_spk)
        self.norm = nn.LayerNorm(D)

        self.triple = Triple_Path_Process(
            num_block=N,
            hidden_dim=hidden_dim,
            D=D,
            n_head=n_head,
            dropout=dropout,
            rel_pos_bias_intra=self.rel_pos_bias_intra,
            rel_pos_bias_inter=self.rel_pos_bias_inter,
        )

        self.out_linear = nn.Linear(D, De)

    def _overlap_chunking(self, E, K):
        """
        Args:
            x: Tensor of shape (B, T', D)
            chunk_size: K, window size
        Returns:
            Tensor of shape (B, K, S, D)
        """
        B, T, D = E.shape
        hop = math.ceil(K / 2)

        # Transpose to (B, D, T) for unfold
        E = E.transpose(1, 2).unsqueeze(2)  # (B, D, T)
        unfold = nn.Unfold(kernel_size=(1, K), stride=(1, hop))
        E_unfold = unfold(E)

        B, DK, S = E_unfold.shape
        E_unfold = E_unfold.view(B, D, K, S)
        E_unfold = E_unfold.permute(0, 2, 3, 1)

        return E_unfold

    def _overlap_add(self, x, output_len):
        """
        Args:
            x: Tensor of shape (B, C, S, K, D)
        Returns:
            Tensor of shape (B, C, T', D)
        """
        B, C, K, S, D = x.shape
        stride = math.ceil(K / 2)

        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B * C, D * K, S)

        fold = nn.Fold(output_size=(1, output_len), kernel_size=(1, K), stride=(1, stride))

        x_fold = fold(x)  # [B*C, D, 1, T]
        x_out = x_fold.squeeze(2).permute(0, 2, 1)  # [B*C, T, D]

        unfold = nn.Unfold(kernel_size=(1, K), stride=(1, stride))

        ones = torch.ones_like(x_fold)  # [B*C, D, 1, T]
        norm = fold(unfold(ones))  # [B*C, D, 1, T]
        norm = norm.squeeze(2).permute(0, 2, 1)  # [B*C, T, D]

        x_out = x_out / norm.clamp(min=1e-8)

        x_out = x_out.view(B, C, output_len, D)

        # print("x out shape {}".format(x_out.shape)) [1, 3, 4001, 128]

        return x_out

    def forward(self, E, C):
        """
        E: encoder output: (B, T', D)
        C: gt number of speaker, could be None during inference
        Returns: chunked representation U: (B, K, S, D)
        """
        # [B, T', De] [1, 8001, 256]
        B, T, De = E.size()

        # linear
        E = self.proj(E)  # (B, T', D)
        # print("e is {}".format(E.shape))

        # chunking
        U = self._overlap_chunking(E, self.K)  # (B, K, S, D))
        # print("u shape {}".format(U.shape)) [1, 96, 82, 128]

        # ====dual path processing====
        # intra
        U_intra = U.permute(0, 2, 1, 3).contiguous()  # (B, S, K, D)
        # print("u intra.shape in separator is {}".format(U_intra.shape)) [1, 166, 128, 96]
        B, S, K, D = U_intra.shape
        U_intra = U_intra.view(B * S, K, D)
        U_intra = self.intra_block(U_intra)  # (B * S, K, D)
        U_intra = U_intra.view(B, S, K, D).permute(0, 2, 1, 3).contiguous()  # (B, K, S, D)

        # inter
        U_inter = U_intra.permute(0, 1, 2, 3).contiguous()  # (B, K, S, D)
        B, K, S, D = U_inter.shape
        U_inter = U_inter.view(B * K, S, D)
        U_inter = self.inter_block(U_inter)  # (B * K, S, D)
        U_inter = U_inter.view(B, K, S, D)

        # Residual + Norm
        U_out = self.norm(U_inter + U)  # (B, K, S, D)
        # ====dual path processing done====

        # print("u out {}".format(U_out.shape))  # [1, 96, 167, 128] [B, K, S, D]

        # TDA
        # attractor_logits length C+1 or C_max+1, tda_out C or C_max
        attractor_logits, tda_out = self.TDA(U_out, C, T)  # [B, C, K, S, D]

        # ====triple path processing====
        triple_out_list = self.triple(tda_out)
        # triple_out_list = [tda_out for _ in range(self.N)]

        # overlap add
        B, new_C, K, S, D = triple_out_list[-1].shape

        # print("triple {}".format(triple_out.shape)) [1, C, 96, 82, 128]

        out_list = []

        for triple_out in triple_out_list:
            triple_out = self._overlap_add(triple_out, T)
            triple_out = self.norm(triple_out)
            triple_out = self.out_linear(triple_out)

            out_list.append(triple_out)

        return out_list, attractor_logits  # [B, C, T', De]
