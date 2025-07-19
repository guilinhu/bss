import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
import math
from src.models.blocks.lstm_attention_block import LSTMAttentionBlock
from src.models.blocks.TDA_module import TDAModule
from src.models.blocks.triple_path_process import Triple_Path_Process


class Separator(nn.Module):
    def __init__(self, De, D, hidden_dim, K, M, n_head, dropout, N, max_spk):
        super(Separator, self).__init__()
        self.De = De  # Encoder output dim
        self.D = D  # Projected dim
        self.K = K  # Chunk size
        self.N = N

        self.proj = nn.Linear(De, D)

        self.intra_block = LSTMAttentionBlock(D, hidden_dim, n_head, dropout)
        self.inter_block = LSTMAttentionBlock(D, hidden_dim, n_head, dropout)
        self.TDA = TDAModule(D, num_layers=M, nhead=8, dropout=0.1, max_spk=max_spk)
        self.norm = nn.LayerNorm(D)

        self.triple = Triple_Path_Process(num_block=N, hidden_dim=hidden_dim, D=D, n_head=n_head, dropout=dropout)

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

        # B, T, D = E.shape
        # hop_size = math.ceil(K / 2)

        # num_chunks_raw = math.ceil(T / hop_size)
        # required_T_for_full_chunks = (num_chunks_raw - 1) * hop_size + K
        # padding_needed = max(0, required_T_for_full_chunks - T)

        # if padding_needed > 0:
        #     padding = torch.zeros(B, padding_needed, D, device=E.device, dtype=E.dtype)
        #     E_padded = torch.cat((E, padding), dim=1)
        # else:
        #     E_padded = E

        # T_padded = E_padded.shape[1]

        # S_per_batch = (T_padded - K) // hop_size + 1

        # U = E_padded.as_strided(
        #     size=(B, S_per_batch, K, D),
        #     stride=(
        #         E_padded.stride(0),
        #         hop_size * E_padded.stride(1),
        #         E_padded.stride(1),
        #         E_padded.stride(2),
        #     ),
        # )

        # # print(U.shape) [1, 84, 96, 128]
        # U = U.permute(0, 2, 1, 3).contiguous()

        # return U

    def _overlap_add(self, x, output_len):
        """
        Args:
            x: Tensor of shape (B, C, S, K, D)
        Returns:
            Tensor of shape (B, C, T', D)
        """
        # B, C, S, K, D = x.shape
        # hop = math.ceil(K / 2)

        # x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, D, S, K)
        # x = x.view(B * C * D, S, K)  # (B*C*D, S, K)
        # x = x.transpose(1, 2)  # (B*C*D, K, S)

        # x = x.reshape(B * C * D, K, S)
        # x = F.fold(x, output_size=(1, output_len), kernel_size=(1, K), stride=(1, hop))  # (B*C*D, 1, 1, T')

        # # Reshape back to (B, C, T', D)
        # x = x.view(B, C, D, output_len).transpose(2, 3)  # (B, C, T', D)

        # # Normalize by overlap count
        # ones = torch.ones(B * C * D, 1, S * K, device=x.device)
        # ones = ones.view(B * C * D, K, S)
        # overlap = F.fold(ones, output_size=(1, output_len), kernel_size=(1, K), stride=(1, hop))
        # overlap = overlap.view(B, C, D, output_len).transpose(2, 3)  # (B, C, T', D)

        # x = x / (overlap + 1e-8)  # Avoid divide by zero
        # return x

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
        attractor, tda_out = self.TDA(U_out, C, T)  # [B, C, K, S, D]
        # attractor is not sliced, length C_max+1
        # if C is given, tda_out only contains C streams. Otherwise, contains C_max streams

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

            # out = torch.zeros(B, C, T, D, device=triple_out.device)
            # norm = torch.zeros(B, C, T, 1, device=triple_out.device)

            # for i in range(S):
            #     start = i * math.ceil(self.K / 2)
            #     end = start + self.K
            #     out[:, :, start:end, :] += triple_out[:, :, :, i, :]
            #     norm[:, :, start:end, :] += 1

            # out = out / norm.clamp(min=1e-8)

            # # Final normalization + projection
            # out = self.norm(out)
            # out = self.out_linear(out)

            # # print("out {}".format(out.shape))

            # out_list.append(out)

        # # assert new_C == C
        # hop_size = math.ceil(self.K / 2)
        # T_prime = (S - 1) * hop_size + self.K

        # C = int(C.max().item())

        # out = torch.zeros(B, C, T_prime, D, device=triple_out.device)
        # norm = torch.zeros(B, C, T_prime, 1, device=triple_out.device)

        # for i in range(S):
        #     start = i * hop_size
        #     end = start + self.K
        #     out[:, :, start:end, :] += triple_out[:, :, :, i, :]
        #     norm[:, :, start:end, :] += 1

        # out = out / norm.clamp(min=1e-8)

        # Final normalization + projection
        # out = self.norm(out)
        # out = self.out_linear(out)

        return out_list, attractor  # [B, C, T', De]
