import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.common.film import FiLM4D


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, skip_self_attn=False):
        super().__init__()
        self.skip_self_attn = skip_self_attn
        if not skip_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask=None):
        x = query
        if not self.skip_self_attn:
            # TODO: understand this
            _x = self.self_attn(x, x, x, attn_mask=mask)[0]
            x = x + self.dropout(_x)
            x = self.norm1(x)

        _x = self.cross_attn(x, context, context)[0]
        x = x + self.dropout(_x)
        x = self.norm2(x)

        _x = self.ffn(x)
        x = x + self.dropout(_x)
        x = self.norm3(x)

        return x


class TDAModule(nn.Module):
    def __init__(self, D, num_layers=2, nhead=8, dropout=0.1, max_spk=5):
        super().__init__()
        self.D = D
        # self.query_bank = nn.Parameter(torch.randn(max_spk + 1, D))
        self.query_bank = nn.Embedding(max_spk + 1, D)

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(D, nhead, dropout, skip_self_attn=(i == 0)) for i in range(num_layers)]
        )

        self.layer_norm = nn.LayerNorm(D)

        self.film = FiLM4D(input_channels=D, embedding_channels=D)

        self.existence_linear = nn.Linear(D, 1)  # Maps D dimensional attractor to a single logit

    def _overlap_add(self, x, original_length):
        """
        Args:
            x: Tensor of shape (B, K, S, D)
        Returns:
            Tensor of shape (B, T', D)
        """
        B, K, S, D = x.shape
        stride = math.ceil(K / 2)

        x = x.permute(0, 3, 1, 2).contiguous().view(B, D * K, S)

        fold = nn.Fold(output_size=(1, original_length), kernel_size=(1, K), stride=(1, stride))

        x_fold = fold(x)  # [B, D, 1, T]
        x_out = x_fold.squeeze(2).permute(0, 2, 1)  # [B, T, D]

        unfold = nn.Unfold(kernel_size=(1, K), stride=(1, stride))

        ones = torch.ones_like(x_fold)  # [B, D, 1, T]
        norm = fold(unfold(ones))  # [B, D, 1, T]
        norm = norm.squeeze(2).permute(0, 2, 1)  # [B, T, D]

        x_out = x_out / norm.clamp(min=1e-8)

        return x_out

    def forward(self, U_out, C=None, T=4001):
        """
        Args:
            U_out: Dual-path output U'' of shape [B, K, S, D]

        Returns:
            A: Attractors, shape [B, C+1, D]
            V0: FiLM output, shape [B, C, K, S, D]
        """
        # [1, 96, 167, 128] [B, K, S, D]
        B, K, S, D = U_out.shape

        if C is not None:
            num_queries = int(C.max().item()) + 1
        else:
            num_queries = self.query_bank.num_embeddings

        idx = torch.arange(num_queries, device=U_out.device, dtype=torch.long)
        batch_idx = idx.unsqueeze(0).expand(B, -1)

        # lookup embeddings → [num_queries, D]
        queries = self.query_bank(batch_idx)
        # print("query {}".format(queries.shape))

        # mask for causal self-attention (C+1 x C+1), upper triangular
        causal_mask = torch.triu(
            torch.ones(queries.size(1), queries.size(1), device=U_out.device, dtype=torch.bool),
            diagonal=1,
        )

        for i, layer in enumerate(self.decoder_layers):
            queries = layer(
                queries,
                context,
                mask=causal_mask if i != 0 else None,  # first layer skips self attn
            )

        attractor_logits = self.existence_linear(queries).squeeze(-1)

        if C is not None:
            # training
            # print(C)
            attractors = queries[:, :C, :]
        else:
            # TODO: double check
            # might want to use a fixed C_max or select based on existence logits
            attractors = queries[:, :-1, :]  # drop dummy last attractor

        # print("attractor {}".format(attractors.shape)) [1, 3, 128] [B, C, D]
        # print("u out {}".format(U_out.shape)) [1, 96, 167, 128] [B, K, S, D]

        # film
        V0 = self.film(U_out.unsqueeze(1).expand(-1, attractors.size(1), -1, -1, -1), attractors)

        return attractor_logits, V0
