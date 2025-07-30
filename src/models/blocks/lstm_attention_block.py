import torch
import torch.nn as nn
from typing import List
from torch.nn import functional as F
from types import SimpleNamespace


class RelativePositionBias(nn.Module):
    """
    Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are
    invalid.

    We use smaller buckets for small absolute relative_position and larger buckets
    for larger absolute relative_positions. All relative positions >=max_distance
    map to the same bucket. All relative positions <=-max_distance map to the
    same bucket. This should allow for more graceful generalization to longer
    sequences than the model has been trained on.

    Args:
        bidirectional (bool): Whether the attention is bidirectional.
        num_buckets (int): Number of buckets.
        max_distance (int): Maximum distance for relative positions.
        num_heads (int): Number of attention heads.

    # REFRANCE: https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """

    def __init__(self, config):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = config.bidirectional
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.num_heads = config.num_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Translate relative position to a bucket number.

        Args:
            relative_position (torch.Tensor): Relative position tensor.
            bidirectional (bool): Whether the attention is bidirectional.
            num_buckets (int): Number of buckets.
            max_distance (int): Maximum distance for relative positions.

        Returns:
            torch.Tensor: Bucket number tensor.
        """
        ret = 0 * relative_position  # Initialized to zero to handle both positive and negative positions
        if bidirectional:
            num_buckets //= 2  # Halve the buckets for bidirectional case
            ret += (relative_position < 0).long() * num_buckets
            relative_position = relative_position.abs()
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Compute val_if_large with safe clamping within [0, num_buckets - 1]
        val_if_large = (
            max_exact
            + (
                torch.log(relative_position.float() / max_exact)
                / torch.log(torch.tensor(max_distance / max_exact, dtype=torch.float))
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.minimum(val_if_large, torch.tensor(num_buckets - 1, dtype=torch.long))

        # Combine small and large relative positions
        ret += torch.where(is_small, relative_position, val_if_large)

        return ret

    def compute_bias(self, qlen, klen):
        """
        Compute binned relative position bias.

        Args:
            qlen (int): Length of the query sequence.
            klen (int): Length of the key sequence.

        Returns:
            torch.Tensor: Relative position bias tensor.
        """
        context_position = torch.arange(qlen, dtype=torch.long, device=self.relative_attention_bias.weight.device)[
            :, None
        ]
        memory_position = torch.arange(klen, dtype=torch.long, device=self.relative_attention_bias.weight.device)[
            None, :
        ]
        relative_position = memory_position - context_position

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        ).to(self.relative_attention_bias.weight.device)

        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        return values

    def forward(self, qlen, klen):
        """
        Forward pass.

        Args:
            qlen (int): Length of the query sequence.
            klen (int): Length of the key sequence.

        Returns:
            torch.Tensor: Relative position bias tensor.
        """
        return self.compute_bias(qlen, klen)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_head=8, dropout=0.1, rel_pos_bias: RelativePositionBias = None):
        super().__init__()
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head

        if rel_pos_bias is not None:
            self.rel_pos_bias = rel_pos_bias
        else:
            config = SimpleNamespace(bidirectional=True, num_buckets=32, max_distance=128, num_heads=n_head)
            self.rel_pos_bias = RelativePositionBias(config)

        assert emb_dim % n_head == 0, "emb_dim must be divisible by n_head"

        # Q, K, V projections - simplified
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        """
        Args:
            x: (B * S, K, D) or (B * K, S, D)
        Returns:
            output: same shape as input
        """

        B_, X, D = x.shape

        # Apply Q, K, V
        q = self.q_proj(x)  # (B_, X, D)
        k = self.k_proj(x)  # (B_, X, D)
        v = self.v_proj(x)  # (B_, X, D)

        # Reshape for multi-head attention
        q = q.view(B_, X, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B_, X, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B_, X, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # add T5 pos encoding
        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias(X, X)  # shape: (1, n_head, X, X)
            # print("t5 bias dim is {}".format(bias.shape)) [1, 8, 167, 167]
            # print("attn score {}".format(attn_scores.shape)) [96, 4, 167, 167]
            attn_scores += bias

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B_, X, D)
        attn_output = self.out_proj(attn_output)

        output = attn_output
        return output


class LSTMAttentionBlock(nn.Module):
    def __init__(self, dim, hidden_dim, n_heads=8, dropout=0.1, rel_pos_bias: RelativePositionBias = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.bilstm = nn.LSTM(dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.linear_proj = nn.Linear(hidden_dim * 2, dim)

        self.attn = MultiHeadAttention(emb_dim=dim, n_head=n_heads, dropout=dropout, rel_pos_bias=rel_pos_bias)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B * S, K, D) or (B * K, S, D): (B_, X, D)
        Returns:
            (B * S, K, D) or (B * K, S, D): (B_, X, D)
        """

        # LSTM
        # print("x.shape is {}".format(x.shape)) [167, 96, 128]
        residual = x
        x = self.norm1(x)
        lstm_out, _ = self.bilstm(x)
        # print("lstm out {}".format(lstm_out.shape)) [384, 82, 512]
        # print("x {}".format(x.shape)) [384, 82, 128]
        x = x + self.linear_proj(lstm_out)  # (B_, X, D)

        # Attention
        residual = x
        x = self.norm2(x)
        x_attn = self.attn(x)
        x = residual + x_attn

        # FFN
        residual = x
        x = self.norm3(x)
        x_ffn = self.ffn(x)
        x = residual + x_ffn

        x = self.norm4(x)

        return x
