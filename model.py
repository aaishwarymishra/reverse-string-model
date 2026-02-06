from torch import nn
import torch
from ignite.metrics import Accuracy, Loss


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")
        if max_len <= 0:
            raise ValueError("max_len must be greater than 0")
        self.embed_dim = embed_dim
        self.max_len = max_len
        pe = torch.zeros(max_len, embed_dim)
        depth = self.embed_dim // 2
        position = torch.arange(0, self.max_len).unsqueeze(1)
        depths = torch.arange(0, depth).unsqueeze(0)
        angle_rates = torch.divide(1, torch.pow(10000, torch.divide(depths, depth)))
        angle_rads = torch.multiply(position, angle_rates)
        pe[:, 0::2] = torch.sin(angle_rads)
        pe[:, 1::2] = torch.cos(angle_rads)
        self.register_buffer("pos_encoding", pe)

    def forward(self, x):
        return torch.add(x, self.pos_encoding[: x.shape[-2]])


class FeedForward(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(hidden, intermediate)
        self.linear2 = nn.Linear(intermediate, hidden)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden)
        self.seq = nn.Sequential(
            self.linear1, self.gelu, self.dropout, self.linear2, self.dropout
        )

    def forward(self, x):
        out = self.seq(x)
        out = torch.add(x, out)
        out = self.layer_norm(out)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        out, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)
        out = torch.add(x, out)
        out = self.layer_norm(out)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, context, key_padding_mask=None):
        out, _ = self.cross_attention(
            x, context, context, key_padding_mask=key_padding_mask
        )
        out = torch.add(x, out)
        out = self.layer_norm(out)
        return out


class CausalAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self.causal_attention = nn.MultiheadAttention(
            embed_dim, heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        out, _ = self.causal_attention(
            x, x, x, is_causal=True, key_padding_mask=key_padding_mask
        )
        out = torch.add(x, out)
        out = self.layer_norm(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, intermediate: int):
        super().__init__()
        self.causal_attention = CausalAttentionBlock(embed_dim, heads)
        self.feed_forward = FeedForward(embed_dim, intermediate)

    def forward(self, x, key_padding_mask=None):
        out = self.causal_attention(x, key_padding_mask=key_padding_mask)
        out = self.feed_forward(out)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        intermediate: int,
        heads: int,
        vocab_size: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.intermediate = intermediate
        self.num_layers = num_layers
        self.layer_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, heads, intermediate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, key_padding_mask=None):
        out = self.layer_embedding(x)
        out = torch.multiply(
            out,
            torch.sqrt(
                torch.tensor(self.embed_dim, dtype=out.dtype, device=out.device)
            ),
        )
        out = self.positional_encoding(out)
        out = self.dropout(out)
        for decoder_block in self.decoder_blocks:
            out = decoder_block(out, key_padding_mask=key_padding_mask)
        return out


class ReverseStringModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        intermediate: int,
        heads: int,
        vocab_size: int,
        pad_idx: int | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.intermediate = intermediate
        self.heads = heads
        self.vocab_size = vocab_size
        self.decoder = Decoder(num_layers, embed_dim, intermediate, heads, vocab_size)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, x):
        key_padding_mask = x == self.pad_idx if self.pad_idx is not None else None
        out = self.decoder(x, key_padding_mask=key_padding_mask)
        out = self.linear(out)
        return out
