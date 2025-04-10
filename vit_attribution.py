
import torch
import torch.nn as nn
from copy import deepcopy


class LinearMapAffine(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return x * self.weight + self.bias

class LinearMap(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return x @ self.weight + self.bias


class MultiHeadAttentionWithoutTransformation(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()

        attn_copied = deepcopy(attn)
        embed_dim = attn_copied.embed_dim

        self.embed_dim = embed_dim

        attn_copied.in_proj_weight = nn.Parameter(torch.cat([torch.eye(embed_dim)] * 3))
        attn_copied.in_proj_bias = nn.Parameter(torch.zeros_like(attn.in_proj_bias))

        attn_copied.out_proj.weight = nn.Parameter(torch.eye(embed_dim))
        attn_copied.out_proj.bias = nn.Parameter(torch.zeros_like(attn.out_proj.bias))

        self.attn = attn_copied

    def forward(self, x):
        embed_dim = self.embed_dim
        query = x[:, :, :embed_dim]
        key = x[:, :, embed_dim:2*embed_dim]
        value = x[:, :, 2*embed_dim:]

        out, _ = self.attn(
            query=query.detach(),
            key=key.detach(),
            value=value,
            need_weights=False
        )
        return out

class Summation(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=-1)

class SummationPositionEmbed(torch.nn.Module):
    def __init__(self, pos_embedding):
        super().__init__()
        self.pos_embedding = pos_embedding

    def forward(self, x):
        return x + self.pos_embedding

class LayerNormStandardizeStep(nn.Module):
    def __init__(self, eps):

        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)

        var = ( (x-mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps) ** 0.5
        std = std.detach()
        y = (x-mean) / std

        return y