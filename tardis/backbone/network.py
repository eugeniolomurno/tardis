"""TabDiff denoising network (UniModMLP with EDM preconditioning).

Adapted from the official TabDiff implementation (https://github.com/MinkaiXu/TabDiff),
itself building on tab-ddpm and TabSyn. Parameter names are unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        d_bias = d_numerical + len(categories)
        self.register_buffer("category_offsets", torch.tensor([0] + list(categories[:-1])).cumsum(0))
        self.cat_weight = nn.Parameter(torch.empty(sum(categories), d_token))
        self.weight = nn.Parameter(torch.empty(d_numerical + 1, d_token))  # +1 for [CLS]
        self.bias = nn.Parameter(torch.empty(d_bias, d_token)) if bias else None
        for p in (self.cat_weight, self.weight, self.bias):
            if p is not None and p.numel() > 0:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, x_num, x_cat):
        x_num = torch.cat([torch.ones(len(x_cat), 1, device=x_cat.device), x_num], dim=1)
        x = self.weight[None] * x_num[:, :, None]
        ends = torch.cat([self.category_offsets[1:], torch.tensor([x_cat.shape[1]], device=x_cat.device)])
        for start, end in zip(self.category_offsets, ends):
            if start < end:
                x = torch.cat([x, x_cat[:, start:end].unsqueeze(1) @ self.cat_weight[start:end][None]], dim=1)
        if self.bias is not None:
            x = x + torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.W_q, self.W_k, self.W_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        b, n, d = x.shape
        return x.reshape(b, n, self.n_heads, d // self.n_heads).transpose(1, 2).reshape(b * self.n_heads, n, -1)

    def forward(self, x):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        b, n, d = q.shape
        att = F.softmax(self._reshape(q) @ self._reshape(k).transpose(1, 2) / math.sqrt(d // self.n_heads), dim=-1)
        x = (att @ self._reshape(v)).reshape(b, self.n_heads, n, -1).transpose(1, 2).reshape(b, n, d)
        return self.W_out(x) if self.W_out is not None else x


class Transformer(nn.Module):
    """Pre-norm transformer encoder (the first layer has no input normalisation)."""

    def __init__(self, n_layers, d_token, n_heads, d_ffn_factor):
        super().__init__()
        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                "attention": MultiheadAttention(d_token, n_heads),
                "linear0": nn.Linear(d_token, d_hidden),
                "linear1": nn.Linear(d_hidden, d_token),
                "norm1": nn.LayerNorm(d_token),
            })
            if i:
                layer["norm0"] = nn.LayerNorm(d_token)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            h = layer["norm0"](x) if "norm0" in layer else x
            x = x + layer["attention"](h)
            x = x + layer["linear1"](F.relu(layer["linear0"](layer["norm1"](x))))
        return x


class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super().__init__()
        self.d_numerical = d_numerical
        self.weight = nn.Parameter(torch.empty(d_numerical, d_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList(nn.Linear(d_token, d) for d in categories)
        for m in self.cat_recons:
            nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))

    def forward(self, h):
        x_num = (h[:, : self.d_numerical] * self.weight[None]).sum(-1)
        x_cat = [m(h[:, self.d_numerical + i]) for i, m in enumerate(self.cat_recons)]
        return x_num, x_cat


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000):
        super().__init__()
        self.num_channels, self.max_positions = num_channels, max_positions

    def forward(self, x):
        freqs = torch.arange(self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = (1 / self.max_positions) ** (freqs / (self.num_channels // 2))
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t), nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        return self.mlp(self.proj(x) + self.time_embed(emb))


class UniModMLP(nn.Module):
    """Tokenizer -> transformer -> MLP (time-conditioned) -> transformer -> detokenizer.

    ``categories`` already include the extra [MASK] state of each categorical column.
    Returns the numerical prediction and the unnormalised categorical logits.
    """

    def __init__(self, d_numerical, categories, num_layers, d_token, n_head=1, factor=4, bias=True, dim_t=512):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder = Transformer(num_layers, d_token, n_head, factor)
        self.mlp = MLPDiffusion(d_token * (d_numerical + len(categories)), dim_t=dim_t)
        self.decoder = Transformer(num_layers, d_token, n_head, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

    def forward(self, x_num, x_cat, timesteps):
        y = self.encoder(self.tokenizer(x_num, x_cat)[:, 1:])  # drop [CLS]
        h = self.decoder(self.mlp(y.reshape(len(y), -1), timesteps).reshape(*y.shape))
        x_num_pred, x_cat_pred = self.detokenizer(h)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if x_cat_pred else torch.zeros_like(x_cat)
        return x_num_pred, x_cat_pred


class Precond(nn.Module):
    """EDM preconditioning (Karras et al., 2022) of the numerical branch."""

    def __init__(self, denoise_fn, sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t, sigma):
        x_num, sigma = x_num.float(), sigma.float()
        # with per-column schedules the network is conditioned on the default rho = 7 schedule
        sigma_cond = (0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))).pow(7)
        s2 = sigma ** 2 + self.sigma_data ** 2
        c_skip, c_out, c_in = self.sigma_data ** 2 / s2, sigma * self.sigma_data / s2.sqrt(), 1 / s2.sqrt()
        F_x, x_cat_pred = self.denoise_fn_F(c_in * x_num, x_cat, (sigma_cond.log() / 4).flatten())
        return c_skip * x_num + c_out * F_x.float(), x_cat_pred


class Model(nn.Module):
    def __init__(self, denoise_fn, sigma_data=1.0):
        super().__init__()
        self.denoise_fn_D = Precond(denoise_fn, sigma_data)

    def forward(self, x_num, x_cat, t, sigma):
        return self.denoise_fn_D(x_num, x_cat, t, sigma)
