import torch
import torch.nn as nn
import math

from ..functions import get_timestep_embedding

DEFAULT_NORMALIZER = nn.LayerNorm
DEFAULT_NONLINEARITY = nn.LeakyReLU(negative_slope=0.02, inplace=True)

Linear = nn.Linear




class Decoder5D_0204(nn.Module):
    """
    Variant without layer-wise weight sharing.
    Instead of [TemporalLayer(...)] * L, a fresh module is created in a
    for-loop for each layer and stored in a ModuleList.
    """
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, mid_features, num_temporal_layers):
        super(Decoder5D_0204, self).__init__()

        self.in_fc = Linear(in_features, mid_features, bias=False)

        # Create a fresh module per layer so each has independent parameters
        self.temp_layers = nn.ModuleList([
            TemporalLayer(mid_features, mid_features, mid_features)
            for _ in range(num_temporal_layers)
        ])

        self.out_norm = self.normalize(mid_features)
        self.out_fc = Linear(mid_features, in_features)

        self.t_proj = nn.Sequential(
            Linear(mid_features, mid_features),
            self.nonlinearity,
        )

        self.mid_features = mid_features
        self.input_dim = in_features

    def forward(self, x, t):
        # get_timestep_embedding is the project's existing helper
        t_emb = get_timestep_embedding(t, self.mid_features)
        t_emb = self.t_proj(t_emb)

        out = self.in_fc(x)

        # Apply layers sequentially (each layer has independent parameters)
        for layer in self.temp_layers:
            out = layer(out, t_emb=t_emb)

        out = self.out_fc(self.out_norm(out))
        return out


class TemporalLayer(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, out_features, temporal_features):
        super(TemporalLayer, self).__init__()
        self.norm1 = self.normalize(in_features)
        self.fc1 = Linear(in_features, out_features, bias=False)
        self.norm2 = self.normalize(out_features)
        self.fc2 = Linear(out_features, out_features, bias=False)
        self.enc = Linear(temporal_features, out_features)

        self.skip = nn.Identity() if in_features == out_features else Linear(in_features, out_features, bias=False)

    def forward(self, x, t_emb):
        out = self.fc1(self.nonlinearity(self.norm1(x)))
        out += self.enc(t_emb)
        out = self.fc2(self.nonlinearity(self.norm2(out)))
        skip = self.skip(x)
        return out + skip


