"""Small CPU-friendly sequence model for offline, time-ordered research."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def sequence_endpoints(original_positions, partition, sequence_length, stride):
    """Only endpoints with a contiguous window wholly inside this partition."""
    if sequence_length < 1 or stride < 1:
        raise ValueError("序列长度和采样间隔必须为正")
    first = partition.start + sequence_length - 1
    endpoints = np.arange(first, partition.stop, stride, dtype=np.int64)
    if not len(endpoints):
        return endpoints
    discontinuities = np.r_[0, np.cumsum(np.diff(original_positions) != 1)]
    return endpoints[discontinuities[endpoints] == discontinuities[endpoints - sequence_length + 1]]


class Sequences(Dataset):
    def __init__(self, x, y_class, y_regression, endpoints, sequence_length):
        self.x = x
        self.y_class = y_class
        self.y_regression = y_regression
        self.endpoints = endpoints
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        end = self.endpoints[index]
        return (torch.from_numpy(self.x[end - self.sequence_length + 1:end + 1]),
                int(self.y_class[end]), torch.from_numpy(self.y_regression[end]))


class PriceTransformer(nn.Module):
    def __init__(self, n_features, sequence_length=120, d_model=32, nhead=4, layers=2, dropout=.1):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.direction = nn.Linear(d_model, 3)
        self.regression = nn.Linear(d_model, 2)

    def forward(self, x):
        features = self.norm(self.encoder(self.input(x) + self.position[:, :x.shape[1]])[:, -1])
        return self.direction(features), self.regression(features)
