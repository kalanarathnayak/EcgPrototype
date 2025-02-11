import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerECG(nn.Module):
    def __init__(self, input_size=180, d_model=256, nhead=16,
                 num_transformer_layers=3, dropout=0.15):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x.squeeze(-1)

class ECGDataset(Dataset):
    def __init__(self, beats, augment=False):
        self.data = torch.FloatTensor(beats)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        beat = self.data[idx].clone()
        if self.augment:
            noise = torch.randn_like(beat) * 0.01
            beat += noise
            shift = torch.randint(-3, 4, (1,)).item()
            if shift != 0:
                beat = torch.roll(beat, shifts=shift)
        return beat