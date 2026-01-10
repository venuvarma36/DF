from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SpecRNetLite(nn.Module):
    def __init__(self, in_branches=3, channels=(16, 32, 64), gru_hidden=128, embed_dim=128, dropout=0.2):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = ConvBlock(in_branches, c1, k=3, s=1, p=1)
        self.layer2 = ConvBlock(c1, c2, k=3, s=2, p=1)
        self.layer3 = ConvBlock(c2, c3, k=3, s=2, p=1)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.gru = nn.GRU(input_size=c3, hidden_size=gru_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        # x: [B, R, F, T]
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)  # [B, C, T', 1]
        x = x.squeeze(-1).transpose(1, 2)  # [B, T', C]
        y, _ = self.gru(x)  # [B, T', 2H]
        y = y.mean(dim=1)  # temporal pooling
        logit = self.head(y).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob
