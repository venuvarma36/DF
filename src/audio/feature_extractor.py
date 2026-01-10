from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MultiResSpectrogram(nn.Module):
    """Multi-resolution mel-spectrogram using standard MelSpectrogram transforms"""
    def __init__(self, fft_list=(256, 512, 1024), hop_length=160, win_length=400, learnable=False, n_mels=64, sr=16000):
        super().__init__()
        # Use MelSpectrogram directly to avoid dimension issues
        self.mel_specs = nn.ModuleList([
            T.MelSpectrogram(
                sample_rate=sr,
                n_fft=f,
                hop_length=hop_length,
                win_length=min(win_length, f),
                n_mels=n_mels,
                power=2.0
            ) for f in fft_list
        ])

    def forward(self, wav):
        # wav: [B, T]
        outs = []
        for mel_spec in self.mel_specs:
            mel = mel_spec(wav)  # [B, n_mels, T]
            outs.append(mel.unsqueeze(1))  # [B, 1, n_mels, T]
        return torch.cat(outs, dim=1)  # [B, R, n_mels, T]
