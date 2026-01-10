from __future__ import annotations
import os
import torch
import torchaudio
# Lazy import transformers to avoid slow initialization
# from transformers import Wav2Vec2Model, WhisperModel
from .feature_extractor import MultiResSpectrogram
from .model_specRNet import SpecRNetLite


def load_audio_model(cfg: dict, device: torch.device, ckpt_dir: str):
    params = cfg["audio"]["params"]
    feat_cfg = cfg["audio"]["feature"]

    # Force single-branch to match trained checkpoint (audio_kaggle_best.pt)
    in_branches = params.get("in_branches", 1)
    feat_ffts = feat_cfg.get("multi_res_ffts", [512])
    feat_cfg["multi_res_ffts"] = feat_ffts[:in_branches]
    model = SpecRNetLite(
        in_branches=in_branches,
        channels=params.get("channels", [16, 32, 64]),
        gru_hidden=params.get("gru_hidden", 128),
        embed_dim=params.get("embed_dim", 128),
        dropout=params.get("dropout", 0.2),
    ).to(device)

    # Load checkpoint if present (prefer the trained Kaggle checkpoint)
    ckpt_candidates = [
        os.path.join(ckpt_dir, "audio_kaggle_best.pt"),
        os.path.join(ckpt_dir, "audio_best.pt"),
    ]
    for ckpt_path in ckpt_candidates:
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            break
    model.eval()
    return model


def preprocess_audio(path: str, cfg: dict, device: torch.device):
    # Use soundfile backend to avoid torchcodec dependency
    import soundfile as sf
    wav_np, sr = sf.read(path, dtype='float32')
    if len(wav_np.shape) > 1:  # stereo to mono
        wav_np = wav_np.mean(axis=1)
    wav = torch.from_numpy(wav_np).unsqueeze(0)  # [1, T]
    
    target_sr = cfg["audio"]["feature"]["sample_rate"]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)


def extract_features(wav: torch.Tensor, cfg: dict):
    feat_cfg = cfg["audio"]["feature"]
    mr = MultiResSpectrogram(
        fft_list=feat_cfg.get("multi_res_ffts", [256, 512, 1024]),
        hop_length=feat_cfg.get("hop_length", 160),
        win_length=feat_cfg.get("win_length", 400),
        learnable=feat_cfg.get("learnable_filterbanks", True),
        n_mels=feat_cfg.get("n_mels", 64),
        sr=feat_cfg.get("sample_rate", 16000),
    ).to(wav.device)
    spec = mr(wav)  # [B, R, F, T]
    return spec


def infer_audio(model, audio_path: str, cfg: dict, device: torch.device):
    wav = preprocess_audio(audio_path, cfg, device)
    spec = extract_features(wav, cfg)
    with torch.no_grad():
        score = model(spec)
    return score.item()
