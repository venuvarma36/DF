from __future__ import annotations
import os
import shutil
from pathlib import Path
import torch
import torchaudio
import subprocess
import tempfile
import numpy as np
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


def _resolve_ffmpeg_path() -> str | None:
    """Find ffmpeg executable (system PATH or winget cache)."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # Search the winget package cache for ffmpeg.exe (covers Gyan.FFmpeg install)
    winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        try:
            found = next(winget_root.rglob("ffmpeg.exe"))
            return str(found)
        except StopIteration:
            pass
    return None


def preprocess_audio(path: str, cfg: dict, device: torch.device):
    """Load audio from any format (WAV, MP3, M4A, FLAC, OGG, AAC, etc.)."""
    import soundfile as sf
    
    target_sr = cfg["audio"]["feature"]["sample_rate"]
    file_ext = Path(path).suffix.lower()
    
    # Try method 1: soundfile (supports WAV, FLAC, OGG)
    try:
        wav_np, sr = sf.read(path, dtype='float32')
        if len(wav_np.shape) > 1:  # stereo to mono
            wav_np = wav_np.mean(axis=1)
        wav = torch.from_numpy(wav_np).unsqueeze(0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.to(device)
    except Exception:
        pass
    
    last_error = None
    # Try method 2: pydub (supports MP3, M4A, AAC, OGG, WMA, and many others)
    try:
        from pydub import AudioSegment
        # Ensure pydub knows where ffmpeg is (covers winget installs not on PATH)
        if not shutil.which("ffmpeg"):
            ffmpeg_guess = _resolve_ffmpeg_path()
            if ffmpeg_guess:
                AudioSegment.converter = ffmpeg_guess
                AudioSegment.ffmpeg = ffmpeg_guess
                AudioSegment.ffprobe = ffmpeg_guess
        
        # Auto-detect format from file extension or let pydub figure it out
        file_format = file_ext.lstrip('.')
        if file_format in ['m4a', 'aac']:
            file_format = 'm4a'
        elif file_format in ['ogg', 'oga']:
            file_format = 'ogg'
        elif file_format in ['wma']:
            file_format = 'wma'
        elif file_format in ['flac']:
            file_format = 'flac'
        
        # Load with pydub
        audio = AudioSegment.from_file(path, format=file_format if file_format else None)
        
        # Resample to target sample rate
        audio = audio.set_frame_rate(target_sr)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Convert stereo to mono
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        elif audio.channels > 2:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)
        
        # Normalize to [-1, 1] range
        samples = samples / (2**15)
        
        wav = torch.from_numpy(samples).unsqueeze(0)
        return wav.to(device)
    except Exception as e:
        last_error = e
    
    # Try method 3: scipy wavfile (basic WAV support)
    try:
        from scipy.io import wavfile
        sr, wav_np = wavfile.read(path)
        wav_np = wav_np.astype(np.float32)
        
        # Normalize based on dtype
        if wav_np.max() > 1:
            wav_np = wav_np / (2**15)
        
        if len(wav_np.shape) > 1:  # stereo to mono
            wav_np = wav_np.mean(axis=1)
        
        wav = torch.from_numpy(wav_np).unsqueeze(0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.to(device)
    except Exception:
        pass
    
    # All methods failed
    extra_hint = ""
    if last_error is not None:
        extra_hint = f"\nLast decoder error: {last_error}"
    raise RuntimeError(
        f"Failed to load audio file: {path}\n"
        f"Supported formats: WAV, MP3, M4A, AAC, FLAC, OGG, WMA, and others\n"
        f"Ensure pydub is installed: pip install pydub\n"
        f"Ensure ffmpeg is installed and on PATH (or run app in a shell where ffmpeg -version works)." + extra_hint
    )


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
