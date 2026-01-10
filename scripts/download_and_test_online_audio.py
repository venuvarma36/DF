"""
Download a few external audio samples (real + synthetic/tts-as-fake) into samples/online, then score them with the trained model.
"""
import sys
import os
from pathlib import Path
import requests
import subprocess
import torch
import torchaudio
from tqdm import tqdm

# Ensure repo root on path
sys.path.insert(0, r"C:\DeepFake_detection")
from src.audio.model_specRNet import SpecRNetLite

CONFIG = {
    "sample_rate": 16000,
    "n_mels": 80,
    "n_fft": 400,
    "hop_length": 160,
    "duration_sec": 5.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CHECKPOINT = Path(r"C:\DeepFake_detection\checkpoints\audio_kaggle_best.pt")
BASE_DIR = Path(r"C:\DeepFake_detection\samples\online")
REAL_DIR = BASE_DIR / "real"
FAKE_DIR = BASE_DIR / "fake"
REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

REAL_URLS = {
    "real1.wav": "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_jackson_0.wav",
    "real2.wav": "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/1_nicolas_1.wav",
}

FAKE_URLS = {
    # These may fail; fallback to TTS synthesis below
    "fake1.wav": "http://festvox.org/examples/eng_female.wav",
    "fake2.wav": "http://festvox.org/examples/eng_male.wav",
}

def download(url_map, out_dir):
    paths = []
    for name, url in url_map.items():
        dest = out_dir / name
        try:
            if dest.exists():
                paths.append(dest)
                continue
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            paths.append(dest)
            print(f"✓ downloaded {name}")
        except Exception as e:
            print(f"✗ failed {name}: {e}")
    return paths


def synthesize_tts(text, dest):
    try:
        from gtts import gTTS
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gtts"])
        from gtts import gTTS
    try:
        tts = gTTS(text=text, lang="en")
        mp3_path = str(dest.with_suffix(".mp3"))
        tts.save(mp3_path)
        # Convert mp3 to wav via torchaudio load+save
        wav, sr = torchaudio.load(mp3_path)
        torchaudio.save(str(dest), wav, sr)
        os.remove(mp3_path)
        print(f"✓ synthesized {dest.name}")
        return True
    except Exception as e:
        print(f"✗ TTS synth failed for {dest}: {e}")
        return False

def load_audio(path: Path):
    wav, sr = torchaudio.load(str(path))
    if sr != CONFIG["sample_rate"]:
        wav = torchaudio.transforms.Resample(sr, CONFIG["sample_rate"])(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.abs().max() > 0:
        wav = wav / wav.abs().max()
    target_len = int(CONFIG["duration_sec"] * CONFIG["sample_rate"])
    cur_len = wav.shape[1]
    if cur_len > target_len:
        wav = wav[:, :target_len]
    elif cur_len < target_len:
        wav = torch.cat([wav, torch.zeros(1, target_len - cur_len)], dim=1)
    return wav

def extract_features(waveform):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["sample_rate"],
        n_mels=CONFIG["n_mels"],
        n_fft=CONFIG["n_fft"],
        hop_length=CONFIG["hop_length"],
        power=2.0,
        center=True,
        pad_mode='reflect',
    ).to(CONFIG["device"])(waveform)
    mel = torch.clamp(mel, min=1e-9)
    mel = torch.log(mel)
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    return mel

def main():
    print("Downloading samples...")
    real_paths = download(REAL_URLS, REAL_DIR)
    fake_paths = download(FAKE_URLS, FAKE_DIR)
    if len(fake_paths) < 2:
        # fallback: synthesize two fake samples via TTS
        texts = [
            "This is an automatically generated sample meant to simulate a synthetic voice.",
            "Synthetic voices can sound natural, but they are not real humans speaking.",
        ]
        for i, txt in enumerate(texts, 1):
            dest = FAKE_DIR / f"fake_tts_{i}.wav"
            if synthesize_tts(txt, dest):
                fake_paths.append(dest)
    sample_list = [("real", p) for p in real_paths] + [("fake", p) for p in fake_paths]
    if not sample_list:
        print("No samples to score.")
        return

    device = torch.device(CONFIG["device"])
    model = SpecRNetLite(in_branches=1).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("\nScoring samples...")
    rows = []
    with torch.no_grad():
        for true_label, path in tqdm(sample_list, desc="Scoring"):
            wav = load_audio(path).to(device)
            feats = extract_features(wav)
            prob_fake = model(feats).squeeze().item()
            pred_label = "fake" if prob_fake >= 0.5 else "real"
            rows.append({
                "file": str(path),
                "true": true_label,
                "prob_fake": prob_fake,
                "pred": pred_label,
                "correct": pred_label == true_label,
            })

    print("\nResults (probability of fake):")
    for r in rows:
        print(f"[{r['true']}] pred={r['pred']:<5} prob_fake={r['prob_fake']:.4f} correct={r['correct']} :: {r['file']}")

if __name__ == "__main__":
    main()
