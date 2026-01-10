"""
Evaluate local samples in samples/ using the trained model.
Supports filenames:
- real/natural: real_*.wav, natural_*.wav
- fake/robotic: fake_*.wav, robotic_*.wav
"""
import sys
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

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
SAMPLES_DIR = Path(r"C:\DeepFake_detection\samples")


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
    device = torch.device(CONFIG["device"])
    model = SpecRNetLite(in_branches=1).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()

    label_patterns = {
        "real": ["real_*.wav", "natural_*.wav"],
        "fake": ["fake_*.wav", "robotic_*.wav"],
    }

    files = []
    for label, patterns in label_patterns.items():
        for pattern in patterns:
            files.extend([(label, p) for p in sorted(SAMPLES_DIR.glob(pattern))])
    if not files:
        print("No local sample wavs found.")
        return

    rows = []
    with torch.no_grad():
        for true_label, path in tqdm(files, desc="Scoring"):
            wav = load_audio(path).to(device)
            feats = extract_features(wav)
            prob_fake = model(feats).squeeze().item()
            pred_label = "fake" if prob_fake >= 0.5 else "real"
            rows.append({
                "file": str(path.name),
                "true": true_label,
                "prob_fake": prob_fake,
                "pred": pred_label,
                "correct": pred_label == true_label,
            })

    correct = sum(r["correct"] for r in rows)
    total = len(rows)
    print(f"\nLocal samples accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print("\nDetails (prob_fake):")
    for r in rows:
        print(f"[{r['true']}] pred={r['pred']:<5} prob_fake={r['prob_fake']:.4f} correct={r['correct']} :: {r['file']}")

if __name__ == "__main__":
    main()
