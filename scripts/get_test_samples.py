import os
import zipfile
import tempfile
from pathlib import Path
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import torchaudio
from torchaudio.functional import resample

# 10 real (Free Spoken Digit Dataset, public wavs)
REAL = [
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_jackson_0.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/1_george_1.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/2_lucas_2.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/3_theo_3.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/4_yweweler_4.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/5_jackson_5.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/6_george_6.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/7_lucas_7.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/8_theo_8.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/9_yweweler_9.wav",
]

# Synthetic (attempt URLs; will fall back to TTS if not enough)
FAKE = [
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/wavenet/LJ001-0001.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/wavenet/LJ001-0002.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/wavenet/LJ001-0003.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/wavenet/LJ001-0004.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/wavenet/LJ001-0005.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/melgan/LJ001-0001.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/melgan/LJ001-0002.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/melgan/LJ001-0003.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/melgan/LJ001-0004.wav",
    "https://zenodo.org/record/5642694/files/generated_audio/ljspeech/melgan/LJ001-0005.wav",
]

TTS_TEXTS = [
    "This is an automatically generated voice sample for testing deepfake detection.",
    "Synthetic speech can sound very natural when produced by modern text to speech models.",
    "We need diverse audio to evaluate whether the classifier can catch fake voices.",
    "Another fake sample generated to act as a negative example.",
    "Deep learning based text to speech produces high quality audio.",
    "Here is yet another artificial voice clip for your evaluation pipeline.",
    "Evaluating robustness requires many kinds of synthetic speech clips.",
    "This short sentence is spoken by a text to speech engine.",
    "Please classify this clip as fake; it is not a real human recording.",
    "Final synthetic clip to reach ten generated examples for testing.",
]

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

def fetch(url: str, dest: Path) -> None:
    urlretrieve(url, dest)


def normalize_audio(inp: Path, out: Path, sample_rate: int = 16000) -> None:
    wav, sr = torchaudio.load(inp)
    if wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = resample(wav, sr, sample_rate)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    torchaudio.save(out, wav, sample_rate)


def process(urls, prefix, start_idx=1):
    success = 0
    for idx, url in enumerate(urls, start_idx):
        tmp_ext = Path(url).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_ext) as tmp:
            tmp_path = Path(tmp.name)
        out_path = SAMPLES_DIR / f"{prefix}_{idx}.wav"
        print(f"→ {out_path.name}: downloading {url}")
        try:
            fetch(url, tmp_path)
            normalize_audio(tmp_path, out_path)
            success += 1
        except Exception as e:
            print(f"  ✗ failed {url}: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)
    return success


def synthesize_tts(count_start: int = 1):
    try:
        from gtts import gTTS
    except ImportError:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "gtts"], stdout=subprocess.DEVNULL)
        from gtts import gTTS

    made = 0
    for i, text in enumerate(TTS_TEXTS, count_start):
        dest = SAMPLES_DIR / f"fake_{i}.wav"
        try:
            mp3_path = dest.with_suffix(".mp3")
            gTTS(text=text, lang="en").save(mp3_path)
            normalize_audio(mp3_path, dest)
            mp3_path.unlink(missing_ok=True)
            made += 1
        except Exception as e:
            print(f"  ✗ TTS failed ({text[:30]}...): {e}")
    return made


def main():
    print("Downloading REAL samples...")
    process(REAL, "real", 1)

    print("\nDownloading FAKE samples...")
    got = process(FAKE, "fake", 1)
    if got < 10:
        print(f"\nFetched {got} fake files; synthesizing {10 - got} via gTTS...")
        synthesize_tts(got + 1)

    print("\nDone. Files saved in samples/ as real_*.wav and fake_*.wav")


if __name__ == "__main__":
    main()

