# Quickstart

Follow these steps to get the web app and CLI running.

## Prerequisites
- Windows with Python 3.11+ in PATH
- (Optional) CUDA-capable GPU for faster inference

## Option A: Web App (recommended)
1. Open PowerShell in the repo root: `cd <path-to>/DeepFake_detection`
2. Run the helper script (creates venv, installs deps, starts Flask):
   ```powershell
   powershell -ExecutionPolicy Bypass -File run_webapp.ps1
   ```
   - Default port: 5000; change with `-Port 8000` if needed.
3. Open the browser at: http://localhost:5000
4. Upload audio and/or image files and click Analyze.

## Option B: Manual Web App Start
1. From repo root, create/activate venv:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Start the server:
   ```powershell
   python web/app.py
   ```
4. Visit http://localhost:5000.

## Option C: CLI Inference
1. Activate venv (see steps above).
2. Run detections:
   - Audio: `python detect.py --audio samples/audio/fake_audio/fake_1.wav`
   - Image: `python detect.py --image samples/fake_images/fake_01.jpg`
   - Multimodal: `python detect.py --audio samples/audio/fake_audio/fake_1.wav --image samples/fake_images/fake_01.jpg`

## Notes
- Audio checkpoint expected at `checkpoints/audio_kaggle_best.pt`.
- The Hugging Face image model downloads automatically to `checkpoints/pretrained_hf` on first run.
- For CPU-only, add `--device cpu --precision fp32` to `detect.py` commands.
