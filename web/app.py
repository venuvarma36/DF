"""
Flask web app for DeepFake Detection
Provides REST API for audio and image deepfake detection
"""

import os
import sys
import json
import time
import torch
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.pipeline import load_audio_model, infer_audio
from src.fusion.fusion import fuse_scores
import yaml
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load config
cfg = yaml.safe_load(open('configs/hparams.yaml', 'r'))
audio_threshold = float(cfg.get('audio', {}).get('threshold', 0.5))
image_threshold = float(cfg.get('image', {}).get('threshold', 0.5))

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
print("Loading models...")
print("  - Loading audio model (trained)...")
audio_model = load_audio_model(cfg, device, 'checkpoints')

print("  - Loading image models (HuggingFace pretrained)...")
cache_dir = "checkpoints/pretrained_hf"

# Load V1 model
model_v1_name = cfg["image"].get("hf_model", "prithivMLmods/deepfake-detector-model-v1")
print(f"  - Loading V1: {model_v1_name}")
image_processor_v1 = AutoImageProcessor.from_pretrained(model_v1_name, cache_dir=cache_dir)
image_model_v1 = SiglipForImageClassification.from_pretrained(model_v1_name, cache_dir=cache_dir)
image_model_v1 = image_model_v1.to(device)
image_model_v1.eval()
print(f"✓ V1 model loaded successfully: {model_v1_name}")

# Load V3 model
model_v3_name = cfg["image"].get("hf_model_v3", "prithivMLmods/open-deepfake-detection")
print(f"  - Loading V3: {model_v3_name}")
image_processor_v3 = AutoImageProcessor.from_pretrained(model_v3_name, cache_dir=cache_dir)
image_model_v3 = SiglipForImageClassification.from_pretrained(model_v3_name, cache_dir=cache_dir)
image_model_v3 = image_model_v3.to(device)
image_model_v3.eval()
print(f"✓ V3 model loaded successfully: {model_v3_name}")

print("Models loaded!")


@app.route('/')
def home():
    """Serve homepage"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """Detect deepfakes for one or many audio/image files."""
    audio_files = request.files.getlist('audio')
    image_files = request.files.getlist('image')

    if not audio_files and not image_files:
        return jsonify({'error': 'No audio or image file provided'}), 400

    results = {
        'audio': [],
        'image': [],
        'error': None
    }

    # Process audio files
    for audio_file in audio_files:
        if not audio_file or not audio_file.filename:
            continue

        file_ext = os.path.splitext(audio_file.filename)[1] or '.wav'
        fd, audio_path = tempfile.mkstemp(suffix=file_ext)
        try:
            os.close(fd)
            audio_file.save(audio_path)

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise FileNotFoundError("Audio file not saved properly")

            start = time.time()
            with torch.no_grad():
                audio_score = infer_audio(audio_model, audio_path, cfg, device)
            latency = (time.time() - start) * 1000

            results['audio'].append({
                'filename': audio_file.filename,
                'prediction': 'FAKE' if audio_score >= audio_threshold else 'REAL',
                'confidence': round(float(audio_score), 3),
                'inference_time_ms': round(latency, 1)
            })
        except Exception as e:
            results['audio'].append({
                'filename': audio_file.filename,
                'error': str(e)
            })
        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

    # Process image files
    for image_file in image_files:
        if not image_file or not image_file.filename:
            continue

        file_ext = os.path.splitext(image_file.filename)[1] or '.jpg'
        fd, image_path = tempfile.mkstemp(suffix=file_ext)
        try:
            os.close(fd)
            image_file.save(image_path)

            if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                raise FileNotFoundError("Image file not saved properly")

            image = Image.open(image_path).convert("RGB")
            temp = float(cfg.get('image', {}).get('temperature', 1.0))
            
            # Run inference with V1 model
            start_v1 = time.time()
            inputs_v1 = image_processor_v1(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_v1 = image_model_v1(**inputs_v1)
                logits_v1 = outputs_v1.logits
                probs_v1 = torch.softmax(logits_v1 / temp, dim=-1)
                fake_prob_v1 = probs_v1[0][1].item()
            latency_v1 = (time.time() - start_v1) * 1000
            
            # Run inference with V3 model
            start_v3 = time.time()
            inputs_v3 = image_processor_v3(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_v3 = image_model_v3(**inputs_v3)
                logits_v3 = outputs_v3.logits
                probs_v3 = torch.softmax(logits_v3 / temp, dim=-1)
                fake_prob_v3 = probs_v3[0][1].item()
            latency_v3 = (time.time() - start_v3) * 1000
            
            # Determine which model has higher confidence (further from 0.5)
            confidence_v1 = abs(fake_prob_v1 - 0.5)
            confidence_v3 = abs(fake_prob_v3 - 0.5)
            
            if confidence_v1 >= confidence_v3:
                final_score = fake_prob_v1
                final_model = "V1"
            else:
                final_score = fake_prob_v3
                final_model = "V3"

            results['image'].append({
                'filename': image_file.filename,
                'v1_result': {
                    'model': 'V1 (deepfake-detector-model-v1)',
                    'prediction': 'FAKE' if fake_prob_v1 >= image_threshold else 'REAL',
                    'confidence': round(float(fake_prob_v1), 3),
                    'inference_time_ms': round(latency_v1, 1)
                },
                'v3_result': {
                    'model': 'V3 (open-deepfake-detection)',
                    'prediction': 'FAKE' if fake_prob_v3 >= image_threshold else 'REAL',
                    'confidence': round(float(fake_prob_v3), 3),
                    'inference_time_ms': round(latency_v3, 1)
                },
                'final_result': {
                    'selected_model': final_model,
                    'prediction': 'FAKE' if final_score >= image_threshold else 'REAL',
                    'confidence': round(float(final_score), 3),
                    'total_inference_time_ms': round(latency_v1 + latency_v3, 1)
                }
            })
        except Exception as e:
            results['image'].append({
                'filename': image_file.filename,
                'error': str(e)
            })
        finally:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception:
                    pass

    return jsonify(results)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'audio_model': 'trained checkpoint (checkpoints/audio_kaggle_best.pt)',
        'image_models': [
            'V1: prithivMLmods/deepfake-detector-model-v1',
            'V3: prithivMLmods/open-deepfake-detection'
        ],
        'mode': 'dual (both models used, highest confidence selected)'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
