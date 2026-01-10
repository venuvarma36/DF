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

print("  - Loading image model (HuggingFace pretrained)...")
model_name = "prithivMLmods/deepfake-detector-model-v1"
cache_dir = "checkpoints/pretrained_hf"
image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
image_model = SiglipForImageClassification.from_pretrained(model_name, cache_dir=cache_dir)
image_model = image_model.to(device)
image_model.eval()
print("Models loaded!")


@app.route('/')
def home():
    """Serve homepage"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    API endpoint for detection
    Expects multipart form data with 'audio' and/or 'image' files
    """
    try:
        results = {
            'audio': None,
            'image': None,
            'multimodal': None,
            'error': None
        }
        
        # Check for audio file
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and audio_file.filename:
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
                audio_file.save(audio_path)
                
                try:
                    start = time.time()
                    with torch.no_grad():
                        audio_score = infer_audio(audio_model, audio_path, cfg, device)
                    latency = (time.time() - start) * 1000
                    
                    results['audio'] = {
                        'prediction': 'FAKE' if audio_score >= audio_threshold else 'REAL',
                        'confidence': round(float(audio_score), 3),
                        'inference_time_ms': round(latency, 1)
                    }
                finally:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
        
        # Check for image file
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
                image_file.save(image_path)
                
                try:
                    start = time.time()
                    # Load and preprocess image
                    image = Image.open(image_path).convert("RGB")
                    inputs = image_processor(images=image, return_tensors="pt").to(device)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = image_model(**inputs)
                        logits = outputs.logits
                        temp = float(cfg.get('image', {}).get('temperature', 1.0))
                        probs = torch.softmax(logits / temp, dim=-1)
                        # probs[0][0] = Real, probs[0][1] = Fake
                        fake_prob = probs[0][1].item()
                    
                    latency = (time.time() - start) * 1000
                    image_score = fake_prob  # Use for fusion
                    
                    results['image'] = {
                        'prediction': 'FAKE' if fake_prob >= image_threshold else 'REAL',
                        'confidence': round(float(fake_prob), 3),
                        'inference_time_ms': round(latency, 1)
                    }
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
        
        # Multimodal fusion if both present
        if results['audio'] and results['image']:
            audio_score_val = audio_score if audio_file else None
            image_score_val = image_score if image_file else None
            
            if audio_score_val is not None and image_score_val is not None:
                fused_score = fuse_scores(audio_score_val, image_score_val, cfg.get('fusion', {}))
                results['multimodal'] = {
                    'prediction': 'FAKE' if fused_score >= 0.5 else 'REAL',
                    'confidence': round(float(fused_score), 3)
                }
        
        if not results['audio'] and not results['image']:
            return jsonify({'error': 'No audio or image file provided'}), 400
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'audio_model': 'trained checkpoint (checkpoints/audio_best.pt)',
        'image_model': 'pretrained HuggingFace (prithivMLmods/deepfake-detector-model-v1)'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
