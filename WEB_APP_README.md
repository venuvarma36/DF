# DeepFake Detection Web App

Modern web interface for audio and image deepfake detection.

## Features

✅ **Audio Detection** - Upload MP3, WAV, or FLAC files  
✅ **Image Detection** - Upload JPG, PNG, or BMP files  
✅ **Multimodal Fusion** - Combined detection when both audio and image provided  
✅ **Drag & Drop** - Easy file uploads  
✅ **Real-time Results** - Instant inference with confidence scores  
✅ **Responsive Design** - Works on desktop and mobile  

## Quick Start

### 1. Activate Virtual Environment
```bash
.\venv\Scripts\Activate.ps1
```

### 2. Start the Web App
```bash
python web/app.py
```

Or use the batch file:
```bash
start_web_app.bat
```

### 3. Open Browser
Navigate to: **http://localhost:5000**

## Usage

1. **Upload Files**
   - Drag and drop or click to upload audio and/or image
   - Supports multiple formats: MP3, WAV, FLAC, JPG, PNG, BMP

2. **Run Detection**
   - Click "Analyze" button
   - Wait for inference (typically 2-5 seconds)

3. **View Results**
   - Prediction: REAL or FAKE
   - Confidence: 0-100%
   - Inference time in milliseconds
   - Multimodal fusion result (if both inputs provided)

## File Structure

```
web/
├── app.py                 # Flask backend server
└── templates/
    └── index.html        # Frontend HTML/CSS/JavaScript
```

## API Endpoints

### POST /api/detect
Detect deepfakes in audio and/or image.

**Request:**
```
Content-Type: multipart/form-data
- audio: [optional] audio file
- image: [optional] image file
```

**Response:**
```json
{
  "audio": {
    "prediction": "FAKE|REAL",
    "confidence": 0.523,
    "inference_time_ms": 2844.5
  },
  "image": {
    "prediction": "FAKE|REAL",
    "confidence": 0.532,
    "inference_time_ms": 3408.3
  },
  "multimodal": {
    "prediction": "FAKE|REAL",
    "confidence": 0.527
  }
}
```

### GET /api/health
Health check.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda|cpu",
  "audio_model": "loaded",
  "image_model": "loaded"
}
```

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Android)

## Performance

- **Audio Inference**: ~2.8 seconds
- **Image Inference**: ~3.4 seconds
- **Multimodal**: ~1.8 seconds (combined)
- **GPU**: Optimized for RTX 2050 (4GB VRAM)
- **CPU**: Supported (slower inference)

## Troubleshooting

**Port 5000 already in use:**
```python
# In app.py, change: app.run(port=5000)
# To: app.run(port=5001)
```

**Models not loading:**
- Ensure `configs/hparams.yaml` exists
- Check `checkpoints/audio_best.pt` and `checkpoints/image_best.pt`

**CORS issues:**
- Web app and browser must be on same machine
- Or add CORS support in `app.py` if needed

## Model Information

- **Audio Model**: SpecRNet-Lite
  - Input: 16kHz audio
  - Output: Fake/Real score
  - Parameters: ~2.8M

- **Image Model**: DeiT-Small or EfficientNet-B0
  - Input: 224×224 RGB image
  - Output: Fake/Real score
  - Parameters: ~43.5M (DeiT) or ~4.2M (EfficientNet)

## Advanced Usage

### Change Model Device
Edit `web/app.py`:
```python
device = torch.device('cpu')  # Force CPU
```

### Change Port
Edit `web/app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
```

### Enable External Access
Edit `web/app.py`:
```python
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
    # Now accessible at: http://<your-ip>:5000
```

## Notes

- First load takes longer (model loading)
- Files are processed and deleted immediately
- No data is stored on server
- GPU acceleration requires CUDA-capable GPU
