import os

audio_exts = {".wav", ".flac", ".mp3", ".m4a"}
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}


def detect_modality(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in audio_exts:
        return "audio"
    if ext in image_exts:
        return "image"
    return None
