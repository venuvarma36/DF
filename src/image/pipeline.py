from __future__ import annotations
import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification


class ImageModel:
    def __init__(self, backend: str, model, processor=None):
        self.backend = backend
        self.model = model
        self.processor = processor


def load_image_model(cfg: dict, device: torch.device, ckpt_dir: str):
    """Load image model.

    - If cfg["image"]["backend"] == "hf", load the Hugging Face SigLIP model.
    - Otherwise fall back to the local/timm checkpoints.
    """
    backend = cfg["image"].get("backend", "timm")

    if backend == "hf":
        model_name = cfg["image"].get("hf_model", "prithivMLmods/deepfake-detector-model-v1")
        cache_dir = os.path.join(ckpt_dir, "pretrained_hf")
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model = SiglipForImageClassification.from_pretrained(model_name, cache_dir=cache_dir)
        model = model.to(device)
        model.eval()
        return ImageModel("hf", model, processor)

    # timm / local checkpoint path
    img_size = cfg["image"]["params"].get("img_size", 224)
    ckpt_path = os.path.join(ckpt_dir, "image_best.pt")
    from timm import create_model

    if os.path.exists(ckpt_path):
        model_name = "efficientnet_b0"
        model = create_model(model_name, pretrained=False, num_classes=1, in_chans=3)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)
    else:
        cfg_name = cfg["image"].get("model", "efficientnet_b0")
        model_name = _timm_name(cfg_name)
        try:
            model = create_model(model_name, pretrained=True, num_classes=1, in_chans=3)
        except Exception:
            model_name = "efficientnet_b0"
            model = create_model(model_name, pretrained=True, num_classes=1, in_chans=3)

    model = model.to(device)
    model.eval()
    return ImageModel("timm", model)


def _timm_name(name: str):
    # Map config-friendly alias to timm model name
    aliases = {
        "deit_small_patch8_distilled": "deit_small_distilled_patch16_224",  # patch8 often custom; using nearest timm
        "vit_small_patch8": "vit_small_patch16_224",  # same note as above
        "efficientnet_v2_s": "efficientnetv2_s"
    }
    return aliases.get(name, name)


def preprocess_image(path: str, img_size: int = 224):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x


def infer_image(image_model: ImageModel, image_path: str, cfg: dict, device: torch.device):
    temp = float(cfg.get("image", {}).get("temperature", 1.0))
    if image_model.backend == "hf":
        image = Image.open(image_path).convert("RGB")
        inputs = image_model.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = image_model.model(**inputs).logits / temp
            probs = torch.softmax(logits, dim=-1)
            fake_prob = probs[0][1].item()  # class 1 = fake
        return fake_prob

    img_size = cfg["image"]["params"].get("img_size", 224)
    x = preprocess_image(image_path, img_size).to(device)
    with torch.no_grad():
        logit = image_model.model(x) / temp
        score = torch.sigmoid(logit)
    return score.item()
