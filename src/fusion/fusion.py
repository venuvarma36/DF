import math

def fuse_scores(audio_score: float, image_score: float, cfg: dict):
    alpha = cfg.get("alpha", 0.6)
    beta = cfg.get("beta", 0.4)
    temp = 1.0
    if cfg.get("temperature_scaling", True):
        temp = 1.0  # placeholder; learned during calibration
    a = 1.0 / (1.0 + math.exp(-(audio_score - 0.5) / max(1e-6, temp)))
    b = 1.0 / (1.0 + math.exp(-(image_score - 0.5) / max(1e-6, temp)))
    fused = alpha * a + beta * b
    # Optional inconsistency penalty if scores disagree strongly
    incons_w = cfg.get("inconsistency_weight", 0.0)
    inconsistency = abs(a - b)
    fused = max(0.0, min(1.0, fused - incons_w * inconsistency))
    return fused
