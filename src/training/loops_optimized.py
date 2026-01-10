from __future__ import annotations
import os
import time
import math
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from .losses import FocalLoss, SupConLoss
from src.utils.metrics import compute_classification_metrics, compute_security_metrics, plot_training_curves
import numpy as np


class EarlyStopper:
    """Early stopping based on metric with patience"""
    def __init__(self, patience=5):
        self.patience = patience
        self.best = None
        self.count = 0

    def step(self, metric):
        if self.best is None or metric < self.best:
            self.best = metric
            self.count = 0
            return True
        else:
            self.count += 1
            return False

    def should_stop(self):
        return self.count >= self.patience


def get_phase_lr_schedule(base_lr, phase_epochs, current_phase_epoch):
    """
    Cosine annealing within each phase
    """
    return base_lr * 0.5 * (1 + np.cos(np.pi * current_phase_epoch / phase_epochs))


def print_detailed_metrics(epoch, phase_name, train_loss, val_metrics, sec_metrics, inference_time_ms, peak_gpu_mb):
    """Print all required metrics in a structured format"""
    print("\n" + "="*80)
    print(f"EPOCH {epoch:3d} | Phase: {phase_name:12s}")
    print("="*80)
    print(f"Train Loss: {train_loss:.4f}")
    print("\n📊 CLASSIFICATION METRICS:")
    print(f"  Accuracy:   {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:  {val_metrics['precision']:.4f}")
    print(f"  Recall:     {val_metrics['recall']:.4f}")
    print(f"  F1-Score:   {val_metrics['f1']:.4f}")
    print(f"  ROC-AUC:    {val_metrics['roc_auc']:.4f}")
    print("\n🔐 SECURITY METRICS:")
    far_curve = sec_metrics['FAR_curve']
    frr_curve = sec_metrics['FRR_curve']
    
    # Handle empty curves
    if len(far_curve) > 0 and len(frr_curve) > 0:
        idx = np.nanargmin(np.abs(far_curve - frr_curve))
        far_at_eer = far_curve[idx]
        frr_at_eer = frr_curve[idx]
        print(f"  FAR (False Accept Rate): {far_at_eer:.4f}")
        print(f"  FRR (False Reject Rate): {frr_at_eer:.4f}")
        print(f"  EER (Equal Error Rate):  {sec_metrics['EER']:.4f} ⭐")
    else:
        print(f"  EER (Equal Error Rate):  {sec_metrics['EER']} (curves empty)")
    print("\n⏱️  PERFORMANCE:")
    print(f"  Inference Latency: {inference_time_ms:.2f} ms")
    print(f"  Peak GPU Memory:   {peak_gpu_mb:.0f} MB")
    print("="*80)


def get_peak_gpu_memory():
    """Get peak GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def train_loop_optimized(
    model, 
    train_ds, 
    val_ds, 
    cfg: dict, 
    device: torch.device, 
    ckpt_path: str, 
    task: str = "audio",
    collate_fn=None
):
    """
    ============================================================================
    OPTIMIZED TRAINING LOOP FOR RTX 2050 (4GB VRAM)
    ============================================================================
    
    Implements:
    ✓ Gradient accumulation (larger effective batch size, lower memory)
    ✓ Gradient checkpointing (40% memory reduction)
    ✓ Phase-based learning rate schedules (warmup → finetune → polish)
    ✓ Mixed precision FP16 (2x memory efficiency)
    ✓ Detailed metric logging (accuracy, precision, recall, F1, ROC-AUC, FAR/FRR/EER)
    ✓ Peak GPU memory tracking
    ✓ Inference latency measurement
    
    TIME TARGETS:
    - Image: ≤14 hours
    - Audio: ≤5 hours
    """
    
    print(f"\n🚀 Starting OPTIMIZED training loop for {task.upper()} on {device}")
    
    # ========================================================================
    # 1. EXTRACT CONFIGURATION BY TASK
    # ========================================================================
    
    if task == "image":
        task_cfg = cfg.get("image_training", {})
        print(f"📐 Image Model: DeiT-Small (224×224)")
    else:  # audio
        task_cfg = cfg.get("audio_training", {})
        print(f"🎙️  Audio Model: SpecRNet-Lite")
    
    bs_train = task_cfg.get("batch_size_train", 32)
    bs_val = task_cfg.get("batch_size_val", 16)
    grad_accum_steps = task_cfg.get("gradient_accumulation_steps", 1)
    early_stop_patience = task_cfg.get("early_stopping_patience", 2)
    gradient_checkpointing = task_cfg.get("gradient_checkpointing", True)
    
    # Effective batch size
    effective_bs = bs_train * grad_accum_steps
    print(f"📦 Batch Size: {bs_train} × {grad_accum_steps} accum = {effective_bs} effective")
    print(f"🔍 Validation Batch Size: {bs_val}")
    print(f"💾 Gradient Checkpointing: {'ENABLED' if gradient_checkpointing else 'DISABLED'}")
    print(f"⏹️  Early Stopping Patience: {early_stop_patience}")
    
    # ========================================================================
    # 2. SETUP DATA LOADERS
    # ========================================================================
    
    dl_train = DataLoader(
        train_ds, 
        batch_size=bs_train, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    dl_val = DataLoader(
        val_ds, 
        batch_size=bs_val, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    print(f"✓ DataLoaders created: train={len(dl_train)} batches, val={len(dl_val)} batches")
    
    # ========================================================================
    # 3. SETUP MODEL & OPTIMIZATION
    # ========================================================================
    
    model = model.to(device)
    
    # Enable gradient checkpointing if supported
    if gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print(f"✓ Gradient checkpointing enabled (40% memory reduction)")
        else:
            print(f"⚠️  Model doesn't support gradient checkpointing")
    
    # Optimizer (will be overridden per-phase)
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
    
    # Loss functions
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    supcon = SupConLoss(temperature=0.1)
    
    # Mixed precision scaler (only for CUDA)
    enable_amp = cfg.get("training", {}).get("enable_amp", True) and device.type == "cuda"
    if device.type == "cuda":
        scaler = torch.amp.GradScaler(enabled=enable_amp)
    else:
        # For CPU, create a dummy scaler that does nothing
        class DummyScaler:
            def step(self, optimizer): optimizer.step()
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def update(self): pass
        scaler = DummyScaler()
    print(f"✓ Mixed Precision (FP16): {'ENABLED' if enable_amp else 'DISABLED'}")
    
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    
    # ========================================================================
    # 4. SETUP PHASE-BASED TRAINING SCHEDULE
    # ========================================================================
    
    phases = []
    for phase_key in ["phase1", "phase2", "phase3"]:
        if phase_key in task_cfg:
            phase_data = task_cfg[phase_key]
            phases.append({
                "name": phase_data.get("name", phase_key),
                "epochs": phase_data.get("epochs", 1),
                "lr": phase_data.get("lr", 1e-4),
                "freeze_backbone": phase_data.get("freeze_backbone", False)
            })
    
    if not phases:
        phases = [{
            "name": "training",
            "epochs": task_cfg.get("total_epochs", 20),
            "lr": task_cfg.get("learning_rate", 1e-4),
            "freeze_backbone": False
        }]
    
    print(f"\n📋 TRAINING SCHEDULE ({sum(p['epochs'] for p in phases)} total epochs):")
    for i, phase in enumerate(phases, 1):
        print(f"   Phase {i}: {phase['name']:12s} | {phase['epochs']:2d} epochs | LR={phase['lr']:.0e} | Freeze={phase['freeze_backbone']}")
    
    # ========================================================================
    # 5. PREPARE DIRECTORIES & TRACKING
    # ========================================================================
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(cfg.get("paths", {}).get("logs", "logs"), exist_ok=True)
    
    early_stopper = EarlyStopper(patience=early_stop_patience)
    best_eer = math.inf

    # Progress tracking
    total_epochs = sum(p['epochs'] for p in phases)
    start_time = time.perf_counter()
    
    history = {
        "epoch": [], "phase": [], 
        "train_loss": [],
        "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": [], "val_roc_auc": [],
        "val_far": [], "val_frr": [], "val_eer": [],
        "inference_time_ms": [], "peak_gpu_mb": []
    }
    
    # ========================================================================
    # 6. PHASE-BASED TRAINING LOOP
    # ========================================================================
    
    global_epoch = 0
    
    for phase_idx, phase in enumerate(phases):
        phase_name = phase["name"]
        phase_epochs = phase["epochs"]
        phase_lr = phase["lr"]
        freeze_backbone = phase["freeze_backbone"]
        
        print(f"\n{'='*80}")
        print(f"🔄 PHASE {phase_idx + 1}: {phase_name.upper()} ({phase_epochs} epochs)")
        print(f"{'='*80}")
        
        # Freeze/unfreeze backbone for this phase
        if hasattr(model, 'backbone') and freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
            print(f"❄️  Backbone FROZEN for {phase_name}")
        elif hasattr(model, 'backbone') and not freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print(f"🔥 Backbone UNFROZEN for {phase_name}")
        
        # Update optimizer learning rate for this phase
        for param_group in opt.param_groups:
            param_group['lr'] = phase_lr
        
        # Train this phase
        for phase_epoch in range(phase_epochs):
            global_epoch = global_epoch + 1
            
            # ================================================================
            # TRAINING STEP
            # ================================================================
            
            model.train()
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            
            train_losses = []
            batch_idx = 0
            
            for batch_idx, batch in enumerate(dl_train):
                if batch is None:
                    continue
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    out = model(x)
                    logits = torch.logit(out.clamp(1e-4, 1 - 1e-4))
                    loss = focal(logits, y)
                    loss = loss / grad_accum_steps  # Scale for accumulation
                
                # Backward pass
                scaler.scale(loss).backward()
                train_losses.append(loss.item() * grad_accum_steps)
                
                # Gradient accumulation step
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Optimizer step
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            
            avg_train_loss = np.mean(train_losses)
            
            # ================================================================
            # VALIDATION STEP
            # ================================================================
            
            model.eval()
            val_outputs = []
            val_targets = []
            inference_times = []
            
            with torch.no_grad():
                for batch in dl_val:
                    if batch is None:
                        continue
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    
                    # Measure inference latency
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    
                    with torch.cuda.amp.autocast(enabled=enable_amp):
                        out = model(x)
                    
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    
                    inference_times.append((t1 - t0) * 1000 / x.size(0))  # ms per sample
                    
                    val_outputs.append(out.detach().cpu())
                    val_targets.append(y.detach().cpu())
            
            val_outputs = torch.cat(val_outputs).numpy()
            val_targets = torch.cat(val_targets).numpy()
            val_preds = (val_outputs >= 0.5).astype(int)
            
            avg_inference_time_ms = np.mean(inference_times)
            peak_gpu_mb = get_peak_gpu_memory()
            
            # ================================================================
            # COMPUTE METRICS
            # ================================================================
            
            cls_metrics = compute_classification_metrics(val_targets, val_preds, val_outputs)
            sec_metrics = compute_security_metrics(val_targets, val_outputs)
            
            eer = sec_metrics["EER"]
            far_curve = sec_metrics["FAR_curve"]
            frr_curve = sec_metrics["FRR_curve"]
            
            # Handle empty curves
            if len(far_curve) > 0 and len(frr_curve) > 0:
                eer_idx = np.nanargmin(np.abs(far_curve - frr_curve))
                far_at_eer = far_curve[eer_idx]
                frr_at_eer = frr_curve[eer_idx]
            else:
                far_at_eer = float('nan')
                frr_at_eer = float('nan')
            
            # ================================================================
            # SAVE HISTORY
            # ================================================================
            
            history["epoch"].append(global_epoch)
            history["phase"].append(phase_name)
            history["train_loss"].append(avg_train_loss)
            history["val_accuracy"].append(cls_metrics["accuracy"])
            history["val_precision"].append(cls_metrics["precision"])
            history["val_recall"].append(cls_metrics["recall"])
            history["val_f1"].append(cls_metrics["f1"])
            history["val_roc_auc"].append(cls_metrics["roc_auc"])
            history["val_far"].append(far_at_eer)
            history["val_frr"].append(frr_at_eer)
            history["val_eer"].append(eer)
            history["inference_time_ms"].append(avg_inference_time_ms)
            history["peak_gpu_mb"].append(peak_gpu_mb)
            
            # ================================================================
            # PRINT DETAILED METRICS
            # ================================================================
            
            print_detailed_metrics(
                global_epoch,
                phase_name,
                avg_train_loss,
                cls_metrics,
                sec_metrics,
                avg_inference_time_ms,
                peak_gpu_mb
            )

            # ============================================================
            # PROGRESS + ETA
            # ============================================================
            elapsed = time.perf_counter() - start_time
            percent = (global_epoch / total_epochs) * 100 if total_epochs else 0
            est_total = elapsed / global_epoch * total_epochs if global_epoch else 0
            eta = max(est_total - elapsed, 0)
            print(f"Progress: {percent:5.1f}% | Elapsed: {elapsed/3600:.2f}h | ETA: {eta/3600:.2f}h (epochs {global_epoch}/{total_epochs})")
            
            # ================================================================
            # CHECKPOINT & EARLY STOPPING
            # ================================================================
            
            if eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), ckpt_path)
                print(f"✅ New best EER: {eer:.4f} | Saved to {ckpt_path}")
            
            if not early_stopper.step(eer):
                print(f"⚠️  No improvement. Early stopping patience: {early_stopper.count}/{early_stop_patience}")
            
            if early_stopper.should_stop():
                print(f"\n🛑 EARLY STOPPING TRIGGERED after {global_epoch} epochs (patience={early_stop_patience})")
                break
    
    # ========================================================================
    # 7. FINAL RESULTS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE for {task.upper()}")
    print(f"{'='*80}")
    print(f"Best EER: {best_eer:.4f}")
    print(f"Final Checkpoint: {ckpt_path}")
    print(f"Peak GPU Memory: {max(history['peak_gpu_mb']):.0f} MB")
    print(f"Avg Inference Latency: {np.mean(history['inference_time_ms']):.2f} ms")
    print(f"Total Epochs: {global_epoch}")
    
    # Save training curves
    logs_dir = cfg.get("paths", {}).get("logs", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    plot_training_curves(history, os.path.join(logs_dir, f"{task}_training_curves.png"))
    
    return best_eer, history


# Keep old function for compatibility
def train_loop(model, train_ds, val_ds, cfg: dict, device: torch.device, ckpt_path: str, task: str = "audio"):
    """Wrapper for backward compatibility"""
    best_eer, history = train_loop_optimized(model, train_ds, val_ds, cfg, device, ckpt_path, task)
    return best_eer
