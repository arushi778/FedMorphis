"""
FedMorphis Phase 2B EXP-1: Aspect-Ratio-Preserving Letterbox Preprocessing
===========================================================================
Experiment: Phase 2B EXP-1 (Letterbox Padding vs. Squashed Resize)

Evaluates whether aspect-ratio-preserving letterbox preprocessing improves
DR lesion segmentation relative to the Phase 2 baseline anchor.

- Canvas: 512x512
- Preprocessing: Isotropic scaling + symmetric black padding
- Loss: Valid-pixel-masked BCE + soft Dice (excluding padding)
- Evaluation: Metrics computed strictly on valid retinal pixels
- Artifacts: Saved ONLY in results/DR/experiments/exp01_letterbox_padding/
- Baseline protection: Checkpoint best_dr_unet.pth is NEVER overwritten.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dr_letterbox_dataset import DDRLetterboxDataset
from shared_unet import SharedUNet


# ===========================================================================
# Configuration
# ===========================================================================
CFG = {
    "experiment_id": "EXP-1",
    "experiment_name": "Phase 2B EXP-1: Aspect-Ratio Letterbox Preprocessing",
    "random_seed": 42,
    "image_size": 512,
    "target_lesion": "all",
    "batch_size": 2,
    "epochs": 20,
    "learning_rate": 1e-4,
    "loss": "masked_bce+masked_dice",
    "threshold": 0.5,
    "exp_dir": str(REPO_ROOT / "results" / "DR" / "experiments" / "exp01_letterbox_padding"),
    "checkpoint_dir": str(REPO_ROOT / "results" / "DR" / "experiments" / "exp01_letterbox_padding" / "checkpoints"),
    "pred_dir": str(REPO_ROOT / "results" / "DR" / "experiments" / "exp01_letterbox_padding" / "val_predictions"),
}


# ===========================================================================
# Reproducibility
# ===========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Masked Loss Functions
# ===========================================================================
def masked_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice loss computed over valid pixels only.
    If valid_mask is None, computes standard unmasked Dice loss over all pixels.
    """
    probs = torch.sigmoid(logits)

    if valid_mask is None:
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return (1.0 - dice).mean()
    else:
        loss_list = []
        for b in range(probs.size(0)):
            vm = valid_mask[b].bool().view(-1)
            p_flat = probs[b].view(-1)[vm]
            t_flat = targets[b].view(-1)[vm]
            intersection = (p_flat * t_flat).sum()
            denominator = p_flat.sum() + t_flat.sum()
            dice = (2.0 * intersection + smooth) / (denominator + smooth)
            loss_list.append(1.0 - dice)
        return torch.stack(loss_list).mean()


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCE loss computed strictly over valid retinal pixels, excluding padding.
    If valid_mask is None, reduces identically to standard binary_cross_entropy_with_logits.
    """
    if valid_mask is None:
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)
    else:
        pixel_bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        masked_bce = (pixel_bce * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        return masked_bce


def masked_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked BCE + Masked Soft Dice loss."""
    return masked_bce_loss(logits, targets, valid_mask) + masked_dice_loss(logits, targets, valid_mask)


# ===========================================================================
# Metrics Function
# ===========================================================================
@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, Union[float, int]]:
    """
    Computes binary lesion segmentation metrics.
    If valid_mask is supplied, flattens only valid retinal pixels,
    strictly excluding padding from TP, FP, FN, and TN.
    """
    preds = (torch.sigmoid(logits) > threshold).float()

    if valid_mask is not None:
        vm = valid_mask.bool().view(-1)
        preds_flat = preds.view(-1)[vm]
        targets_flat = targets.view(-1)[vm]
    else:
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1.0 - targets_flat)).sum()
    fn = ((1.0 - preds_flat) * targets_flat).sum()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum()

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)

    pred_px = tp + fp
    gt_px = tp + fn
    ratio_pred_to_gt = (pred_px.float() / gt_px.float().clamp(min=1.0)).item()

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "precision": precision.item(),
        "pred_to_gt_ratio": ratio_pred_to_gt,
        "tp": int(tp.item()),
        "fp": int(fp.item()),
        "fn": int(fn.item()),
        "tn": int(tn.item()),
    }


# ===========================================================================
# GPU Pre-Flight Dry Run
# ===========================================================================
def run_cuda_dry_run(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Executes a single forward, loss, backward, and optimizer step on batch size 2
    to measure actual peak VRAM consumption and test for OOM vulnerability.
    """
    print("\n--- Running GPU Pre-Flight Memory Dry Run ---")
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    images, masks, valid_masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    valid_masks = valid_masks.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = masked_combined_loss(logits, masks, valid_masks)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    stats = {}
    if device.type == "cuda":
        alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        res_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        stats = {
            "allocated_gb": alloc_gb,
            "reserved_gb": res_gb,
            "total_gb": total_gb,
            "loss": loss.item(),
        }
        print(f"  Device:               {torch.cuda.get_device_name(0)}")
        print(f"  Total Physical VRAM:  {total_gb:.2f} GB")
        print(f"  Peak Memory Allocated:{alloc_gb:.2f} GB")
        print(f"  Peak Memory Reserved: {res_gb:.2f} GB ({res_gb/total_gb*100:.1f}% of total)")
        print(f"  Dry Run Step Loss:    {loss.item():.4f}")
    else:
        stats = {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0, "loss": loss.item()}
        print("  Running on CPU. CUDA memory statistics not applicable.")

    print("--- GPU Pre-Flight Memory Dry Run PASSED ---\n")
    return stats


# ===========================================================================
# Training & Validation Loops
# ===========================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    exp_dir: Path,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks, valid_masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)

        try:
            optimizer.zero_grad()
            logits = model(images)
            loss = masked_combined_loss(logits, masks, valid_masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[FATAL] CUDA Out of Memory during training at batch {batch_idx}!")
            torch.cuda.empty_cache()
            oom_report = {
                "status": "FAILED_OOM",
                "batch_idx": batch_idx,
                "error": str(e),
                "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0,
                "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3) if device.type == "cuda" else 0,
            }
            exp_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_dir / "oom_failure_report.json", "w") as f:
                json.dump(oom_report, f, indent=2)
            sys.exit(1)

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Union[float, Dict]]:
    model.eval()
    running_loss = 0.0

    all_logits = []
    all_targets = []
    all_valid_masks = []
    per_image_metrics = []

    for images, masks, valid_masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)

        logits = model(images)
        loss = masked_combined_loss(logits, masks, valid_masks)
        running_loss += loss.item()

        # Per-image metric calculation
        for b in range(images.size(0)):
            m_img = compute_metrics(
                logits[b : b + 1].cpu(),
                masks[b : b + 1].cpu(),
                valid_mask=valid_masks[b : b + 1].cpu(),
                threshold=CFG["threshold"],
            )
            per_image_metrics.append(m_img)

        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())
        all_valid_masks.append(valid_masks.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_valid_masks = torch.cat(all_valid_masks, dim=0)

    # Global pooled metrics strictly on valid pixels
    global_metrics = compute_metrics(
        all_logits, all_targets, valid_mask=all_valid_masks, threshold=CFG["threshold"]
    )
    global_metrics["loss"] = running_loss / max(len(loader), 1)

    # Macro averages across all validation images
    global_metrics["macro_dice"] = float(np.mean([m["dice"] for m in per_image_metrics]))
    global_metrics["macro_iou"] = float(np.mean([m["iou"] for m in per_image_metrics]))
    global_metrics["macro_sensitivity"] = float(np.mean([m["sensitivity"] for m in per_image_metrics]))
    global_metrics["macro_precision"] = float(np.mean([m["precision"] for m in per_image_metrics]))

    return global_metrics


# ===========================================================================
# Qualitative Visualizations
# ===========================================================================
@torch.no_grad()
def save_validation_predictions(
    model: nn.Module,
    dataset: DDRLetterboxDataset,
    device: torch.device,
    pred_dir: Path,
    num_samples: int = 6,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    selected = []
    for idx in range(len(dataset)):
        img_t, mask_t, vmask_t = dataset[idx]
        if mask_t.sum() > 100:
            selected.append((idx, img_t, mask_t, vmask_t))
        if len(selected) >= num_samples:
            break

    print(f"Generating {len(selected)} letterboxed validation prediction panels...")

    for idx, img_t, gt_mask_t, vmask_t in selected:
        info = dataset.get_sample_letterbox_info(idx)
        img_id = info["image_id"]

        logits = model(img_t.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits.cpu()).squeeze(0)
        pred_mask = (prob > CFG["threshold"]).float()

        # Mask predicted output with valid mask so padding remains zero
        pred_mask = pred_mask * vmask_t

        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_np = gt_mask_t.squeeze(0).numpy()
        pred_np = pred_mask.squeeze(0).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

        # 1. Fundus Image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Fundus (Letterbox)\n{img_id}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        # 2. Ground Truth Mask
        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Ground Truth Mask\n({int(gt_np.sum())} lesion px)", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        # 3. Predicted Mask
        axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"Predicted Mask\n({int(pred_np.sum())} lesion px)", fontsize=10, fontweight="bold")
        axes[2].axis("off")

        # 4. Overlay
        axes[3].imshow(img_np)
        overlay = np.zeros((*gt_np.shape, 4), dtype=np.uint8)
        tp_mask = (gt_np > 0) & (pred_np > 0)
        fp_mask = (gt_np == 0) & (pred_np > 0)
        fn_mask = (gt_np > 0) & (pred_np == 0)
        overlay[tp_mask] = [0, 220, 80, 200]    # Green = TP
        overlay[fp_mask] = [50, 120, 255, 180]  # Blue  = FP
        overlay[fn_mask] = [255, 40, 40, 200]   # Red   = FN
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay: TP=Green FP=Blue FN=Red", fontsize=10, fontweight="bold")
        axes[3].axis("off")

        m = compute_metrics(
            logits.cpu(), gt_mask_t.unsqueeze(0), valid_mask=vmask_t.unsqueeze(0), threshold=CFG["threshold"]
        )
        fig.suptitle(
            f"EXP-1 [LETTERBOX] Dice={m['dice']:.3f}  IoU={m['iou']:.3f}  "
            f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  Prec={m['precision']:.3f}  Ratio={m['pred_to_gt_ratio']:.2f}",
            fontsize=11,
        )

        plt.tight_layout()
        save_path = pred_dir / f"exp01_val_pred_{img_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(selected)} prediction panels to {pred_dir}")


# ===========================================================================
# Metric Curves Plotting
# ===========================================================================
def save_metric_curves(history: Dict[str, List[float]], output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], "r-s", markersize=3, label="Val Loss")
    axes[0, 0].set_title("Training & Validation Loss (Masked BCE + Dice)", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Dice & IoU
    axes[0, 1].plot(epochs, history["val_dice"], "g-o", markersize=3, label="Global Val Dice")
    axes[0, 1].plot(epochs, history["macro_dice"], "g--", label="Macro Val Dice")
    axes[0, 1].plot(epochs, history["val_iou"], "b-s", markersize=3, label="Global Val IoU")
    axes[0, 1].axhline(0.3609, color="gray", linestyle=":", label="Baseline Global Dice (0.3609)")
    axes[0, 1].set_title("Validation Dice & IoU", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Sensitivity & Precision
    axes[1, 0].plot(epochs, history["val_sensitivity"], "orange", marker="^", markersize=3, label="Sensitivity (Recall)")
    axes[1, 0].plot(epochs, history["val_precision"], "purple", marker="v", markersize=3, label="Precision")
    axes[1, 0].axhline(0.2782, color="orange", linestyle=":", label="Baseline Sens (0.2782)")
    axes[1, 0].set_title("Validation Sensitivity vs. Precision", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Pred/GT Ratio
    axes[1, 1].plot(epochs, history["val_ratio"], "teal", marker="x", markersize=3, label="Pred/GT Pixel Ratio")
    axes[1, 1].axhline(1.0, color="black", linestyle="-.", label="Perfect Calibration (1.0)")
    axes[1, 1].axhline(0.542, color="gray", linestyle=":", label="Baseline Ratio (0.542)")
    axes[1, 1].set_title("Predicted / Ground-Truth Lesion Area Ratio", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].set_ylim(0, 2)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle("Phase 2B EXP-1: Letterbox Preprocessing vs Baseline Anchor", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "metric_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric curves: {out_path}")


# ===========================================================================
# Main Execution Harness
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2B EXP-1: Letterbox Preprocessing")
    parser.add_argument("--dry-run-only", action="store_true", help="Run 1-batch GPU dry run and exit without full training.")
    args = parser.parse_args()

    print("=" * 80)
    print(f"EXPERIMENT: {CFG['experiment_name']}")
    print(f"ID:         {CFG['experiment_id']}")
    print("=" * 80)

    set_seed(CFG["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = Path(CFG["exp_dir"])
    ckpt_dir = Path(CFG["checkpoint_dir"])
    pred_dir = Path(CFG["pred_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Save exact configuration
    with open(exp_dir / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    # Datasets
    print("\nLoading datasets...")
    train_ds = DDRLetterboxDataset(split="train", image_size=CFG["image_size"])
    valid_ds = DDRLetterboxDataset(split="valid", image_size=CFG["image_size"])
    print(f"  Train: {len(train_ds)} images")
    print(f"  Valid: {len(valid_ds)} images (Test split locked)")

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model & Optimizer
    model = SharedUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])

    # GPU Dry Run
    dry_run_stats = run_cuda_dry_run(model, train_loader, optimizer, device)

    if args.dry_run_only:
        print("\nDry run requested. Exiting successfully before full training.")
        return

    # Full 20-epoch training loop
    print(f"\nStarting 20-epoch training run for {CFG['experiment_id']}...")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "macro_dice": [],
        "val_iou": [],
        "val_sensitivity": [],
        "val_specificity": [],
        "val_precision": [],
        "val_ratio": [],
    }

    best_val_dice = 0.0
    best_epoch = 0
    best_metrics = {}

    training_start = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, exp_dir)
        val_metrics = validate_epoch(model, valid_loader, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["macro_dice"].append(val_metrics["macro_dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["val_specificity"].append(val_metrics["specificity"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_ratio"].append(val_metrics["pred_to_gt_ratio"])

        print(
            f"Epoch {epoch:02d}/{CFG['epochs']:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Global Dice: {val_metrics['dice']:.4f} (Macro: {val_metrics['macro_dice']:.4f}) | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Sens: {val_metrics['sensitivity']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Ratio: {val_metrics['pred_to_gt_ratio']:.3f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpointing based on Global Validation Dice
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            best_metrics = {k: v for k, v in val_metrics.items()}
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "cfg": CFG,
                },
                ckpt_dir / "best_model.pth",
            )
            print(f"  -> New best checkpoint saved to {ckpt_dir / 'best_model.pth'} (Global Dice: {best_val_dice:.4f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - training_start

    # Save final checkpoint
    torch.save(
        {
            "epoch": CFG["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "cfg": CFG,
        },
        ckpt_dir / "final_model.pth",
    )

    # Save history and curves
    with open(exp_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    save_metric_curves(history, exp_dir)

    # Generate prediction panels using best checkpoint
    print("\nLoading best checkpoint for prediction panels...")
    best_ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    save_validation_predictions(model, valid_ds, device, pred_dir, num_samples=6)

    # Comparative Summary Report
    report_path = exp_dir / "exp01_vs_baseline_report.txt"
    with open(report_path, "w") as f:
        f.write("Phase 2B EXP-1 (Letterbox Preprocessing) vs Baseline Anchor\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total training time: {total_time / 60:.1f} minutes\n")
        f.write(f"Best Epoch: {best_epoch} / {CFG['epochs']}\n\n")
        f.write(f"{'Metric':<24} {'Baseline Anchor':>18} {'EXP-1 (Letterbox)':>18} {'Delta':>12}\n")
        f.write("-" * 74 + "\n")
        f.write(f"{'Global Dice':<24} {'0.3609':>18} {best_metrics.get('dice', float('nan')):>18.4f} {best_metrics.get('dice', 0) - 0.3609:>+12.4f}\n")
        f.write(f"{'Macro Dice':<24} {'0.3191':>18} {best_metrics.get('macro_dice', float('nan')):>18.4f} {best_metrics.get('macro_dice', 0) - 0.3191:>+12.4f}\n")
        f.write(f"{'Global IoU':<24} {'0.2201':>18} {best_metrics.get('iou', float('nan')):>18.4f} {best_metrics.get('iou', 0) - 0.2201:>+12.4f}\n")
        f.write(f"{'Sensitivity (Recall)':<24} {'0.2782':>18} {best_metrics.get('sensitivity', float('nan')):>18.4f} {best_metrics.get('sensitivity', 0) - 0.2782:>+12.4f}\n")
        f.write(f"{'Specificity':<24} {'0.9991':>18} {best_metrics.get('specificity', float('nan')):>18.4f} {best_metrics.get('specificity', 0) - 0.9991:>+12.4f}\n")
        f.write(f"{'Precision':<24} {'0.5133':>18} {best_metrics.get('precision', float('nan')):>18.4f} {best_metrics.get('precision', 0) - 0.5133:>+12.4f}\n")
        f.write(f"{'Pred / GT Ratio':<24} {'0.5420':>18} {best_metrics.get('pred_to_gt_ratio', float('nan')):>18.4f} {best_metrics.get('pred_to_gt_ratio', 0) - 0.5420:>+12.4f}\n")

    print(f"\nEXP-1 training complete. Summary report saved: {report_path}")


if __name__ == "__main__":
    main()
