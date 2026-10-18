"""
Phase 2: DR Local Binary Lesion Segmentation Baseline
=====================================================
Experiment: DR Local Binary Lesion Segmentation Baseline

Trains a U-Net (SharedUNet architecture) on DDR pixel-level lesion segmentation
using a binary all-lesion mask (background=0, any DR lesion=1).

Architecture reused:   src/shared_unet.py  (SharedUNet -> SharedEncoder + decoder)
Dataset loader reused: src/dr_segmentation_dataset.py (DDRLesionDataset)

Output artifacts:
  results/DR/checkpoints/best_dr_unet.pth        -- best validation Dice checkpoint
  results/DR/checkpoints/final_dr_unet.pth       -- final epoch checkpoint
  results/DR/dr_local_training_curves.png        -- loss + Dice curves
  results/DR/dr_local_metric_curves.png          -- all val metrics curves
  results/DR/validation_predictions/             -- qualitative prediction panels
  results/DR/dr_local_test_metrics.txt           -- test set evaluation results
"""

import os
import sys
import random
import time
from pathlib import Path

# Allow importing project modules from src/
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dr_segmentation_dataset import DDRLesionDataset
from shared_unet import SharedUNet


# ===========================================================================
# Configuration — all hyper-parameters in one place for reproducibility
# ===========================================================================
CFG = {
    # Experiment identity
    "experiment_name": "DR Local Binary Lesion Segmentation Baseline",
    "random_seed": 42,

    # Data
    "image_size": 512,
    "target_lesion": "all",        # binary union of EX, HE, MA, SE

    # Training
    "batch_size": 2,
    "epochs": 20,
    "learning_rate": 1e-4,
    "loss": "bce+dice",
    "threshold": 0.5,

    # I/O
    "results_dir": "results/DR",
    "checkpoint_dir": "results/DR/checkpoints",
    "pred_dir": "results/DR/validation_predictions",
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
# Loss functions
# ===========================================================================
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss computed over each sample in the batch, then averaged."""
    probs = torch.sigmoid(logits)
    # Flatten spatial dims per sample
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE + Dice loss for binary segmentation under heavy class imbalance."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    d_loss = dice_loss(logits, targets)
    return bce + d_loss


# ===========================================================================
# Evaluation metrics
# ===========================================================================
@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> dict:
    """
    Computes binary segmentation metrics for a single batch or accumulated tensors.

    Returns Dice, IoU, Sensitivity (Recall), Specificity.
    Operates on raw logits + targets in range {0, 1}.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
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

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
    }


# ===========================================================================
# Sanity check
# ===========================================================================
def run_sanity_check(model: nn.Module, device: torch.device, dataset: DDRLesionDataset) -> None:
    """
    Verifies shapes, loss finiteness, and gradient flow before full training.
    Raises AssertionError if any check fails.
    """
    print("\n--- Sanity Check ---")

    # 1. Dataset output shapes and value ranges
    img, mask = dataset[0]
    assert img.shape == (3, 512, 512), f"Unexpected image shape: {img.shape}"
    assert mask.shape == (1, 512, 512), f"Unexpected mask shape: {mask.shape}"
    assert img.min() >= 0.0 and img.max() <= 1.0, "Image values outside [0,1]"
    assert set(mask.unique().tolist()).issubset({0.0, 1.0}), f"Unexpected mask values: {mask.unique()}"
    print(f"  [OK] Dataset shapes: image={list(img.shape)}, mask={list(mask.shape)}")
    print(f"  [OK] Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  [OK] Mask unique values: {mask.unique().tolist()}")

    # 2. Forward pass
    model.eval()
    x = img.unsqueeze(0).to(device)
    y = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 1, 512, 512), f"Unexpected output shape: {logits.shape}"
    print(f"  [OK] Forward pass output shape: {list(logits.shape)}")

    # 3. Loss is finite
    loss = combined_loss(logits, y)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"  [OK] Loss is finite: {loss.item():.4f}")

    # 4. One training step — check gradients are finite
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    logits = model(x)
    loss = combined_loss(logits, y)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    optimizer.step()
    print("  [OK] Gradients are finite after backward pass")
    print("--- Sanity Check PASSED ---\n")


# ===========================================================================
# Training one epoch
# ===========================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


# ===========================================================================
# Validation one epoch
# ===========================================================================
@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0
    # Accumulate raw predictions and targets across batches for global metrics
    all_logits = []
    all_targets = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = combined_loss(logits, masks)
        running_loss += loss.item()

        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_logits, all_targets, threshold=CFG["threshold"])
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


# ===========================================================================
# Plotting
# ===========================================================================
def save_training_curves(
    history: dict,
    output_dir: Path,
) -> None:
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # --- Loss + Dice curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax1.set_title("Training & Validation Loss (BCE + Dice)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_range, history["val_dice"], label="Val Dice", color="green", marker="o", markersize=3)
    ax2.set_title("Validation Dice Score", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("DR Local Binary Lesion Segmentation Baseline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "dr_local_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves: {out_path}")

    # --- All metrics curve ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs_range, history["val_dice"], color="green", marker="o", markersize=3)
    axes[0, 0].set_title("Val Dice", fontweight="bold")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs_range, history["val_iou"], color="blue", marker="o", markersize=3)
    axes[0, 1].set_title("Val IoU", fontweight="bold")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs_range, history["val_sensitivity"], color="orange", marker="o", markersize=3)
    axes[1, 0].set_title("Val Sensitivity (Recall)", fontweight="bold")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs_range, history["val_specificity"], color="purple", marker="o", markersize=3)
    axes[1, 1].set_title("Val Specificity", fontweight="bold")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    plt.suptitle("DR Baseline — Validation Metrics per Epoch", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "dr_local_metric_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric curves: {out_path}")


# ===========================================================================
# Qualitative validation predictions
# ===========================================================================
@torch.no_grad()
def save_validation_predictions(
    model: nn.Module,
    dataset: DDRLesionDataset,
    device: torch.device,
    pred_dir: Path,
    num_samples: int = 6,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Pick samples with visible lesions for informative visualizations
    selected = []
    for idx in range(len(dataset)):
        img_t, mask_t = dataset[idx]
        if mask_t.sum() > 100:   # Has meaningful lesion area at 512×512
            selected.append((idx, img_t, mask_t))
        if len(selected) >= num_samples:
            break

    print(f"Generating {len(selected)} validation prediction panels...")

    for idx, img_t, gt_mask_t in selected:
        info = dataset.get_sample_info(idx)
        img_id = info["image_id"]

        # Run model
        logits = model(img_t.unsqueeze(0).to(device))
        pred_mask = (torch.sigmoid(logits.cpu()) > CFG["threshold"]).float().squeeze()

        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_np = gt_mask_t.squeeze(0).numpy()
        pred_np = pred_mask.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

        # 1. Original fundus
        axes[0].imshow(img_np)
        axes[0].set_title(f"Fundus Image\n{img_id}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        # 2. Ground truth mask
        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(
            f"Ground Truth Mask\n({int(gt_np.sum())} lesion px)", fontsize=10, fontweight="bold"
        )
        axes[1].axis("off")

        # 3. Predicted mask
        axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(
            f"Predicted Mask\n({int(pred_np.sum())} lesion px)", fontsize=10, fontweight="bold"
        )
        axes[2].axis("off")

        # 4. Overlay: TP=green, FP=blue, FN=red on fundus
        axes[3].imshow(img_np)
        overlay = np.zeros((*gt_np.shape, 4), dtype=np.uint8)
        tp_mask = (gt_np > 0) & (pred_np > 0)
        fp_mask = (gt_np == 0) & (pred_np > 0)
        fn_mask = (gt_np > 0) & (pred_np == 0)
        overlay[tp_mask] = [0, 220, 80, 200]    # Green  = TP
        overlay[fp_mask] = [50, 120, 255, 180]  # Blue   = FP
        overlay[fn_mask] = [255, 40, 40, 200]   # Red    = FN
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay: TP=Green FP=Blue FN=Red", fontsize=10, fontweight="bold")
        axes[3].axis("off")

        # Per-sample metrics
        m = compute_metrics(logits.cpu(), gt_mask_t.unsqueeze(0))
        fig.suptitle(
            f"Dice={m['dice']:.3f}  IoU={m['iou']:.3f}  "
            f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}",
            fontsize=11,
        )

        plt.tight_layout()
        save_path = pred_dir / f"val_pred_{img_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(selected)} prediction panels to {pred_dir}")


# ===========================================================================
# Test set evaluation
# ===========================================================================
@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    results_dir: Path,
) -> dict:
    model.eval()
    all_logits = []
    all_targets = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets, threshold=CFG["threshold"])

    print("\nTest Set Results:")
    print(f"  Dice:        {metrics['dice']:.4f}")
    print(f"  IoU:         {metrics['iou']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")

    # Save to text file
    txt_path = results_dir / "dr_local_test_metrics.txt"
    with open(txt_path, "w") as f:
        f.write("DR Local Binary Lesion Segmentation Baseline — Test Set Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment: {CFG['experiment_name']}\n")
        f.write(f"Model:      SharedUNet (SharedEncoder + U-Net decoder)\n")
        f.write(f"Image size: {CFG['image_size']}x{CFG['image_size']}\n")
        f.write(f"Threshold:  {CFG['threshold']}\n")
        f.write(f"Lesion mask: binary union of EX, HE, MA, SE\n\n")
        f.write("Metric       | Value\n")
        f.write("-------------|-------\n")
        f.write(f"Dice         | {metrics['dice']:.4f}\n")
        f.write(f"IoU          | {metrics['iou']:.4f}\n")
        f.write(f"Sensitivity  | {metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity  | {metrics['specificity']:.4f}\n")
    print(f"  Test metrics saved to: {txt_path}")
    return metrics


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("=" * 70)
    print(f"EXPERIMENT: {CFG['experiment_name']}")
    print("=" * 70)

    # -- Seed --
    set_seed(CFG["random_seed"])
    print(f"Random seed: {CFG['random_seed']}")

    # -- Device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem_gb:.1f} GB")

    # -- Directories --
    results_dir = Path(CFG["results_dir"])
    ckpt_dir = Path(CFG["checkpoint_dir"])
    pred_dir = Path(CFG["pred_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # -- Datasets --
    print("\nLoading datasets...")
    train_ds = DDRLesionDataset(split="train", image_size=CFG["image_size"])
    valid_ds = DDRLesionDataset(split="valid", image_size=CFG["image_size"])
    test_ds = DDRLesionDataset(split="test", image_size=CFG["image_size"])

    print(f"  Train: {len(train_ds)} images")
    print(f"  Valid: {len(valid_ds)} images")
    print(f"  Test:  {len(test_ds)} images")

    # -- DataLoaders --
    num_workers = 0  # Windows: multiprocessing with PIL can be fragile
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -- Model --
    model = SharedUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SharedUNet")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Input:  [B, 3, 512, 512] -> Output: [B, 1, 512, 512]")

    # -- Sanity Check --
    run_sanity_check(model, device, train_ds)

    # -- Optimizer --
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])

    # -- Training configuration summary --
    print("Training Configuration:")
    print(f"  Image size:     {CFG['image_size']}x{CFG['image_size']}")
    print(f"  Batch size:     {CFG['batch_size']}")
    print(f"  Epochs:         {CFG['epochs']}")
    print(f"  Learning rate:  {CFG['learning_rate']}")
    print(f"  Loss:           {CFG['loss']}")
    print(f"  Threshold:      {CFG['threshold']}")
    print(f"  Batches/epoch:  {len(train_loader)}")
    print()

    # -- Training loop --
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_sensitivity": [],
        "val_specificity": [],
    }

    best_val_dice = 0.0
    best_epoch = 0
    best_metrics = {}
    training_start = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate_epoch(model, valid_loader, device)

        val_loss = val_metrics["loss"]
        val_dice = val_metrics["dice"]
        val_iou = val_metrics["iou"]
        val_sensitivity = val_metrics["sensitivity"]
        val_specificity = val_metrics["specificity"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_sensitivity"].append(val_sensitivity)
        history["val_specificity"].append(val_specificity)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{CFG['epochs']:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Sensitivity: {val_sensitivity:.4f} | "
            f"Specificity: {val_specificity:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpoint: best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            best_metrics = {k: v for k, v in val_metrics.items()}
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_sensitivity": val_sensitivity,
                    "val_specificity": val_specificity,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "cfg": CFG,
                },
                ckpt_dir / "best_dr_unet.pth",
            )
            print(f"  -> New best checkpoint saved (Val Dice: {val_dice:.4f})")

    total_time = time.time() - training_start

    # Save final checkpoint
    torch.save(
        {
            "epoch": CFG["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": history["val_dice"][-1],
            "val_iou": history["val_iou"][-1],
            "val_loss": history["val_loss"][-1],
            "train_loss": history["train_loss"][-1],
            "cfg": CFG,
        },
        ckpt_dir / "final_dr_unet.pth",
    )

    # -- Summary --
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total training time:   {total_time / 60:.1f} minutes")
    print(f"  Best epoch:            {best_epoch} / {CFG['epochs']}")
    print(f"  Best Val Dice:         {best_val_dice:.4f}")
    print(f"  Best Val IoU:          {best_metrics.get('iou', float('nan')):.4f}")
    print(f"  Best Val Sensitivity:  {best_metrics.get('sensitivity', float('nan')):.4f}")
    print(f"  Best Val Specificity:  {best_metrics.get('specificity', float('nan')):.4f}")
    print(f"  Final Train Loss:      {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss:        {history['val_loss'][-1]:.4f}")

    # -- Training curves --
    save_training_curves(history, results_dir)

    # -- Load best checkpoint for evaluation --
    print("\nLoading best checkpoint for evaluation...")
    best_ckpt = torch.load(ckpt_dir / "best_dr_unet.pth", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # -- Validation predictions --
    save_validation_predictions(model, valid_ds, device, pred_dir, num_samples=6)

    # -- Test evaluation --
    print("\nEvaluating on test set (best checkpoint)...")
    test_metrics = evaluate_on_test(model, test_loader, device, results_dir)

    # -- Final results table --
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"{'Metric':<14} {'Validation':>12} {'Test':>12}")
    print("-" * 40)
    metrics_to_show = ["dice", "iou", "sensitivity", "specificity"]
    for m in metrics_to_show:
        val_val = best_metrics.get(m, float("nan"))
        test_val = test_metrics.get(m, float("nan"))
        print(f"{m.capitalize():<14} {val_val:>12.4f} {test_val:>12.4f}")

    print("\nArtifacts saved:")
    print(f"  Checkpoints:          {ckpt_dir}")
    print(f"  Training curves:      {results_dir / 'dr_local_training_curves.png'}")
    print(f"  Metric curves:        {results_dir / 'dr_local_metric_curves.png'}")
    print(f"  Val predictions:      {pred_dir}")
    print(f"  Test metrics:         {results_dir / 'dr_local_test_metrics.txt'}")


if __name__ == "__main__":
    main()
