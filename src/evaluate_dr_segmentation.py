"""
Phase 3: DR Lesion Segmentation Evaluation Script
==================================================
Evaluates a trained DR segmentation model checkpoint on the DDR test set.
Generates per-image metrics and qualitative prediction panels.

Usage:
    python src/evaluate_dr_segmentation.py
    python src/evaluate_dr_segmentation.py --checkpoint results/DR/checkpoints/best_dr_unet.pth
"""

import argparse
import sys
from pathlib import Path

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
# Metrics
# ===========================================================================
@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> dict:
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
        "tp": int(tp.item()),
        "fp": int(fp.item()),
        "fn": int(fn.item()),
        "tn": int(tn.item()),
    }


# ===========================================================================
# Per-image evaluation
# ===========================================================================
@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataset: DDRLesionDataset,
    device: torch.device,
    threshold: float = 0.5,
    split_name: str = "test",
) -> tuple[dict, list]:
    """Returns global metrics dict and per-image metrics list."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    per_image = []
    all_logits = []
    all_targets = []

    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)

        # Per-image metrics
        img_metrics = compute_metrics(logits.cpu(), masks.cpu(), threshold=threshold)
        info = dataset.get_sample_info(i)
        img_metrics["image_id"] = info["image_id"]
        per_image.append(img_metrics)

        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    global_metrics = compute_metrics(all_logits, all_targets, threshold=threshold)

    print(f"\n{split_name.upper()} Evaluation ({len(dataset)} images):")
    print(f"  Dice:        {global_metrics['dice']:.4f}")
    print(f"  IoU:         {global_metrics['iou']:.4f}")
    print(f"  Sensitivity: {global_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {global_metrics['specificity']:.4f}")

    return global_metrics, per_image


# ===========================================================================
# Save prediction panels
# ===========================================================================
@torch.no_grad()
def save_prediction_panels(
    model: nn.Module,
    dataset: DDRLesionDataset,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    threshold: float = 0.5,
    num_samples: int = 8,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select samples with visible lesions
    selected = []
    for idx in range(len(dataset)):
        img_t, mask_t = dataset[idx]
        if mask_t.sum() > 100:
            selected.append((idx, img_t, mask_t))
        if len(selected) >= num_samples:
            break

    for idx, img_t, gt_mask_t in selected:
        info = dataset.get_sample_info(idx)
        img_id = info["image_id"]

        logits = model(img_t.unsqueeze(0).to(device))
        pred_mask = (torch.sigmoid(logits.cpu()) > threshold).float().squeeze()

        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_np = gt_mask_t.squeeze(0).numpy()
        pred_np = pred_mask.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

        axes[0].imshow(img_np)
        axes[0].set_title(f"Fundus\n{img_id}", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Ground Truth\n({int(gt_np.sum())} px)", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"Predicted\n({int(pred_np.sum())} px)", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(img_np)
        overlay = np.zeros((*gt_np.shape, 4), dtype=np.uint8)
        overlay[(gt_np > 0) & (pred_np > 0)] = [0, 220, 80, 200]
        overlay[(gt_np == 0) & (pred_np > 0)] = [50, 120, 255, 180]
        overlay[(gt_np > 0) & (pred_np == 0)] = [255, 40, 40, 200]
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay\nGreen=TP  Blue=FP  Red=FN", fontsize=9, fontweight="bold")
        axes[3].axis("off")

        m = compute_metrics(logits.cpu(), gt_mask_t.unsqueeze(0))
        fig.suptitle(
            f"[{split_name.upper()}] Dice={m['dice']:.3f}  IoU={m['iou']:.3f}  "
            f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}",
            fontsize=11,
        )

        plt.tight_layout()
        save_path = output_dir / f"{split_name}_pred_{img_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(selected)} panels to {output_dir}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DR Lesion Segmentation Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/DR/checkpoints/best_dr_unet.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run train_dr_segmentation.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    # Load model
    model = SharedUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Saved val Dice:   {ckpt.get('val_dice', float('nan')):.4f}")

    results_dir = Path("results/DR")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Validate set
    valid_ds = DDRLesionDataset(split="valid", image_size=args.image_size)
    val_metrics, _ = evaluate_dataset(model, valid_ds, device, args.threshold, "valid")
    save_prediction_panels(model, valid_ds, device, results_dir / "validation_predictions", "val",
                           args.threshold, num_samples=8)

    # Test set
    test_ds = DDRLesionDataset(split="test", image_size=args.image_size)
    test_metrics, _ = evaluate_dataset(model, test_ds, device, args.threshold, "test")
    save_prediction_panels(model, test_ds, device, results_dir / "test_predictions", "test",
                           args.threshold, num_samples=8)

    # Save combined results
    out_path = results_dir / "dr_local_test_metrics.txt"
    with open(out_path, "w") as f:
        f.write("DR Local Binary Lesion Segmentation Baseline — Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {ckpt_path}\n\n")
        f.write(f"{'Metric':<14} {'Validation':>12} {'Test':>12}\n")
        f.write("-" * 40 + "\n")
        for key in ["dice", "iou", "sensitivity", "specificity"]:
            f.write(f"{key.capitalize():<14} {val_metrics[key]:>12.4f} {test_metrics[key]:>12.4f}\n")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
