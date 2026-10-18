"""
Comprehensive Diagnostic Audit of the FedMorphis DR Binary Lesion-Segmentation Baseline.
Performs:
1. Ground-truth vs predicted lesion-pixel fractions on validation set.
2. Dataset-wide lesion composition and empty mask analysis (train, valid, test).
3. Metric implementation analysis: global pooled vs macro-averaged, empty mask behavior.
4. Resizing, aspect ratio distortion, and nearest-neighbor lesion erasure analysis.
5. Validation-only threshold sweep [0.01 to 0.95] without retraining.
6. Diagnostic plotting (threshold curves, PR curve, ROC curve, pixel fraction comparisons).
"""
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dr_segmentation_dataset import DDRLesionDataset, LESION_TYPES
from shared_unet import SharedUNet
from evaluate_dr_segmentation import compute_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    diag_dir = REPO_ROOT / "results" / "DR" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = REPO_ROOT / "results" / "DR" / "checkpoints" / "best_dr_unet.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    model = SharedUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}")
    print(f"Saved checkpoint val Dice: {ckpt.get('val_dice'):.4f}, val Sens: {ckpt.get('val_sensitivity'):.4f}")

    # =========================================================================
    # PART A: DATASET INTEGRITY & RAW VS RESIZED LESION STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART A: DATASET INTEGRITY & RESIZING AUDIT")
    print("=" * 80)
    
    splits_data = {}
    for split_name in ["train", "valid", "test"]:
        ds = DDRLesionDataset(split=split_name, image_size=512)
        splits_data[split_name] = ds
        print(f"Split '{split_name}': {len(ds)} images")

    # Detailed inspection of validation split
    val_ds = splits_data["valid"]
    val_raw_aspect_ratios = []
    val_raw_sizes = []
    val_raw_lesion_px = {l: 0 for l in LESION_TYPES}
    val_raw_total_lesion_px = 0
    val_raw_total_px = 0
    val_empty_gt_count = 0
    val_gt_px_list = []
    
    # Track lesion presence counts
    val_lesion_img_counts = {l: 0 for l in LESION_TYPES}
    
    for i in range(len(val_ds)):
        info = val_ds.get_sample_info(i)
        with Image.open(info["image_path"]) as img:
            w, h = img.size
            val_raw_sizes.append((w, h))
            val_raw_aspect_ratios.append(w / h)
            val_raw_total_px += (w * h)
            
        combined_raw = None
        for l in LESION_TYPES:
            m_path = info["mask_paths"][l]
            if m_path.exists():
                arr = np.array(Image.open(m_path))
                pos = int(np.sum(arr > 0))
                if pos > 0:
                    val_raw_lesion_px[l] += pos
                    val_lesion_img_counts[l] += 1
                if combined_raw is None:
                    combined_raw = (arr > 0)
                else:
                    combined_raw = np.logical_or(combined_raw, arr > 0)
        
        if combined_raw is not None:
            raw_pos = int(np.sum(combined_raw))
        else:
            raw_pos = 0
        val_raw_total_lesion_px += raw_pos
        
        # Check 512x512 mask
        _, mask_t = val_ds[i]
        gt_512_pos = int(mask_t.sum().item())
        val_gt_px_list.append(gt_512_pos)
        if gt_512_pos == 0:
            val_empty_gt_count += 1

    unique_sizes = set(val_raw_sizes)
    print(f"Validation raw image resolutions found ({len(unique_sizes)} unique): {unique_sizes}")
    print(f"Aspect ratios: min={min(val_raw_aspect_ratios):.2f}, max={max(val_raw_aspect_ratios):.2f}, mean={np.mean(val_raw_aspect_ratios):.2f}")
    print(f"Validation images with NO lesions (empty mask at 512x512): {val_empty_gt_count} / {len(val_ds)} ({val_empty_gt_count/len(val_ds)*100:.1f}%)")
    print(f"Validation images WITH lesions: {len(val_ds) - val_empty_gt_count} / {len(val_ds)} ({(len(val_ds) - val_empty_gt_count)/len(val_ds)*100:.1f}%)")
    print(f"Validation images containing each lesion type:")
    for l in LESION_TYPES:
        print(f"  {l}: {val_lesion_img_counts[l]} images ({val_lesion_img_counts[l]/len(val_ds)*100:.1f}%) | Raw px: {val_raw_lesion_px[l]:,}")
    print(f"Raw total pixels: {val_raw_total_px:,} | Raw total lesion pixels: {val_raw_total_lesion_px:,}")
    print(f"Raw ground-truth lesion pixel fraction: {val_raw_total_lesion_px / val_raw_total_px * 100:.4f}%")

    # =========================================================================
    # PART B: INFERENCE ON VALIDATION SET & LESION PIXEL FRACTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B: VALIDATION SET INFERENCE & PIXEL FRACTION ANALYSIS")
    print("=" * 80)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    all_logits = []
    all_targets = []
    per_image_stats = []
    
    total_val_pixels = len(val_ds) * 512 * 512
    total_val_gt_px = 0
    total_val_pred_px_05 = 0
    
    with torch.no_grad():
        for i, (img, mask) in enumerate(val_loader):
            img = img.to(device)
            logit = model(img).cpu() # [1, 1, 512, 512]
            prob = torch.sigmoid(logit)
            
            all_logits.append(logit)
            all_targets.append(mask)
            
            gt_px = int(mask.sum().item())
            pred_px_05 = int((prob > 0.5).sum().item())
            
            total_val_gt_px += gt_px
            total_val_pred_px_05 += pred_px_05
            
            info = val_ds.get_sample_info(i)
            per_image_stats.append({
                "image_id": info["image_id"],
                "gt_px": gt_px,
                "pred_px_05": pred_px_05,
                "gt_fraction": gt_px / (512 * 512),
                "pred_fraction_05": pred_px_05 / (512 * 512),
                "ratio_pred_to_gt": pred_px_05 / max(gt_px, 1) if gt_px > 0 else np.nan,
                "prob_max": float(prob.max().item()),
                "prob_mean": float(prob.mean().item()),
            })

    all_logits_tensor = torch.cat(all_logits, dim=0) # [149, 1, 512, 512]
    all_targets_tensor = torch.cat(all_targets, dim=0) # [149, 1, 512, 512]
    
    val_gt_fraction = total_val_gt_px / total_val_pixels
    val_pred_fraction_05 = total_val_pred_px_05 / total_val_pixels
    
    print(f"Total validation pixels (149 x 512 x 512): {total_val_pixels:,}")
    print(f"Ground-Truth lesion pixels at 512x512:     {total_val_gt_px:,} ({val_gt_fraction * 100:.4f}% of all pixels)")
    print(f"Predicted lesion pixels (@ thr=0.5):      {total_val_pred_px_05:,} ({val_pred_fraction_05 * 100:.4f}% of all pixels)")
    print(f"Ratio of Predicted to Ground-Truth:       {total_val_pred_px_05 / total_val_gt_px:.4f}")
    print(f"Severe under-prediction factor:           {total_val_gt_px / max(total_val_pred_px_05, 1):.2f}x under-prediction at threshold 0.5")

    # False positive rate on empty validation images
    empty_img_preds = [s["pred_px_05"] for s in per_image_stats if s["gt_px"] == 0]
    nonempty_img_preds = [s["pred_px_05"] for s in per_image_stats if s["gt_px"] > 0]
    print(f"\nOn {len(empty_img_preds)} empty validation images (GT = 0):")
    print(f"  Empty images with 0 predicted lesion px: {sum(p == 0 for p in empty_img_preds)} / {len(empty_img_preds)}")
    print(f"  Total false positive pixels on empty images: {sum(empty_img_preds)}")
    print(f"On {len(nonempty_img_preds)} lesion-positive validation images (GT > 0):")
    print(f"  Images with 0 predicted lesion px (completely missed): {sum(p == 0 for p in nonempty_img_preds)} / {len(nonempty_img_preds)}")

    # =========================================================================
    # PART C: METRIC IMPLEMENTATION & AGGREGATION AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART C: METRIC IMPLEMENTATION AUDIT")
    print("=" * 80)
    
    # 1. Global pooled metrics (as done in baseline)
    global_05 = compute_metrics(all_logits_tensor, all_targets_tensor, threshold=0.5)
    print(f"1. Global (Dataset-Pooled) Metrics @ threshold=0.5:")
    print(f"   Dice:        {global_05['dice']:.4f}")
    print(f"   IoU:         {global_05['iou']:.4f}")
    print(f"   Sensitivity: {global_05['sensitivity']:.4f}")
    print(f"   Specificity: {global_05['specificity']:.4f}")
    print(f"   TP: {global_05['tp']:,}  FP: {global_05['fp']:,}  FN: {global_05['fn']:,}  TN: {global_05['tn']:,}")
    
    # 2. Per-image metrics (Macro-averaged) across ALL images
    per_img_all = [compute_metrics(all_logits_tensor[i:i+1], all_targets_tensor[i:i+1], threshold=0.5) for i in range(len(val_ds))]
    macro_dice_all = np.mean([m["dice"] for m in per_img_all])
    macro_iou_all = np.mean([m["iou"] for m in per_img_all])
    macro_sens_all = np.mean([m["sensitivity"] for m in per_img_all])
    macro_spec_all = np.mean([m["specificity"] for m in per_img_all])
    print(f"\n2. Macro-Averaged (Mean Per-Image) across ALL 149 images @ threshold=0.5:")
    print(f"   Dice:        {macro_dice_all:.4f}")
    print(f"   IoU:         {macro_iou_all:.4f}")
    print(f"   Sensitivity: {macro_sens_all:.4f}")
    print(f"   Specificity: {macro_spec_all:.4f}")

    # 3. Macro-averaged across ONLY images with ground-truth lesions
    per_img_pos = [compute_metrics(all_logits_tensor[i:i+1], all_targets_tensor[i:i+1], threshold=0.5) for i in range(len(val_ds)) if all_targets_tensor[i].sum() > 0]
    macro_dice_pos = np.mean([m["dice"] for m in per_img_pos])
    macro_iou_pos = np.mean([m["iou"] for m in per_img_pos])
    macro_sens_pos = np.mean([m["sensitivity"] for m in per_img_pos])
    macro_spec_pos = np.mean([m["specificity"] for m in per_img_pos])
    print(f"\n3. Macro-Averaged across ONLY {len(per_img_pos)} Lesion-Positive Images @ threshold=0.5:")
    print(f"   Dice:        {macro_dice_pos:.4f}")
    print(f"   IoU:         {macro_iou_pos:.4f}")
    print(f"   Sensitivity: {macro_sens_pos:.4f}")
    print(f"   Specificity: {macro_spec_pos:.4f}")

    # =========================================================================
    # PART D: VALIDATION-ONLY THRESHOLD SWEEP
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART D: VALIDATION-ONLY THRESHOLD SWEEP (NO RETRAINING)")
    print("=" * 80)
    
    thresholds = np.concatenate([
        np.array([0.01, 0.02, 0.03, 0.04]),
        np.linspace(0.05, 0.95, 19),
        np.array([0.98, 0.99])
    ])
    thresholds = np.unique(np.round(thresholds, 4))
    thresholds.sort()

    sweep_results = []
    
    all_probs = torch.sigmoid(all_logits_tensor)
    flat_probs = all_probs.view(-1)
    flat_targets = all_targets_tensor.view(-1)

    for thr in thresholds:
        m_glob = compute_metrics(all_logits_tensor, all_targets_tensor, threshold=float(thr))
        
        # Predicted pixels at this threshold
        pred_px = m_glob["tp"] + m_glob["fp"]
        pred_frac = pred_px / total_val_pixels
        
        # Macro metrics on lesion-positive images
        m_pos_list = [compute_metrics(all_logits_tensor[i:i+1], all_targets_tensor[i:i+1], threshold=float(thr)) for i in range(len(val_ds)) if all_targets_tensor[i].sum() > 0]
        macro_dice = np.mean([m["dice"] for m in m_pos_list])
        macro_sens = np.mean([m["sensitivity"] for m in m_pos_list])
        
        # Precision
        precision = (m_glob["tp"] + 1e-6) / (m_glob["tp"] + m_glob["fp"] + 1e-6)
        
        sweep_results.append({
            "threshold": float(thr),
            "global_dice": m_glob["dice"],
            "global_iou": m_glob["iou"],
            "global_sensitivity": m_glob["sensitivity"],
            "global_specificity": m_glob["specificity"],
            "precision": float(precision),
            "macro_dice_pos": float(macro_dice),
            "macro_sens_pos": float(macro_sens),
            "pred_pixels": pred_px,
            "pred_fraction": pred_frac,
            "ratio_pred_to_gt": pred_px / total_val_gt_px,
            "tp": m_glob["tp"],
            "fp": m_glob["fp"],
            "fn": m_glob["fn"],
            "tn": m_glob["tn"],
        })

    # Print summary table of selected key thresholds
    print(f"{'Threshold':<10} {'Global Dice':<12} {'Global IoU':<12} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<12} {'Pred/GT':<10}")
    print("-" * 82)
    key_thrs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    for r in sweep_results:
        if any(abs(r["threshold"] - kt) < 1e-4 for kt in key_thrs):
            print(f"{r['threshold']:<10.2f} {r['global_dice']:<12.4f} {r['global_iou']:<12.4f} {r['global_sensitivity']:<12.4f} {r['global_specificity']:<12.4f} {r['precision']:<12.4f} {r['ratio_pred_to_gt']:<10.3f}")

    best_dice_res = max(sweep_results, key=lambda x: x["global_dice"])
    print(f"\nOptimal Validation Threshold for Global Dice: {best_dice_res['threshold']:.2f}")
    print(f"  Validation Dice at {best_dice_res['threshold']:.2f}:        {best_dice_res['global_dice']:.4f} (vs {global_05['dice']:.4f} @ 0.5, +{best_dice_res['global_dice'] - global_05['dice']:.4f})")
    print(f"  Validation IoU at {best_dice_res['threshold']:.2f}:         {best_dice_res['global_iou']:.4f} (vs {global_05['iou']:.4f} @ 0.5)")
    print(f"  Validation Sensitivity at {best_dice_res['threshold']:.2f}: {best_dice_res['global_sensitivity']:.4f} (vs {global_05['sensitivity']:.4f} @ 0.5, +{best_dice_res['global_sensitivity'] - global_05['sensitivity']:.4f})")
    print(f"  Validation Specificity at {best_dice_res['threshold']:.2f}: {best_dice_res['global_specificity']:.4f} (vs {global_05['specificity']:.4f} @ 0.5)")
    print(f"  Predicted / GT ratio at {best_dice_res['threshold']:.2f}:   {best_dice_res['ratio_pred_to_gt']:.3f} (vs {total_val_pred_px_05 / total_val_gt_px:.3f} @ 0.5)")

    # =========================================================================
    # PART E: DIAGNOSTIC PLOTS GENERATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART E: SAVING DIAGNOSTIC PLOTS")
    print("=" * 80)
    
    thrs = [r["threshold"] for r in sweep_results]
    dices = [r["global_dice"] for r in sweep_results]
    ious = [r["global_iou"] for r in sweep_results]
    senss = [r["global_sensitivity"] for r in sweep_results]
    specs = [r["global_specificity"] for r in sweep_results]
    precs = [r["precision"] for r in sweep_results]
    ratios = [r["ratio_pred_to_gt"] for r in sweep_results]

    # Plot 1: Validation Threshold Sweep Curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.plot(thrs, dices, "g-o", linewidth=2, markersize=4, label="Global Dice")
    ax1.plot(thrs, ious, "b--s", linewidth=1.5, markersize=3, label="Global IoU")
    ax1.axvline(0.5, color="red", linestyle=":", label="Default thr=0.5")
    ax1.axvline(best_dice_res["threshold"], color="green", linestyle="--", label=f"Best thr={best_dice_res['threshold']:.2f}")
    ax1.set_title("Dice & IoU vs Classification Threshold", fontweight="bold")
    ax1.set_xlabel("Sigmoid Threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(thrs, senss, "orange", marker="^", linewidth=2, markersize=4, label="Sensitivity (Recall)")
    ax2.plot(thrs, precs, "purple", marker="v", linewidth=2, markersize=4, label="Precision")
    ax2.axvline(0.5, color="red", linestyle=":", label="Default thr=0.5")
    ax2.axvline(best_dice_res["threshold"], color="green", linestyle="--", label=f"Best thr={best_dice_res['threshold']:.2f}")
    ax2.set_title("Sensitivity vs Precision Trade-off", fontweight="bold")
    ax2.set_xlabel("Sigmoid Threshold")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(thrs, specs, "m-d", linewidth=2, markersize=4, label="Specificity")
    ax3.axvline(0.5, color="red", linestyle=":", label="Default thr=0.5")
    ax3.set_title("Specificity vs Threshold", fontweight="bold")
    ax3.set_xlabel("Sigmoid Threshold")
    ax3.set_ylabel("Specificity")
    ax3.set_ylim(0.98, 1.0005)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(thrs, ratios, "teal", marker="x", linewidth=2, markersize=4, label="Pred/GT Pixel Ratio")
    ax4.axhline(1.0, color="gray", linestyle="-.", label="Perfect Calibration (1.0)")
    ax4.axvline(0.5, color="red", linestyle=":", label="Default thr=0.5")
    ax4.axvline(best_dice_res["threshold"], color="green", linestyle="--", label=f"Best thr={best_dice_res['threshold']:.2f}")
    ax4.set_title("Predicted / Ground-Truth Pixel Volume Ratio", fontweight="bold")
    ax4.set_xlabel("Sigmoid Threshold")
    ax4.set_ylabel("Ratio (Pred Px / GT Px)")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle("Validation-Only Diagnostic Threshold Sweep (Checkpoint: best_dr_unet.pth)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path1 = diag_dir / "val_threshold_sweep_curves.png"
    plt.savefig(plot_path1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path1}")

    # Plot 2: Precision-Recall Curve on Validation Set
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(senss, precs, "b-o", linewidth=2, markersize=4, label="Validation PR Curve")
    # Mark thr=0.5 and best threshold
    p05 = next(r for r in sweep_results if abs(r["threshold"] - 0.5) < 1e-4)
    ax.plot(p05["global_sensitivity"], p05["precision"], "ro", markersize=9, label=f"Threshold 0.5 (Dice={p05['global_dice']:.3f}, Sens={p05['global_sensitivity']:.3f})")
    ax.plot(best_dice_res["global_sensitivity"], best_dice_res["precision"], "go", markersize=9, label=f"Best Threshold {best_dice_res['threshold']:.2f} (Dice={best_dice_res['global_dice']:.3f}, Sens={best_dice_res['global_sensitivity']:.3f})")
    ax.set_title("Validation Precision-Recall Curve", fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path2 = diag_dir / "val_precision_recall_curve.png"
    plt.savefig(plot_path2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path2}")

    # Plot 3: Pixel fraction distribution & Under-prediction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    gt_fractions = [s["gt_fraction"] * 100 for s in per_image_stats if s["gt_px"] > 0]
    pred_fractions = [s["pred_fraction_05"] * 100 for s in per_image_stats if s["gt_px"] > 0]
    
    ax1.hist(gt_fractions, bins=25, color="steelblue", alpha=0.7, label=f"GT Lesion % (mean={np.mean(gt_fractions):.3f}%)")
    ax1.hist(pred_fractions, bins=25, color="crimson", alpha=0.5, label=f"Pred Lesion % @ 0.5 (mean={np.mean(pred_fractions):.3f}%)")
    ax1.set_title("Validation Lesion-Positive Image Pixel % Distributions", fontweight="bold")
    ax1.set_xlabel("Lesion Area (% of 512x512 Image)")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter of Pred px vs GT px
    gt_pxs = [s["gt_px"] for s in per_image_stats]
    pred_pxs = [s["pred_px_05"] for s in per_image_stats]
    ax2.scatter(gt_pxs, pred_pxs, color="navy", alpha=0.6, edgecolors="none")
    max_val = max(max(gt_pxs), max(pred_pxs))
    ax2.plot([0, max_val], [0, max_val], "k--", label="Identity (Pred = GT)")
    ax2.set_title("Validation Per-Image Pred vs GT Pixels (@ thr=0.5)", fontweight="bold")
    ax2.set_xlabel("Ground Truth Pixels")
    ax2.set_ylabel("Predicted Pixels")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path3 = diag_dir / "val_pixel_distributions.png"
    plt.savefig(plot_path3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path3}")

    print("\nDiagnostic audit calculations complete.")

if __name__ == "__main__":
    main()
