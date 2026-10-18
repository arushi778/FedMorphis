"""
Probability distribution analysis on the validation set for best_dr_unet.pth.
Analyzes what probabilities the model actually assigns to Ground Truth lesion pixels:
- Are false negative pixels near 0.49 (borderline), or are they near 0.00 (completely missed)?
- Also analyzes the impact of NEAREST downsampling on small lesion components (e.g. MAs).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from dr_segmentation_dataset import DDRLesionDataset, LESION_TYPES
from shared_unet import SharedUNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / "results" / "DR" / "checkpoints" / "best_dr_unet.pth"
    model = SharedUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = DDRLesionDataset(split="valid", image_size=512)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    gt_pixel_probs = []
    bg_pixel_probs_sample = [] # sample to keep memory bounded

    total_fn_probs = []
    total_tp_probs = []

    print("Analyzing predicted probability distributions on validation set...")
    
    with torch.no_grad():
        for i, (img, mask) in enumerate(val_loader):
            img = img.to(device)
            prob = torch.sigmoid(model(img)).cpu().squeeze() # [512, 512]
            mask = mask.squeeze() # [512, 512]

            gt_mask = (mask > 0)
            bg_mask = (mask == 0)

            if gt_mask.sum() > 0:
                probs_on_gt = prob[gt_mask].numpy()
                gt_pixel_probs.extend(probs_on_gt)

                # Probabilities on TP and FN (at threshold 0.5)
                tp_probs = probs_on_gt[probs_on_gt > 0.5]
                fn_probs = probs_on_gt[probs_on_gt <= 0.5]
                total_tp_probs.extend(tp_probs)
                total_fn_probs.extend(fn_probs)

            # Sample 1000 background pixels per image
            bg_sample = prob[bg_mask].numpy()
            if len(bg_sample) > 1000:
                bg_sample = np.random.choice(bg_sample, size=1000, replace=False)
            bg_pixel_probs_sample.extend(bg_sample)

    gt_pixel_probs = np.array(gt_pixel_probs)
    total_fn_probs = np.array(total_fn_probs)
    total_tp_probs = np.array(total_tp_probs)
    bg_pixel_probs_sample = np.array(bg_pixel_probs_sample)

    print("\n" + "=" * 80)
    print("PREDICTED PROBABILITIES ON GROUND TRUTH LESION PIXELS (125,782 total GT px)")
    print("=" * 80)
    print(f"Mean probability on GT pixels:   {np.mean(gt_pixel_probs):.4f}")
    print(f"Median probability on GT pixels: {np.median(gt_pixel_probs):.4f}")
    print(f"25th percentile:                 {np.percentile(gt_pixel_probs, 25):.4f}")
    print(f"75th percentile:                 {np.percentile(gt_pixel_probs, 75):.4f}")
    print(f"90th percentile:                 {np.percentile(gt_pixel_probs, 90):.4f}")

    print("\nProbability Bin Distribution on GT Lesion Pixels:")
    bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(gt_pixel_probs, bins=bins)
    for i in range(len(counts)):
        pct = counts[i] / len(gt_pixel_probs) * 100
        print(f"  [{bins[i]:4.2f} - {bins[i+1]:4.2f}): {counts[i]:6d} px ({pct:5.2f}%)")

    print("\n" + "=" * 80)
    print("ANATOMY OF FALSE NEGATIVES (FN Pixels where GT=1 but Pred <= 0.5)")
    print("=" * 80)
    print(f"Total FN pixels: {len(total_fn_probs):,}")
    print(f"FN pixels with prob < 0.01:  {np.sum(total_fn_probs < 0.01):,} ({np.mean(total_fn_probs < 0.01)*100:.2f}%)")
    print(f"FN pixels with prob < 0.05:  {np.sum(total_fn_probs < 0.05):,} ({np.mean(total_fn_probs < 0.05)*100:.2f}%)")
    print(f"FN pixels with prob < 0.10:  {np.sum(total_fn_probs < 0.10):,} ({np.mean(total_fn_probs < 0.10)*100:.2f}%)")
    print(f"FN pixels with prob [0.10, 0.50]: {np.sum((total_fn_probs >= 0.10) & (total_fn_probs <= 0.50)):,} ({np.mean((total_fn_probs >= 0.10) & (total_fn_probs <= 0.50))*100:.2f}%)")

    print("\n" + "=" * 80)
    print("PREDICTED PROBABILITIES ON BACKGROUND PIXELS (GT = 0)")
    print("=" * 80)
    print(f"Mean probability on background:   {np.mean(bg_pixel_probs_sample):.6f}")
    print(f"99th percentile on background:    {np.percentile(bg_pixel_probs_sample, 99):.6f}")
    print(f"99.9th percentile on background:  {np.percentile(bg_pixel_probs_sample, 99.9):.6f}")

    # =========================================================================
    # PART F: LESION COMPONENT ERASURE ANALYSIS (RAW VS 512x512 NEAREST)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART F: NEAREST-NEIGHBOR DOWNSAMPLING LESION ERASURE AUDIT")
    print("=" * 80)

    total_raw_components = 0
    total_surviving_components = 0
    raw_comp_by_lesion = {l: 0 for l in LESION_TYPES}
    surv_comp_by_lesion = {l: 0 for l in LESION_TYPES}

    for i in range(len(val_ds)):
        info = val_ds.get_sample_info(i)
        for l in LESION_TYPES:
            m_path = info["mask_paths"][l]
            if not m_path.exists():
                continue
            pil_raw = Image.open(m_path).convert("L")
            arr_raw = (np.array(pil_raw) > 0).astype(np.uint8)
            if np.sum(arr_raw) == 0:
                continue

            # Connected components at raw resolution
            labeled_raw, num_features = ndimage.label(arr_raw)
            if num_features == 0:
                continue
            raw_comp_by_lesion[l] += num_features
            total_raw_components += num_features

            # Downsample using NEAREST
            pil_down = pil_raw.resize((512, 512), resample=Image.Resampling.NEAREST)
            arr_down = (np.array(pil_down) > 0).astype(np.uint8)

            labeled_down, num_surv = ndimage.label(arr_down)
            surv_comp_by_lesion[l] += num_surv
            total_surviving_components += num_surv

    print(f"Total raw lesion connected components in validation set: {total_raw_components:,}")
    print(f"Total surviving components after 512x512 NEAREST resize: {total_surviving_components:,}")
    print(f"Component retention rate: {total_surviving_components / total_raw_components * 100:.2f}%")
    print(f"Estimated components completely erased by downsampling: {total_raw_components - total_surviving_components:,} ({ (1 - total_surviving_components / total_raw_components) * 100:.2f}%)")
    print("\nBreakdown by Lesion Type:")
    for l in LESION_TYPES:
        raw_cnt = raw_comp_by_lesion[l]
        surv_cnt = surv_comp_by_lesion[l]
        ret_rate = (surv_cnt / raw_cnt * 100) if raw_cnt > 0 else 0
        erased_rate = 100.0 - ret_rate
        print(f"  {l}: Raw={raw_cnt:5d} comps -> 512x512={surv_cnt:5d} comps (Retention: {ret_rate:5.1f}%, Erased: {erased_rate:5.1f}%)")

if __name__ == "__main__":
    main()
