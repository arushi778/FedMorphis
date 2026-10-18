"""
Phase 1: DDR Lesion Segmentation Dataset Inspection Script

Inspects DDR retinal lesion segmentation data (EX, HE, MA, SE) across train,
validation, and test splits. Verifies image-mask pairs, checks raw and preprocessed
dimensions, evaluates positive pixel distributions, and saves qualitative overlay
visualizations under results/DR/.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/automated runs
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from dr_segmentation_dataset import DDRLesionDataset, LESION_TYPES


def inspect_split(dataset: DDRLesionDataset, split_name: str, max_workers: int = 8) -> Dict:
    """
    Examines data integrity, file existence, image/mask dimensions,
    and lesion presence statistics for a given split.
    """
    total_images = len(dataset)
    print(f"\n{'=' * 60}")
    print(f"INSPECTING SPLIT: {split_name.upper()} ({total_images} images)")
    print(f"{'=' * 60}")

    all_masks_exist = True
    missing_masks_report: List[str] = []
    
    # Store unique resolutions and value checks
    image_shapes = set()
    mask_shapes = set()
    unique_mask_values = set()

    # Lesion presence counters
    lesion_image_counts = {lesion: 0 for lesion in LESION_TYPES}
    any_lesion_count = 0
    total_pixels = 0
    positive_pixels = 0

    def analyze_sample(sample_info: Dict) -> Tuple[str, Tuple[int, int], Dict[str, bool], int, int, set]:
        stem = sample_info["image_id"]
        img_path = sample_info["image_path"]
        mask_paths = sample_info["mask_paths"]

        with Image.open(img_path) as img:
            img_size = img.size  # (W, H)

        sample_has_lesion = {lesion: False for lesion in LESION_TYPES}
        sample_values = set()
        
        # Combined mask at raw resolution
        combined_raw = None

        for lesion in LESION_TYPES:
            m_path = mask_paths[lesion]
            if not m_path.exists():
                return stem, img_size, sample_has_lesion, 0, img_size[0] * img_size[1], {"MISSING"}

            with Image.open(m_path) as m:
                # Fast check using extrema
                extrema = m.getextrema()
                if extrema != (0, 0):
                    arr = np.array(m)
                    sample_values.update(np.unique(arr).tolist())
                    if np.any(arr > 0):
                        sample_has_lesion[lesion] = True
                    if combined_raw is None:
                        combined_raw = (arr > 0)
                    else:
                        combined_raw = np.logical_or(combined_raw, arr > 0)
                else:
                    sample_values.add(0)

        num_total = img_size[0] * img_size[1]
        num_pos = int(np.sum(combined_raw)) if combined_raw is not None else 0

        return stem, img_size, sample_has_lesion, num_pos, num_total, sample_values

    # Parallel processing of all samples in split
    sample_infos = [dataset.get_sample_info(i) for i in range(total_images)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_sample, sample_infos))

    valid_pairs = 0
    for stem, img_size, has_lesion, pos_px, tot_px, vals in results:
        if "MISSING" in vals:
            all_masks_exist = False
            missing_masks_report.append(stem)
            continue

        valid_pairs += 1
        image_shapes.add(img_size)
        mask_shapes.add(img_size)
        unique_mask_values.update(vals)

        has_any = False
        for lesion in LESION_TYPES:
            if has_lesion[lesion]:
                lesion_image_counts[lesion] += 1
                has_any = True

        if has_any:
            any_lesion_count += 1

        positive_pixels += pos_px
        total_pixels += tot_px

    pos_pct = (positive_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    print(f"Number of images:                 {total_images}")
    print(f"Number of valid image-mask pairs: {valid_pairs} / {total_images}")
    print(f"All 4 lesion masks present:       {'YES' if all_masks_exist else 'NO'}")
    if missing_masks_report:
        print(f"  Missing mask IDs: {missing_masks_report[:10]}")
    print(f"Raw image dimensions (W, H):      {sorted(list(image_shapes))}")
    print(f"Raw mask dimensions (W, H):       {sorted(list(mask_shapes))}")
    print(f"Unique values in raw masks:       {sorted(list(unique_mask_values))}")
    print(f"Total positive lesion pixels:     {positive_pixels:,}")
    print(f"Percentage of positive pixels:    {pos_pct:.4f}%")
    print(f"Images with at least one lesion:  {any_lesion_count} / {total_images} ({any_lesion_count / total_images * 100:.1f}%)")
    for lesion in LESION_TYPES:
        cnt = lesion_image_counts[lesion]
        print(f"  - Images with {lesion:2s}:                {cnt:3d} ({cnt / total_images * 100:5.1f}%)")

    # Verify dataset tensor output
    sample_img_t, sample_mask_t = dataset[0]
    print(f"Dataset tensor output shapes:     Image: {list(sample_img_t.shape)}, Mask: {list(sample_mask_t.shape)}")
    print(f"Dataset tensor mask unique vals:  {torch.unique(sample_mask_t).tolist()}")

    return {
        "split": split_name,
        "total_images": total_images,
        "valid_pairs": valid_pairs,
        "all_masks_exist": all_masks_exist,
        "image_shapes": list(image_shapes),
        "unique_values": list(unique_mask_values),
        "positive_pixels": positive_pixels,
        "total_pixels": total_pixels,
        "pos_percentage": pos_pct,
        "any_lesion_count": any_lesion_count,
        "lesion_counts": lesion_image_counts,
    }


def save_visualizations(
    datasets: Dict[str, DDRLesionDataset],
    output_dir: Path,
    num_samples_overview: int = 3,
):
    """
    Renders and saves qualitative inspection figures displaying:
      1. Original fundus image
      2. Combined lesion segmentation mask
      3. Original fundus with lesion overlay
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating and saving visualizations to {output_dir}...")

    # Pick samples that clearly have positive lesions for informative visualization
    selected_samples = []
    for split_name, ds in datasets.items():
        count = 0
        for idx in range(len(ds)):
            img_t, mask_t = ds[idx]
            if mask_t.sum() > 50:  # Has visible lesions
                info = ds.get_sample_info(idx)
                selected_samples.append((split_name, idx, info["image_id"], img_t, mask_t))
                count += 1
                if count >= 2:
                    break

    # Save individual detailed 3-panel figures
    for split_name, idx, img_id, img_t, mask_t in selected_samples:
        img_np = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        mask_np = mask_t.squeeze(0).numpy()  # [512, 512] float

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Original fundus image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original Fundus ({split_name.upper()} | {img_id})", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Combined binary lesion mask
        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title(f"Combined Lesion Mask (0=BG, 1=Lesion)\nPositive Px: {int(mask_np.sum()):,}", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Lesion overlay (bright red/orange on lesions)
        axes[2].imshow(img_np)
        overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        pos_mask = mask_np > 0
        overlay[pos_mask] = [255, 30, 30, 200]  # Vivid red with high opacity
        axes[2].imshow(overlay)
        axes[2].set_title(f"Fundus + Lesion Mask Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        single_path = output_dir / f"ddr_inspection_{split_name}_{img_id}.png"
        plt.savefig(single_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {single_path.name}")

    # Create composite overview figure (rows = samples, cols = original / mask / overlay)
    n_rows = min(len(selected_samples), 4)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx in range(n_rows):
        split_name, idx, img_id, img_t, mask_t = selected_samples[r_idx]
        img_np = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        mask_np = mask_t.squeeze(0).numpy()

        # Original
        axes[r_idx, 0].imshow(img_np)
        axes[r_idx, 0].set_title(f"{split_name.upper()} Fundus: {img_id}", fontsize=11, fontweight="bold")
        axes[r_idx, 0].axis("off")

        # Mask
        axes[r_idx, 1].imshow(mask_np, cmap="gray")
        axes[r_idx, 1].set_title(f"Combined Lesion Mask ({int(mask_np.sum())} px)", fontsize=11, fontweight="bold")
        axes[r_idx, 1].axis("off")

        # Overlay
        axes[r_idx, 2].imshow(img_np)
        overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        overlay[mask_np > 0] = [255, 40, 40, 210]
        axes[r_idx, 2].imshow(overlay)
        axes[r_idx, 2].set_title(f"Lesion Overlay", fontsize=11, fontweight="bold")
        axes[r_idx, 2].axis("off")

    plt.tight_layout()
    overview_path = output_dir / "ddr_inspection_samples.png"
    plt.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite overview figure: {overview_path.name}")


def main():
    print("=" * 60)
    print("FEDMORPHIS: DDR RETINAL LESION SEGMENTATION INSPECTION")
    print("=" * 60)

    # 1. Initialize datasets for all splits
    train_dataset = DDRLesionDataset(split="train", image_size=512)
    valid_dataset = DDRLesionDataset(split="valid", image_size=512)
    test_dataset = DDRLesionDataset(split="test", image_size=512)

    datasets = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    # 2. Run inspection across all splits
    stats = {}
    for split_name, ds in datasets.items():
        stats[split_name] = inspect_split(ds, split_name)

    # 3. Print overall summary table
    print(f"\n{'=' * 60}")
    print("DDR LESION SEGMENTATION: OVERALL SUMMARY")
    print(f"{'=' * 60}")
    total_imgs = sum(s["total_images"] for s in stats.values())
    total_pairs = sum(s["valid_pairs"] for s in stats.values())
    total_any = sum(s["any_lesion_count"] for s in stats.values())
    total_pos_px = sum(s["positive_pixels"] for s in stats.values())
    total_all_px = sum(s["total_pixels"] for s in stats.values())
    overall_pos_pct = (total_pos_px / total_all_px * 100) if total_all_px > 0 else 0

    print(f"Total Dataset Images:        {total_imgs}")
    print(f"Total Valid Pairs:           {total_pairs} / {total_imgs}")
    print(f"Total Images with Lesions:   {total_any} / {total_imgs} ({total_any / total_imgs * 100:.1f}%)")
    print(f"Total Lesion Pixels:         {total_pos_px:,} ({overall_pos_pct:.4f}% foreground density)")
    print(f"\nPer-split distribution:")
    print(f"{'Split':<8} {'Images':<8} {'With Lesion':<14} {'EX':<8} {'HE':<8} {'MA':<8} {'SE':<8}")
    for split_name, s in stats.items():
        lc = s["lesion_counts"]
        print(
            f"{split_name:<8} {s['total_images']:<8} "
            f"{s['any_lesion_count']:<14} "
            f"{lc['EX']:<8} {lc['HE']:<8} {lc['MA']:<8} {lc['SE']:<8}"
        )

    # 4. Save qualitative visual artifacts
    results_dr_dir = Path(__file__).resolve().parent.parent / "results" / "DR"
    save_visualizations(datasets, results_dr_dir)

    print(f"\nInspection completed successfully!")
    print(f"Results and figures saved under: {results_dr_dir}")


if __name__ == "__main__":
    main()
