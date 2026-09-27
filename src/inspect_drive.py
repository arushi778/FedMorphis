"""
Phase 1: DRIVE Dataset Inspection Script

Inspects one sample pair (21_training.tif and 21_manual1.gif) from the DRIVE dataset
without modifying, resizing, normalizing, or preprocessing any files.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def inspect_drive_sample(save_path: Path | None = None):
    # Resolve repository root and dataset paths
    repo_root = Path(__file__).resolve().parent.parent
    img_path = repo_root / "data" / "DRIVE" / "training" / "images" / "21_training.tif"
    mask_path = repo_root / "data" / "DRIVE" / "training" / "1st_manual" / "21_manual1.gif"

    if not img_path.exists():
        raise FileNotFoundError(f"Training image not found at: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Manual vessel mask not found at: {mask_path}")

    # 1 & 2. Read image (TIFF) and mask (GIF) without resizing or normalization
    # PIL natively handles LZW-compressed TIFFs and indexed/grayscale GIF masks
    with Image.open(img_path) as im:
        image_np = np.array(im.convert("RGB"))

    with Image.open(mask_path) as mk:
        mask_np = np.array(mk.convert("L"))

    # 3. Print required sample details
    print("=" * 55)
    print("DRIVE SAMPLE INSPECTION DETAILS")
    print("=" * 55)
    print(f"Image Filename:       {img_path.name}")
    print(f"Image Type:           {type(image_np)}")
    print(f"Image Shape:          {image_np.shape}")
    print(f"Image Dtype:          {image_np.dtype}")
    print(f"Mask Filename:        {mask_path.name}")
    print(f"Mask Type:            {type(mask_np)}")
    print(f"Mask Shape:           {mask_np.shape}")
    print(f"Mask Dtype:           {mask_np.dtype}")
    print(f"Mask Unique Values:   {np.unique(mask_np)}")
    print("=" * 55)

    # 4. Display three panels using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original DRIVE retinal image
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original DRIVE Retinal Image\n({img_path.name})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Ground-truth vessel segmentation mask
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title(f"Ground-Truth Vessel Mask\n({mask_path.name} | 0=BG, 255=Vessel)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Original image with vessel mask overlaid transparently
    axes[2].imshow(image_np)

    # Create semi-transparent RGBA overlay for vessel pixels
    vessel_overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    vessel_pixels = mask_np > 0
    vessel_overlay[vessel_pixels] = [0, 255, 120, 180]  # Vibrant green with ~70% alpha

    axes[2].imshow(vessel_overlay)
    axes[2].set_title("Original + Vessel Mask Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    # 5. Save visualization
    if save_path is None:
        save_path = repo_root / "results" / "drive_inspection_21.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")

    # Display plot
    plt.show()


if __name__ == "__main__":
    inspect_drive_sample()
