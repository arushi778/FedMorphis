"""
Phase 1: STARE Dataset Inspection Script

Inspects one sample pair (im0001.ppm and im0001.ah.ppm) from the STARE dataset
without modifying, resizing, normalizing, or preprocessing any files.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def inspect_stare_sample(save_path: Path | None = None):
    # Resolve repository root and file paths
    repo_root = Path(__file__).resolve().parent.parent
    img_path = repo_root / "data" / "STARE" / "raw_images" / "im0001.ppm"
    mask_path = repo_root / "data" / "STARE" / "raw_masks" / "im0001.ah.ppm"

    if not img_path.exists():
        raise FileNotFoundError(f"STARE image not found at: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"STARE mask not found at: {mask_path}")

    # 1. Load image and mask without resizing, normalizing, or preprocessing
    with Image.open(img_path) as im:
        image_np = np.array(im.convert("RGB"))

    with Image.open(mask_path) as mk:
        mask_np = np.array(mk.convert("L"))

    # 2. Print required sample details
    print("=" * 55)
    print("STARE SAMPLE INSPECTION DETAILS")
    print("=" * 55)
    print(f"Image Type:           {type(image_np)}")
    print(f"Image Shape:          {image_np.shape}")
    print(f"Image Dtype:          {image_np.dtype}")
    print(f"Mask Type:            {type(mask_np)}")
    print(f"Mask Shape:           {mask_np.shape}")
    print(f"Mask Dtype:           {mask_np.dtype}")
    print(f"Mask Unique Values:   {np.unique(mask_np)}")
    print("=" * 55)

    # 3. Display three panels using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original STARE retinal image
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original STARE Retinal Image\n({img_path.name})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Adam Hoover vessel segmentation mask
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title(f"Adam Hoover Vessel Mask\n({mask_path.name} | 0=BG, 255=Vessel)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Original image with vessel mask overlaid transparently
    axes[2].imshow(image_np)

    # Semi-transparent RGBA green overlay for vessel pixels
    vessel_overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    vessel_pixels = mask_np > 0
    vessel_overlay[vessel_pixels] = [0, 255, 120, 180]  # Vibrant green with ~70% opacity

    axes[2].imshow(vessel_overlay)
    axes[2].set_title("Original + Vessel Mask Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    # 4. Save the figure
    if save_path is None:
        save_path = repo_root / "results" / "stare_inspection_im0001.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")

    # Display plot
    plt.show()


if __name__ == "__main__":
    inspect_stare_sample()
