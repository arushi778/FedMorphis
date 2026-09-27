"""
Phase 1: CHASE-DB1 Dataset Inspection Script

Inspects one sample pair (Image_01L.jpg and Image_01L_1stHO.png) from the CHASE-DB1 dataset
without modifying, resizing, normalizing, or preprocessing any files.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def inspect_chase_sample(save_path: Path | None = None):
    # Resolve repository root and file paths
    repo_root = Path(__file__).resolve().parent.parent
    img_path = repo_root / "data" / "CHASE_DB1" / "raw" / "Image_01L.jpg"
    mask_path = repo_root / "data" / "CHASE_DB1" / "raw" / "Image_01L_1stHO.png"

    if not img_path.exists():
        raise FileNotFoundError(f"CHASE image not found at: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"CHASE mask not found at: {mask_path}")

    # Load image and mask without resizing, normalizing, or preprocessing
    with Image.open(img_path) as im:
        image_np = np.array(im.convert("RGB"))

    with Image.open(mask_path) as mk:
        mask_np = np.array(mk.convert("L"))

    # Print required sample details
    print("=" * 55)
    print("CHASE-DB1 SAMPLE INSPECTION DETAILS")
    print("=" * 55)
    print(f"Image Shape:          {image_np.shape}")
    print(f"Image Dtype:          {image_np.dtype}")
    print(f"Mask Shape:           {mask_np.shape}")
    print(f"Mask Dtype:           {mask_np.dtype}")
    print(f"Unique Mask Values:   {np.unique(mask_np)}")
    print("=" * 55)

    # Display three panels using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original retinal image
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original Retinal Image\n({img_path.name})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: 1st Human Observer vessel mask
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title(f"1st Human Observer Vessel Mask\n({mask_path.name} | 0=BG, 255=Vessel)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Transparent vessel-mask overlay on original image
    axes[2].imshow(image_np)

    # Semi-transparent RGBA green overlay for vessel pixels
    vessel_overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    vessel_pixels = mask_np > 0
    vessel_overlay[vessel_pixels] = [0, 255, 120, 180]  # Vibrant green with ~70% opacity

    axes[2].imshow(vessel_overlay)
    axes[2].set_title("Original + Vessel Mask Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    # Save figure
    if save_path is None:
        save_path = repo_root / "results" / "chase_inspection_Image_01L.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")

    # Display plot
    plt.show()


if __name__ == "__main__":
    inspect_chase_sample()
