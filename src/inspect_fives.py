"""
Phase 1: FIVES Dataset Inspection Script

Inspects raw retinal images and vessel segmentation masks from the FIVES Parquet files
without modifying, resizing, normalizing, or preprocessing the original data.
"""

import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


def get_default_parquet_path() -> Path:
    """Locates the first available FIVES parquet data file."""
    search_dir = Path(__file__).resolve().parent.parent / "data" / "FIVES" / "data"
    parquet_files = sorted(search_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {search_dir}. "
            "Please ensure the FIVES dataset is downloaded."
        )
    return parquet_files[0]


def extract_bytes(field_data) -> bytes:
    """Extracts raw byte payload from Hugging Face struct or direct bytes."""
    if isinstance(field_data, dict):
        return field_data["bytes"]
    if isinstance(field_data, (bytes, bytearray)):
        return bytes(field_data)
    raise TypeError(f"Unexpected image field type: {type(field_data)}")


def inspect_fives_sample(parquet_path: Path | None = None, save_figure: bool = True):
    if parquet_path is None:
        parquet_path = get_default_parquet_path()

    print(f"Reading sample from: {parquet_path}")

    # Read exactly one record without loading full dataset into RAM
    parquet_file = pq.ParquetFile(str(parquet_path))
    batch = next(parquet_file.iter_batches(batch_size=1))
    row = batch.to_pandas().iloc[0]

    image_id = row.get("image_id", "Unknown ID")
    raw_img_field = row["image"]
    raw_mask_field = row["mask"]

    # Extract image and mask raw bytes
    img_bytes = extract_bytes(raw_img_field)
    mask_bytes = extract_bytes(raw_mask_field)

    # Decode without resizing, normalizing, or preprocessing
    pil_image = Image.open(io.BytesIO(img_bytes))
    pil_mask = Image.open(io.BytesIO(mask_bytes))

    image_np = np.array(pil_image)
    mask_np = np.array(pil_mask)

    # 3. Print required sample details
    print("=" * 50)
    print("FIVES SAMPLE INSPECTION DETAILS")
    print("=" * 50)
    print(f"Sample ID:            {image_id}")
    print(f"Image Type:           {type(image_np)}")
    print(f"Image Dtype:          {image_np.dtype}")
    print(f"Image Shape:          {image_np.shape}")
    print(f"Mask Type:            {type(mask_np)}")
    print(f"Mask Dtype:           {mask_np.dtype}")
    print(f"Mask Shape:           {mask_np.shape}")
    print(f"Mask Unique Values:   {np.unique(mask_np)}")
    print("=" * 50)

    # 4. Display three images using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Original Retinal Image
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original Retinal Image ({image_id})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Subplot 2: Vessel Segmentation Mask
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Vessel Segmentation Mask (0=BG, 255=Vessel)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Subplot 3: Original Image with Mask Overlaid (transparently combined)
    axes[2].imshow(image_np)

    # Construct an RGBA green overlay for positive vessel pixels
    vessel_overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    vessel_mask = mask_np > 0
    vessel_overlay[vessel_mask] = [0, 255, 120, 180]  # Bright green with ~70% opacity

    axes[2].imshow(vessel_overlay)
    axes[2].set_title("Original + Vessel Mask Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    # Save figure to results/ for persistent artifact review
    if save_figure:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_plot_path = results_dir / f"fives_inspection_{image_id}.png"
        plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization figure to: {output_plot_path}")

    # Display window (interactive)
    plt.show()


if __name__ == "__main__":
    inspect_fives_sample()
