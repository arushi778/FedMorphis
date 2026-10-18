"""
DDR Retinal Lesion Segmentation Dataset Loader.

Supports loading fundus images and pixel-level lesion annotations (EX, HE, MA, SE)
from the DDR dataset (data/DR/DDR-dataset/lesion_segmentation/).

Provides binary segmentation masks (background = 0, any DR lesion = 1)
compatible with PyTorch segmentation models (e.g., SharedUNet, UNet).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


LESION_TYPES = ["EX", "HE", "MA", "SE"]


class DDRLesionDataset(Dataset):
    """
    PyTorch Dataset for DDR Lesion Segmentation.

    Directory structure expected:
        root_dir/
            train/
                image/
                label/{EX, HE, MA, SE}/
            valid/
                image/
                segmentation label/{EX, HE, MA, SE}/ (or label/)
            test/
                image/
                label/{EX, HE, MA, SE}/

    Returns:
        image: torch.FloatTensor of shape [3, H, W] in range [0.0, 1.0]
        mask:  torch.FloatTensor of shape [1, H, W] in range [0.0, 1.0]
               (where 1.0 denotes presence of any lesion if target_lesion='all')
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        image_size: Union[int, Tuple[int, int]] = 512,
        target_lesion: str = "all",
    ):
        """
        Args:
            root_dir: Path to lesion_segmentation folder. If None, resolves relative to repo root.
            split: 'train', 'valid' (or 'val'), or 'test'.
            image_size: Target square dimension (int) or (height, width) tuple.
            target_lesion: 'all' for union of all lesions, or one of 'EX', 'HE', 'MA', 'SE'.
        """
        super().__init__()

        if root_dir is None:
            # Default to repo root / data / DR / DDR-dataset / lesion_segmentation
            self.root_dir = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "DR"
                / "DDR-dataset"
                / "lesion_segmentation"
            )
        else:
            self.root_dir = Path(root_dir)

        # Normalize split name ('val' -> 'valid')
        norm_split = split.lower().strip()
        if norm_split in ["val", "validation", "valid"]:
            self.split = "valid"
        elif norm_split == "train":
            self.split = "train"
        elif norm_split == "test":
            self.split = "test"
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'valid', or 'test'.")

        self.split_dir = self.root_dir / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.target_lesion = target_lesion.upper() if target_lesion != "all" else "all"
        if self.target_lesion != "all" and self.target_lesion not in LESION_TYPES:
            raise ValueError(
                f"target_lesion must be 'all' or one of {LESION_TYPES}, got '{target_lesion}'"
            )

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        # Locate image directory
        self.image_dir = self.split_dir / "image"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Locate label directory: handle 'label' vs 'segmentation label'
        self.label_dir = self._find_label_dir()

        # Discover and pair samples
        self.samples: List[Dict[str, Union[str, Path, Dict[str, Path]]]] = []
        self._index_samples()

    def _find_label_dir(self) -> Path:
        """Finds the mask directory, handling 'segmentation label' in valid."""
        candidates = [
            self.split_dir / "label",
            self.split_dir / "segmentation label",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Could not find label directory in {self.split_dir}. Looked for: {candidates}"
        )

    def _index_samples(self) -> None:
        """Indexes all images and verifies corresponding mask paths for all 4 lesions."""
        valid_extensions = {".jpg", ".jpeg", ".png"}
        image_files = sorted(
            [f for f in self.image_dir.iterdir() if f.suffix.lower() in valid_extensions]
        )

        for img_path in image_files:
            stem = img_path.stem

            mask_paths: Dict[str, Path] = {}
            missing_lesions: List[str] = []

            for lesion in LESION_TYPES:
                lesion_mask_dir = self.label_dir / lesion
                mask_path = lesion_mask_dir / f"{stem}.tif"
                if not mask_path.exists():
                    # Check for alternate case or extensions if needed
                    alt_candidates = list(lesion_mask_dir.glob(f"{stem}.*"))
                    if alt_candidates:
                        mask_path = alt_candidates[0]
                    else:
                        missing_lesions.append(lesion)
                mask_paths[lesion] = mask_path

            self.samples.append(
                {
                    "image_id": stem,
                    "image_path": img_path,
                    "mask_paths": mask_paths,
                    "missing_lesions": missing_lesions,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get_sample_info(self, index: int) -> Dict[str, Union[str, Path, Dict[str, Path]]]:
        """Returns metadata for a given index without loading heavy image tensors."""
        return self.samples[index]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        img_path = sample["image_path"]
        mask_paths = sample["mask_paths"]

        # 1. Load and resize fundus image
        pil_image = Image.open(img_path).convert("RGB")
        pil_image = pil_image.resize(
            self.image_size,
            resample=Image.Resampling.BILINEAR
        )
        image_np = np.array(pil_image, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        image_np = np.transpose(image_np, (2, 0, 1))             # [3, H, W]

        # 2. Load and resize masks
        target_w, target_h = self.image_size
        combined_mask = np.zeros((target_h, target_w), dtype=np.float32)

        lesions_to_load = LESION_TYPES if self.target_lesion == "all" else [self.target_lesion]

        for lesion in lesions_to_load:
            m_path = mask_paths[lesion]
            if m_path.exists():
                pil_mask = Image.open(m_path).convert("L")
                pil_mask = pil_mask.resize(
                    self.image_size,
                    resample=Image.Resampling.NEAREST
                )
                mask_arr = np.array(pil_mask)
                # Any pixel > 0 is treated as foreground lesion
                lesion_binary = (mask_arr > 0).astype(np.float32)
                combined_mask = np.maximum(combined_mask, lesion_binary)

        combined_mask = np.expand_dims(combined_mask, axis=0)  # [1, H, W]

        image_tensor = torch.from_numpy(image_np)
        mask_tensor = torch.from_numpy(combined_mask)

        return image_tensor, mask_tensor


# Alias for compatibility
DRSegmentationDataset = DDRLesionDataset


if __name__ == "__main__":
    print("Testing DDRLesionDataset...")
    for split_name in ["train", "valid", "test"]:
        ds = DDRLesionDataset(split=split_name, image_size=512)
        print(f"Loaded split='{split_name}': {len(ds)} samples, Label dir: {ds.label_dir.name}")
        img, mask = ds[0]
        print(f"  Sample 0 -> Image: {img.shape} (dtype={img.dtype}, min={img.min():.2f}, max={img.max():.2f})")
        print(f"  Sample 0 -> Mask:  {mask.shape} (dtype={mask.dtype}, unique={torch.unique(mask).tolist()})")
