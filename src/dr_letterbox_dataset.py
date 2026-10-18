"""
DDR Retinal Lesion Segmentation Dataset Loader with Aspect-Ratio-Preserving Letterboxing.

Implements DDRLetterboxDataset for FedMorphis Phase 2B EXP-1.
Preserves the native aspect ratio of fundus images and lesion masks by scaling
isotropically and centering onto a 512x512 canvas with black/zero padding.

Outputs:
    image:      torch.FloatTensor of shape [3, H, W] in [0.0, 1.0] (bilinear interpolation)
    mask:       torch.FloatTensor of shape [1, H, W] in {0.0, 1.0} (nearest-neighbor)
    valid_mask: torch.FloatTensor of shape [1, H, W] in {0.0, 1.0} (1.0 for valid retina, 0.0 for padding)

Leaves existing src/dr_segmentation_dataset.py untouched.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from dr_segmentation_dataset import DDRLesionDataset, LESION_TYPES


class DDRLetterboxDataset(DDRLesionDataset):
    """
    PyTorch Dataset for DDR Lesion Segmentation with Aspect-Ratio-Preserving Letterbox Preprocessing.
    Inherits directory resolution, split parsing, and sample indexing from DDRLesionDataset.
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
            image_size: Target square dimension (int) or (width, height) tuple. Default: 512.
            target_lesion: 'all' for union of all lesions, or one of 'EX', 'HE', 'MA', 'SE'.
        """
        super().__init__(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            target_lesion=target_lesion,
        )

        # Ensure image_size is stored as (target_w, target_h)
        if isinstance(image_size, int):
            self.target_w = image_size
            self.target_h = image_size
        else:
            self.target_w, self.target_h = image_size

    def compute_letterbox_geometry(
        self, raw_w: int, raw_h: int
    ) -> Dict[str, Union[int, float, Tuple[int, int]]]:
        """
        Computes the isotropic scaling factor and centered padding offsets.

        Returns:
            dict containing scale factor, scaled dimensions, padding offsets, and valid bounding box.
        """
        scale = min(self.target_w / raw_w, self.target_h / raw_h)
        w_new = int(round(raw_w * scale))
        h_new = int(round(raw_h * scale))

        # Clamp to avoid boundary overflow due to rounding
        w_new = min(w_new, self.target_w)
        h_new = min(h_new, self.target_h)

        pad_x = (self.target_w - w_new) // 2
        pad_y = (self.target_h - h_new) // 2

        return {
            "scale": scale,
            "raw_size": (raw_w, raw_h),
            "scaled_size": (w_new, h_new),
            "pad_offsets": (pad_x, pad_y),
            "bbox": (pad_x, pad_y, pad_x + w_new, pad_y + h_new),
        }

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_tensor:      [3, 512, 512] float32 in [0, 1]
            mask_tensor:       [1, 512, 512] float32 in {0, 1}
            valid_mask_tensor: [1, 512, 512] float32 in {0, 1} (1 for retina content, 0 for padding)
        """
        sample = self.samples[index]
        img_path = sample["image_path"]
        mask_paths = sample["mask_paths"]

        # 1. Load raw fundus image to determine native resolution
        pil_image = Image.open(img_path).convert("RGB")
        raw_w, raw_h = pil_image.size

        geom = self.compute_letterbox_geometry(raw_w, raw_h)
        w_new, h_new = geom["scaled_size"]
        pad_x, pad_y = geom["pad_offsets"]

        # 2. Resize fundus image with BILINEAR interpolation and embed on black canvas
        pil_image_resized = pil_image.resize(
            (w_new, h_new), resample=Image.Resampling.BILINEAR
        )
        canvas_img = Image.new("RGB", (self.target_w, self.target_h), (0, 0, 0))
        canvas_img.paste(pil_image_resized, (pad_x, pad_y))

        image_np = np.array(canvas_img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        image_np = np.transpose(image_np, (2, 0, 1))             # [3, H, W]

        # 3. Resize constituent lesion masks with NEAREST interpolation and embed on zero canvas
        canvas_mask = np.zeros((self.target_h, self.target_w), dtype=np.float32)
        lesions_to_load = LESION_TYPES if self.target_lesion == "all" else [self.target_lesion]

        for lesion in lesions_to_load:
            m_path = mask_paths[lesion]
            if m_path.exists():
                pil_mask = Image.open(m_path).convert("L")
                pil_mask_resized = pil_mask.resize(
                    (w_new, h_new), resample=Image.Resampling.NEAREST
                )
                mask_arr = np.array(pil_mask_resized)
                # Any pixel > 0 is treated as foreground lesion
                lesion_binary = (mask_arr > 0).astype(np.float32)

                # Union into valid canvas area
                canvas_mask[pad_y : pad_y + h_new, pad_x : pad_x + w_new] = np.maximum(
                    canvas_mask[pad_y : pad_y + h_new, pad_x : pad_x + w_new],
                    lesion_binary,
                )

        # 4. Construct valid-retinal indicator mask (1.0 for valid retina, 0.0 for padding)
        valid_mask = np.zeros((self.target_h, self.target_w), dtype=np.float32)
        valid_mask[pad_y : pad_y + h_new, pad_x : pad_x + w_new] = 1.0

        # Expand channel dimensions to [1, H, W]
        combined_mask = np.expand_dims(canvas_mask, axis=0)
        valid_mask = np.expand_dims(valid_mask, axis=0)

        image_tensor = torch.from_numpy(image_np)
        mask_tensor = torch.from_numpy(combined_mask)
        valid_mask_tensor = torch.from_numpy(valid_mask)

        return image_tensor, mask_tensor, valid_mask_tensor

    def get_sample_letterbox_info(self, index: int) -> Dict:
        """Returns sample metadata combined with letterbox geometry."""
        info = self.get_sample_info(index).copy()
        with Image.open(info["image_path"]) as img:
            raw_w, raw_h = img.size
        info["geometry"] = self.compute_letterbox_geometry(raw_w, raw_h)
        return info


if __name__ == "__main__":
    print("Testing DDRLetterboxDataset...")
    for split_name in ["train", "valid"]:
        ds = DDRLetterboxDataset(split=split_name, image_size=512)
        print(f"Loaded split='{split_name}': {len(ds)} samples")
        img, mask, vmask = ds[0]
        print(f"  Sample 0 -> Image: shape={list(img.shape)}, range=[{img.min():.2f}, {img.max():.2f}]")
        print(f"  Sample 0 -> Mask:  shape={list(mask.shape)}, unique={torch.unique(mask).tolist()}")
        print(f"  Sample 0 -> Valid: shape={list(vmask.shape)}, unique={torch.unique(vmask).tolist()}, valid_fraction={vmask.mean():.3f}")
        info = ds.get_sample_letterbox_info(0)
        print(f"  Sample 0 Geometry: raw={info['geometry']['raw_size']}, scaled={info['geometry']['scaled_size']}, pad={info['geometry']['pad_offsets']}")
