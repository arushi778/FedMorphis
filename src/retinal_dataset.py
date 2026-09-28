import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RetinalDataset(Dataset):
    """
    Common dataset interface for retinal vessel segmentation.

    Each item returns:
        image -> [3, H, W], float32, range [0, 1]
        mask  -> [1, H, W], float32, range [0, 1]
    """

    def __init__(self, samples, image_size=512):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask