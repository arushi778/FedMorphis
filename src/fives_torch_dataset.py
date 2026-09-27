import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FIVESDataset(Dataset):
    def __init__(self, parquet_path, image_size=512):
        self.parquet_path = Path(parquet_path)
        self.image_size = image_size

        self.df = pd.read_parquet(self.parquet_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Read image and mask from Parquet
        image = Image.open(
            io.BytesIO(row["image"]["bytes"])
        ).convert("RGB")

        mask = Image.open(
            io.BytesIO(row["mask"]["bytes"])
        ).convert("L")

        # Resize
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        # Convert to NumPy
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        # Normalize image to [0, 1]
        image = image / 255.0

        # Convert mask from {0, 255} to {0, 1}
        mask = mask / 255.0

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))

        # Add channel dimension to mask
        mask = np.expand_dims(mask, axis=0)

        # NumPy → PyTorch tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask


if __name__ == "__main__":
    dataset = FIVESDataset(
        "data/FIVES/data/train-00000-of-00003.parquet"
    )

    print("Dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image range:", image.min().item(), image.max().item())

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask values:", torch.unique(mask))