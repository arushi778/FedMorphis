import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FIVESDataset(Dataset):
    def __init__(self, parquet_paths, image_size=512):
        self.image_size = image_size

        if isinstance(parquet_paths, (str, Path)):
            parquet_paths = [parquet_paths]

        dataframes = [
            pd.read_parquet(path)
            for path in parquet_paths
        ]

        self.df = pd.concat(
            dataframes,
            ignore_index=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = Image.open(
            io.BytesIO(row["image"]["bytes"])
        ).convert("RGB")

        mask = Image.open(
            io.BytesIO(row["mask"]["bytes"])
        ).convert("L")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        image = image / 255.0
        mask = mask / 255.0

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask


if __name__ == "__main__":
    dataset = FIVESDataset([
        "data/FIVES/data/train-00000-of-00003.parquet",
        "data/FIVES/data/train-00001-of-00003.parquet",
        "data/FIVES/data/train-00002-of-00003.parquet",
    ])

    print("Total dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", torch.unique(mask))