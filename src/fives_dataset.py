import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class FIVESDataset:
    def __init__(self, parquet_path):
        self.parquet_path = Path(parquet_path)

        self.df = pd.read_parquet(self.parquet_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        mask = Image.open(io.BytesIO(row["mask"]["bytes"])).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        return image, mask


if __name__ == "__main__":
    dataset = FIVESDataset(
        "data/FIVES/data/train-00000-of-00003.parquet"
    )

    print("Dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask values:", np.unique(mask))