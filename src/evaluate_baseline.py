import io

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from unet import UNet


class FIVESTestDataset(Dataset):
    def __init__(self, parquet_path, image_size=512):
        self.df = pd.read_parquet(parquet_path)
        self.image_size = image_size

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

        image = torch.tensor(
            __import__("numpy").array(image),
            dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        mask = torch.tensor(
            __import__("numpy").array(mask),
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        return image, mask


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (
        (2 * intersection + smooth)
        / (
            predictions.sum()
            + targets.sum()
            + smooth
        )
    )

    return dice.item()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    dataset = FIVESTestDataset(
        "data/FIVES/data/test-00000-of-00001.parquet"
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("Test samples:", len(dataset))

    model = UNet().to(device)

    model.load_state_dict(
        torch.load(
            "best_unet_fives.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)

            total_dice += dice_score(
                logits,
                masks
            )

    average_dice = total_dice / len(loader)

    print()
    print("Test Dice:", average_dice)


if __name__ == "__main__":
    main()