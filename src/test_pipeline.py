import torch
from torch.utils.data import DataLoader

from fives_torch_dataset import FIVESDataset
from unet import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = FIVESDataset(
        "data/FIVES/data/train-00000-of-00003.parquet"
    )

    # One batch for testing
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    images, masks = next(iter(loader))

    images = images.to(device)
    masks = masks.to(device)

    # Model
    model = UNet().to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        predictions = model(images)

    print("Device:", device)
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    print("Predictions shape:", predictions.shape)
    print("Prediction range:",
          predictions.min().item(),
          predictions.max().item())


if __name__ == "__main__":
    main()