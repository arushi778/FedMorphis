import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fives_torch_dataset import FIVESDataset
from shared_unet import SharedUNet


def dice_loss(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + smooth) / (
        denominator + smooth
    )

    return 1 - dice.mean()


def dice_score(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    denominator = (
        predictions.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + smooth) / (
        denominator + smooth
    )

    return dice.mean().item()


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = FIVESDataset(
        [
            "data/FIVES/data/train-00000-of-00003.parquet",
            "data/FIVES/data/train-00001-of-00003.parquet",
            "data/FIVES/data/train-00002-of-00003.parquet",
        ],
        image_size=512
    )

    # Same 80/20 split used for the original baseline
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    train_indices = indices[:480]
    val_indices = indices[480:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # ---------------------------------------------------------
    # Shared U-Net
    # ---------------------------------------------------------

    model = SharedUNet().to(device)

    # ---------------------------------------------------------
    # Load MIM-pretrained encoder
    # ---------------------------------------------------------

    checkpoint = torch.load(
        "checkpoints/shared_mim_pretrained.pth",
        map_location=device
    )

    mim_state = checkpoint["model_state_dict"]

    encoder_state = {
        key.replace("encoder.", "", 1): value
        for key, value in mim_state.items()
        if key.startswith("encoder.")
    }

    model.encoder.load_state_dict(
        encoder_state,
        strict=True
    )

    print("Loaded MIM-pretrained encoder.")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    bce_loss = nn.BCEWithLogitsLoss()

    epochs = 5

    best_val_dice = 0.0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss(logits, masks)

            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        model.eval()

        validation_dice = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)

                validation_dice += dice_score(
                    logits,
                    masks
                )

        validation_dice /= len(val_loader)

        print(
            f"\nEpoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Dice: {validation_dice:.4f}\n"
        )

        if validation_dice > best_val_dice:

            best_val_dice = validation_dice

            torch.save(
                model.state_dict(),
                "checkpoints/best_mim_shared_unet.pth"
            )

            print(
                f"Saved best model. "
                f"Val Dice: {best_val_dice:.4f}"
            )

    print("\nTraining complete.")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(
        "Saved: "
        "checkpoints/best_mim_shared_unet.pth"
    )


if __name__ == "__main__":
    main()
    