import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fives_torch_dataset import FIVESDataset
from unet import UNet


# -------------------------
# Dice loss
# -------------------------
def dice_loss(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(-1)
    targets = targets.view(-1)

    intersection = (probabilities * targets).sum()

    dice = (
        (2 * intersection + smooth)
        / (
            probabilities.sum()
            + targets.sum()
            + smooth
        )
    )

    return 1 - dice


# -------------------------
# Combined loss
# -------------------------
def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(logits, targets)

    return bce + dice


# -------------------------
# Dice metric
# -------------------------
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


# -------------------------
# Main
# -------------------------
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    dataset = FIVESDataset([
        "data/FIVES/data/train-00000-of-00003.parquet",
        "data/FIVES/data/train-00001-of-00003.parquet",
        "data/FIVES/data/train-00002-of-00003.parquet",
    ])

    # Load saved split
    train_indices = torch.tensor(
        __import__("pandas")
        .read_csv("data/FIVES/splits/train_indices.csv")["index"]
        .values
    )

    val_indices = torch.tensor(
        __import__("pandas")
        .read_csv("data/FIVES/splits/val_indices.csv")["index"]
        .values
    )

    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -------------------------
    # Model
    # -------------------------
    model = UNet().to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # -------------------------
    # Training
    # -------------------------
    epochs = 5

    best_val_dice = 0.0

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)

            loss = combined_loss(
                logits,
                masks
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        total_val_dice = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)

                dice = dice_score(
                    logits,
                    masks
                )

                total_val_dice += dice

        average_val_dice = (
            total_val_dice / len(val_loader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Dice: {average_val_dice:.4f}"
        )

        # Save best model
        if average_val_dice > best_val_dice:

            best_val_dice = average_val_dice

            torch.save(
                model.state_dict(),
                "best_unet_fives.pth"
            )

            print("  Saved best model.")

    print()
    print("Training complete.")
    print("Best validation Dice:", best_val_dice)


if __name__ == "__main__":
    main()