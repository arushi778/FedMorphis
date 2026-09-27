import torch
from torch.utils.data import DataLoader

from fives_torch_dataset import FIVESDataset
from shared_unet import SharedUNet


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
    # Official FIVES test set
    # ---------------------------------------------------------

    test_dataset = FIVESDataset(
        "data/FIVES/data/test-00000-of-00001.parquet",
        image_size=512
    )

    print("Test samples:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = SharedUNet().to(device)

    checkpoint_path = (
        "checkpoints/best_mim_shared_unet.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    print("Loaded:", checkpoint_path)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    model.eval()

    total_dice = 0.0

    with torch.no_grad():

        for batch_idx, (images, masks) in enumerate(test_loader):

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            batch_dice = dice_score(
                logits,
                masks
            )

            total_dice += batch_dice

            if (batch_idx + 1) % 25 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(test_loader)}"
                )

    test_dice = total_dice / len(test_loader)

    print("\n==============================")
    print("MIM + Shared U-Net Test Result")
    print("==============================")
    print(f"Test Dice: {test_dice:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()