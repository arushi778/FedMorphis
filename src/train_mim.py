import torch
from torch.utils.data import DataLoader

from fives_torch_dataset import FIVESDataset
from mim_masking import create_patch_mask
from mim_model import MIMModel
from test_mim_loss import apply_patch_mask, masked_mse_loss


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = FIVESDataset([
        "data/FIVES/data/train-00000-of-00003.parquet",
        "data/FIVES/data/train-00001-of-00003.parquet",
        "data/FIVES/data/train-00002-of-00003.parquet",
    ])

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    print("Images available for MIM:", len(dataset))

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = MIMModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 5

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for images, _ in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            # Create random patch mask
            mask = create_patch_mask(
                batch_size=images.shape[0],
                image_size=512,
                patch_size=32,
                mask_ratio=0.75,
                device=device,
            )

            # Hide patches
            masked_images = apply_patch_mask(
                images,
                mask,
                patch_size=32
            )

            # Forward pass
            reconstruction, _ = model(
                masked_images
            )

            # Loss only on masked patches
            loss = masked_mse_loss(
                reconstruction,
                images,
                mask,
                patch_size=32
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"MIM Loss: {average_loss:.6f}"
        )

    # -------------------------------------------------
    # Save pretrained model
    # -------------------------------------------------

    torch.save(
        model.state_dict(),
        "mim_pretrained.pth"
    )

    print()
    print("MIM pretraining complete.")
    print("Saved model: mim_pretrained.pth")


if __name__ == "__main__":
    main()