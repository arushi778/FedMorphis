import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fives_torch_dataset import FIVESDataset
from shared_mim_model import SharedMIMModel
from mim_masking import create_patch_mask


def apply_patch_mask(images, mask, patch_size=32):
    """
    Replace masked 32x32 patches with zeros.
    mask: [B, num_patches], True = masked
    """
    masked_images = images.clone()

    batch_size, _, height, width = images.shape
    patches_per_side = height // patch_size

    for b in range(batch_size):
        patch_idx = 0

        for row in range(patches_per_side):
            for col in range(patches_per_side):
                if mask[b, patch_idx]:
                    y1 = row * patch_size
                    y2 = y1 + patch_size
                    x1 = col * patch_size
                    x2 = x1 + patch_size

                    masked_images[b, :, y1:y2, x1:x2] = 0

                patch_idx += 1

    return masked_images


def masked_mse_loss(reconstruction, target, mask, patch_size=32):
    """
    Calculate MSE only inside masked patches.
    """
    batch_size, _, height, width = target.shape
    patches_per_side = height // patch_size

    total_loss = 0.0
    total_pixels = 0

    for b in range(batch_size):
        patch_idx = 0

        for row in range(patches_per_side):
            for col in range(patches_per_side):

                if mask[b, patch_idx]:
                    y1 = row * patch_size
                    y2 = y1 + patch_size
                    x1 = col * patch_size
                    x2 = x1 + patch_size

                    pred_patch = reconstruction[
                        b, :, y1:y2, x1:x2
                    ]

                    target_patch = target[
                        b, :, y1:y2, x1:x2
                    ]

                    total_loss += torch.sum(
                        (pred_patch - target_patch) ** 2
                    )

                    total_pixels += pred_patch.numel()

                patch_idx += 1

    return total_loss / total_pixels


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Dataset
    dataset = FIVESDataset(
    [
        "data/FIVES/data/train-00000-of-00003.parquet",
        "data/FIVES/data/train-00001-of-00003.parquet",
        "data/FIVES/data/train-00002-of-00003.parquet",
    ],
    image_size=512
)

    print("Training samples:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # Shared MIM model
    model = SharedMIMModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 5

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(loader):

            # IMPORTANT:
            # masks are deliberately ignored during SSL pretraining.
            images = images.to(device)

            # Create 75% patch mask
            patch_mask = create_patch_mask(
                batch_size=images.shape[0],
                image_size=512,
                patch_size=32,
                mask_ratio=0.75
            ).to(device)

            # Mask the images
            masked_images = apply_patch_mask(
                images,
                patch_mask,
                patch_size=32
            )

            # Forward pass
            reconstruction, _ = model(masked_images)

            # Loss only on masked regions
            loss = masked_mse_loss(
                reconstruction,
                images,
                patch_mask,
                patch_size=32
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx + 1}/{len(loader)} | "
                    f"Loss {loss.item():.6f}"
                )

        epoch_loss = running_loss / len(loader)

        print(
            f"\nEpoch {epoch + 1}/{epochs} "
            f"| Average MIM Loss: {epoch_loss:.6f}\n"
        )

    # Save pretrained model
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = "checkpoints/shared_mim_pretrained.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "mask_ratio": 0.75,
            "patch_size": 32,
        },
        checkpoint_path
    )

    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()