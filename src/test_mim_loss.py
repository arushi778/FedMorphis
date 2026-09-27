import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fives_torch_dataset import FIVESDataset
from mim_masking import create_patch_mask
from mim_model import MIMModel


def apply_patch_mask(images, mask, patch_size=32):
    """
    Replace masked patches with zeros.
    """

    masked_images = images.clone()

    batch_size = images.shape[0]
    num_patches_per_side = images.shape[-1] // patch_size

    mask = mask.view(
        batch_size,
        num_patches_per_side,
        num_patches_per_side
    )

    for b in range(batch_size):
        for row in range(num_patches_per_side):
            for col in range(num_patches_per_side):

                if mask[b, row, col]:

                    row_start = row * patch_size
                    row_end = row_start + patch_size

                    col_start = col * patch_size
                    col_end = col_start + patch_size

                    masked_images[
                        b,
                        :,
                        row_start:row_end,
                        col_start:col_end
                    ] = 0

    return masked_images


def masked_mse_loss(
    reconstruction,
    original,
    mask,
    patch_size=32
):
    """
    Calculate reconstruction loss only
    on masked patches.
    """

    batch_size = original.shape[0]
    num_patches_per_side = original.shape[-1] // patch_size

    mask = mask.view(
        batch_size,
        num_patches_per_side,
        num_patches_per_side
    )

    total_loss = 0.0
    masked_count = 0

    for b in range(batch_size):

        for row in range(num_patches_per_side):

            for col in range(num_patches_per_side):

                if mask[b, row, col]:

                    row_start = row * patch_size
                    row_end = row_start + patch_size

                    col_start = col * patch_size
                    col_end = col_start + patch_size

                    predicted_patch = reconstruction[
                        b,
                        :,
                        row_start:row_end,
                        col_start:col_end
                    ]

                    original_patch = original[
                        b,
                        :,
                        row_start:row_end,
                        col_start:col_end
                    ]

                    total_loss += F.mse_loss(
                        predicted_patch,
                        original_patch
                    )

                    masked_count += 1

    return total_loss / masked_count


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Real FIVES images
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
    )

    images, _ = next(iter(loader))

    images = images.to(device)

    print("Original images:", images.shape)

    # Create mask
    mask = create_patch_mask(
        batch_size=images.shape[0],
        image_size=512,
        patch_size=32,
        mask_ratio=0.75,
        device=device,
    )

    print("Masked patches:", mask.sum(dim=1))

    # Apply mask
    masked_images = apply_patch_mask(
        images,
        mask,
        patch_size=32
    )

    print("Masked images:", masked_images.shape)

    # MIM model
    model = MIMModel().to(device)

    model.eval()

    with torch.no_grad():

        reconstruction, representation = model(
            masked_images
        )

    print("Reconstruction:", reconstruction.shape)

    # Calculate masked reconstruction loss
    loss = masked_mse_loss(
        reconstruction,
        images,
        mask,
        patch_size=32
    )

    print("Masked reconstruction loss:", loss.item())


if __name__ == "__main__":
    main()
    