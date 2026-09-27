import torch


def create_patch_mask(
    batch_size,
    image_size=512,
    patch_size=32,
    mask_ratio=0.75,
    device="cpu",
):
    num_patches_per_side = image_size // patch_size
    num_patches = num_patches_per_side ** 2

    num_masked = int(num_patches * mask_ratio)

    mask = torch.zeros(
        batch_size,
        num_patches,
        dtype=torch.bool,
        device=device,
    )

    for i in range(batch_size):
        indices = torch.randperm(
            num_patches,
            device=device
        )[:num_masked]

        mask[i, indices] = True

    return mask


if __name__ == "__main__":

    mask = create_patch_mask(
        batch_size=2,
        image_size=512,
        patch_size=32,
        mask_ratio=0.75,
    )

    print("Mask shape:", mask.shape)
    print("Masked patches per image:", mask[0].sum().item())
    print("Total patches:", mask.shape[1])
    print("Mask ratio:", mask[0].float().mean().item())