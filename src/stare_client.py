
from pathlib import Path
from retinal_dataset import RetinalDataset


def create_stare_client(image_size=512):
    image_dir = Path("data/STARE/raw_images")
    mask_dir = Path("data/STARE/raw_masks")

    samples = []

    for image_path in sorted(image_dir.glob("*.ppm")):
        mask_path = mask_dir / f"{image_path.stem}.ah.ppm"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        samples.append((image_path, mask_path))

    if not samples:
        raise RuntimeError("No STARE images found.")

    return RetinalDataset(samples, image_size=image_size)


if __name__ == "__main__":
    dataset = create_stare_client()

    print("STARE client dataset")
    print("--------------------")
    print("Number of samples:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", mask.unique())
