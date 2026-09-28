
from pathlib import Path
from retinal_dataset import RetinalDataset


def create_drive_client(image_size=512):
    image_dir = Path("data/DRIVE/training/images")
    mask_dir = Path("data/DRIVE/training/1st_manual")

    samples = []

    for image_path in sorted(image_dir.glob("*.tif")):
        image_id = image_path.stem.split("_")[0]
        mask_path = mask_dir / f"{image_id}_manual1.gif"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        samples.append((image_path, mask_path))

    if not samples:
        raise RuntimeError("No DRIVE training images found.")

    return RetinalDataset(samples, image_size=image_size)


if __name__ == "__main__":
    dataset = create_drive_client()

    print("DRIVE client dataset")
    print("--------------------")
    print("Number of samples:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", mask.unique())
