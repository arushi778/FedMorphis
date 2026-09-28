
from pathlib import Path
from retinal_dataset import RetinalDataset


def create_chase_client(image_size=512):
    data_dir = Path("data/CHASE_DB1/raw")

    samples = []

    for image_path in sorted(data_dir.glob("*.jpg")):
        mask_path = data_dir / f"{image_path.stem}_1stHO.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        samples.append((image_path, mask_path))

    if not samples:
        raise RuntimeError("No CHASE-DB1 images found.")

    return RetinalDataset(samples, image_size=image_size)


if __name__ == "__main__":
    dataset = create_chase_client()

    print("CHASE-DB1 client dataset")
    print("------------------------")
    print("Number of samples:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", mask.unique())
