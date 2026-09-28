from fives_torch_dataset import FIVESDataset


def create_fives_client(image_size=512):
    """
    Create the FIVES federated client dataset.

    Uses the 600-image FIVES training pool.
    """

    parquet_paths = [
        "data/FIVES/data/train-00000-of-00003.parquet",
        "data/FIVES/data/train-00001-of-00003.parquet",
        "data/FIVES/data/train-00002-of-00003.parquet",
    ]

    dataset = FIVESDataset(
        parquet_paths,
        image_size=image_size
    )

    return dataset


if __name__ == "__main__":

    dataset = create_fives_client()

    print("FIVES client dataset")
    print("--------------------")
    print("Number of samples:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", mask.unique())