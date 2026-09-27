from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parquet_paths = [
        "data/FIVES/data/train-00000-of-00003.parquet",
        "data/FIVES/data/train-00001-of-00003.parquet",
        "data/FIVES/data/train-00002-of-00003.parquet",
    ]

    # Load all 600 training samples
    dataframes = [
        pd.read_parquet(path)
        for path in parquet_paths
    ]

    df = pd.concat(dataframes, ignore_index=True)

    print("Total samples:", len(df))

    # Create reproducible 80/20 split
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))

    # Save indices rather than copying the actual images
    split_dir = Path("data/FIVES/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df.index.to_series().to_csv(
        split_dir / "train_indices.csv",
        index=False,
        header=["index"],
    )

    val_df.index.to_series().to_csv(
        split_dir / "val_indices.csv",
        index=False,
        header=["index"],
    )

    print("Split files saved to:", split_dir)


if __name__ == "__main__":
    main()