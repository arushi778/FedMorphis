
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared_unet import SharedUNet
from fives_client import create_fives_client
from drive_client import create_drive_client
from stare_client import create_stare_client
from chase_client import create_chase_client
from fedavg import fedavg


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
LOCAL_EPOCHS = 1
MAX_BATCHES = 2  # Smoke test only; set to None for full local epochs
LEARNING_RATE = 1e-4


def dice_loss(logits, masks, smooth=1e-6):
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    intersection = (probabilities * masks).sum(dim=1)
    denominator = probabilities.sum(dim=1) + masks.sum(dim=1)

    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def train_local(model, dataset, client_name):
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    batches = 0

    for epoch in range(LOCAL_EPOCHS):
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            if MAX_BATCHES is not None and batches >= MAX_BATCHES:
                break

        if MAX_BATCHES is not None and batches >= MAX_BATCHES:
            break

    average_loss = total_loss / max(batches, 1)
    print(f"{client_name}: batches={batches}, loss={average_loss:.4f}")

    return model.state_dict()


def main():
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    clients = {
        "FIVES": create_fives_client(),
        "DRIVE": create_drive_client(),
        "STARE": create_stare_client(),
        "CHASE-DB1": create_chase_client(),
    }

    global_model = SharedUNet().to(DEVICE)
    client_states = []
    sample_counts = []

    print("\nStarting federated round 1")

    for name, dataset in clients.items():
        print(f"\nTraining client: {name} ({len(dataset)} samples)")

        local_model = copy.deepcopy(global_model)
        local_state = train_local(local_model, dataset, name)

        client_states.append(local_state)
        sample_counts.append(len(dataset))

        del local_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    aggregated_state = fedavg(client_states, sample_counts)
    global_model.load_state_dict(aggregated_state)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "fedavg_round1_smoke.pth"
    torch.save(global_model.state_dict(), checkpoint_path)

    print("\nFederated round 1 completed.")
    print("Aggregated clients:", len(clients))
    print("Checkpoint saved to:", checkpoint_path)


if __name__ == "__main__":
    main()
