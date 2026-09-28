
from fives_client import create_fives_client
from drive_client import create_drive_client
from stare_client import create_stare_client
from chase_client import create_chase_client

clients = {
    "FIVES": create_fives_client(),
    "DRIVE": create_drive_client(),
    "STARE": create_stare_client(),
    "CHASE-DB1": create_chase_client(),
}

print("\nFEDERATED CLIENT VALIDATION")
print("=" * 45)

total_samples = 0

for name, dataset in clients.items():
    image, mask = dataset[0]

    assert image.shape == (3, 512, 512), f"{name}: Incorrect image shape"
    assert mask.shape == (1, 512, 512), f"{name}: Incorrect mask shape"
    assert image.min() >= 0 and image.max() <= 1, f"{name}: Invalid image range"
    assert mask.min() >= 0 and mask.max() <= 1, f"{name}: Invalid mask range"

    count = len(dataset)
    total_samples += count

    print(f"\n{name}")
    print(f"  Samples: {count}")
    print(f"  Image shape: {tuple(image.shape)}")
    print(f"  Mask shape: {tuple(mask.shape)}")
    print("  Status: PASSED")

print("\n" + "=" * 45)
print(f"Total samples: {total_samples}")
print("All clients validated successfully.")
