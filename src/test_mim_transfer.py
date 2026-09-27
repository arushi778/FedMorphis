import torch

from shared_unet import SharedUNet


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Create segmentation model
    model = SharedUNet().to(device)

    # Load MIM checkpoint
    checkpoint = torch.load(
        "checkpoints/shared_mim_pretrained.pth",
        map_location=device
    )

    mim_state = checkpoint["model_state_dict"]

    # Extract only encoder weights
    encoder_state = {
        key.replace("encoder.", "", 1): value
        for key, value in mim_state.items()
        if key.startswith("encoder.")
    }

    # Load into Shared U-Net encoder
    result = model.encoder.load_state_dict(
        encoder_state,
        strict=True
    )

    print("MIM encoder loaded successfully.")
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    # Forward test
    x = torch.randn(
        1,
        3,
        512,
        512,
        device=device
    )

    with torch.no_grad():
        output = model(x)

    print("Input:", x.shape)
    print("Segmentation output:", output.shape)


if __name__ == "__main__":
    main()