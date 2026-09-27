import torch
import torch.nn as nn

from shared_encoder import SharedEncoder


class SharedMIMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SharedEncoder()

        # The bottleneck from SharedEncoder has 1024 channels
        # and spatial size 32x32 for a 512x512 input.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 512,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                512, 256,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 3,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        features = self.encoder(x)

        bottleneck = features[-1]

        reconstruction = self.decoder(bottleneck)

        return reconstruction, features


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SharedMIMModel().to(device)

    x = torch.randn(
        2, 3, 512, 512,
        device=device
    )

    with torch.no_grad():
        reconstruction, features = model(x)

    print("Device:", device)
    print("Input:", x.shape)

    for i, feature in enumerate(features):
        print(
            f"Feature {i + 1}:",
            feature.shape
        )

    print("Reconstruction:", reconstruction.shape)

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )