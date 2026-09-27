import torch
import torch.nn as nn


class MIMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=2, stride=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 64, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 3, kernel_size=3, padding=1
            ),
        )

    def forward(self, x):
        representation = self.encoder(x)
        reconstruction = self.decoder(representation)

        return reconstruction, representation


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MIMModel().to(device)

    # Simulated masked retinal image
    x = torch.randn(
        2, 3, 512, 512,
        device=device
    )

    with torch.no_grad():
        reconstruction, representation = model(x)

    print("Device:", device)
    print("Input shape:", x.shape)
    print("Representation shape:", representation.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )
    