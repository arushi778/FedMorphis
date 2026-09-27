import torch
import torch.nn as nn

from shared_encoder import SharedEncoder


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        # Handle any possible spatial-size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class SharedUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Same encoder used during MIM pretraining
        self.encoder = SharedEncoder()

        # Decoder
        self.up4 = UpBlock(
            in_channels=1024,
            skip_channels=512,
            out_channels=512
        )

        self.up3 = UpBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.up2 = UpBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.up1 = UpBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        features = self.encoder(x)

        f1, f2, f3, f4, bottleneck = features

        x = self.up4(bottleneck, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        output = self.final(x)

        return output


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SharedUNet().to(device)

    x = torch.randn(
        2,
        3,
        512,
        512,
        device=device
    )

    with torch.no_grad():
        output = model(x)

    print("Device:", device)
    print("Input:", x.shape)
    print("Output:", output.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )