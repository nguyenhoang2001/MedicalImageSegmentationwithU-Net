"""
U-Net Architecture Implementation for Image Segmentation

Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
       by Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)

Architecture:
- Encoder (Contracting Path): Downsampling with max pooling
- Bottleneck: Bridge between encoder and decoder
- Decoder (Expanding Path): Upsampling with transpose convolutions
- Skip Connections: Concatenate encoder features with decoder features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) * 2

    This is the basic building block used in both encoder and decoder.
    Each block consists of two 3x3 convolutions, each followed by
    batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int, optional): Number of intermediate channels.
                                         If None, uses out_channels
        """
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling Block: MaxPool2d -> DoubleConv

    Used in the encoder (contracting path) to reduce spatial dimensions
    while increasing the number of feature channels.
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling Block: ConvTranspose2d/Upsample -> Concatenate -> DoubleConv

    Used in the decoder (expanding path) to increase spatial dimensions
    and combine with skip connections from the encoder.
    """

    def __init__(self, in_channels, out_channels, bilinear=False):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bilinear (bool): Use bilinear upsampling (True) or transpose convolution (False)
        """
        super(Up, self).__init__()

        # Use bilinear upsampling or transpose convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Input from the previous decoder layer
            x2: Skip connection from the encoder

        Returns:
            Upsampled and concatenated features
        """
        x1 = self.up(x1)

        # Handle input sizes that are not perfectly divisible
        # Pad x1 to match x2's size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Convolution: 1x1 Conv to produce final segmentation mask
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture

    A fully convolutional network for semantic segmentation with:
    - Encoder: 4 downsampling blocks
    - Bottleneck: 1 middle block
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 1x1 convolution for final prediction
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        """
        Args:
            n_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            n_classes (int): Number of output classes (1 for binary segmentation)
            bilinear (bool): Use bilinear upsampling instead of transpose convolution
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial convolution (no downsampling)
        self.inc = DoubleConv(n_channels, 64)

        # Encoder (Contracting Path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder (Expanding Path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the U-Net

        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, n_classes, height, width)
        """
        # Encoder with skip connections
        x1 = self.inc(x)  # First level
        x2 = self.down1(x1)  # Second level
        x3 = self.down2(x2)  # Third level
        x4 = self.down3(x3)  # Fourth level
        x5 = self.down4(x4)  # Bottleneck

        # Decoder with skip connections
        x = self.up1(x5, x4)  # Concatenate with x4
        x = self.up2(x, x3)  # Concatenate with x3
        x = self.up3(x, x2)  # Concatenate with x2
        x = self.up4(x, x1)  # Concatenate with x1

        # Output
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """
        Enable gradient checkpointing to save memory during training.
        Useful for training with larger batch sizes or higher resolution images.
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


def test_model():
    """
    Test function to verify the U-Net model
    """
    # Create a sample input tensor (batch_size=2, channels=3, height=256, width=256)
    x = torch.randn(2, 3, 256, 256)

    # Initialize model
    model = UNet(n_channels=3, n_classes=1)

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (2, 1, 256, 256), "Output shape mismatch!"
    print("✓ Model test passed!")


if __name__ == "__main__":
    test_model()
