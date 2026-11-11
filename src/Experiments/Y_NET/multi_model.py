import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextBlock, self).__init__()
        self.d1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, dilation=1
        )
        self.d2 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=2, dilation=2
        )
        self.d4 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=4, dilation=4
        )
        self.d8 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=8, dilation=8
        )

        self.fusion = nn.Sequential(
            nn.BatchNorm3d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        o1 = self.d1(x)
        o2 = self.d2(x)
        o4 = self.d4(x)
        o8 = self.d8(x)
        return self.fusion(torch.cat([o1, o2, o4, o8], dim=1))


class SpineResUNet(nn.Module):
    """
    Y-Net: one shared encoder (3 levels + Context Block), two decoders.
      - segmentation decoder -> `out_channels` logits (29 classes)
      - reconstruction decoder -> `recon` channels in [0, 1] (sigmoid), 1 for the clean image
    Input has `in_channels` = 2: the (possibly masked) image and the mask map.
    """

    def __init__(self, in_channels=2, out_channels=29, recon=1, base_filters=16):
        super(SpineResUNet, self).__init__()

        # ---------------- shared encoder ----------------
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResidualBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ContextBlock(base_filters * 4, base_filters * 8)

        # ---------------- segmentation decoder ----------------
        self.up3 = nn.ConvTranspose3d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose3d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)

        self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)

        # ---------------- reconstruction decoder (the Y arm) ----------------
        self.yup3 = nn.ConvTranspose3d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.ydec3 = ResidualBlock(base_filters * 8, base_filters * 4)

        self.yup2 = nn.ConvTranspose3d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.ydec2 = ResidualBlock(base_filters * 4, base_filters * 2)

        self.yup1 = nn.ConvTranspose3d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.ydec1 = ResidualBlock(base_filters * 2, base_filters)

        self.yfinal = nn.Conv3d(base_filters, recon, kernel_size=1)

    def forward(self, x, with_recon=True):
        # shared encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        # segmentation decoder
        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(
                d3, size=e3.shape[2:], mode="trilinear", align_corners=False
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(
                d2, size=e2.shape[2:], mode="trilinear", align_corners=False
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(
                d1, size=e1.shape[2:], mode="trilinear", align_corners=False
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        seg_logits = self.final(d1)

        if not with_recon:
            return seg_logits, None

        # reconstruction decoder
        dy3 = self.yup3(b)
        if dy3.shape != e3.shape:
            dy3 = F.interpolate(
                dy3, size=e3.shape[2:], mode="trilinear", align_corners=False
            )
        dy3 = torch.cat([dy3, e3], dim=1)
        dy3 = self.ydec3(dy3)

        dy2 = self.yup2(dy3)
        if dy2.shape != e2.shape:
            dy2 = F.interpolate(
                dy2, size=e2.shape[2:], mode="trilinear", align_corners=False
            )
        dy2 = torch.cat([dy2, e2], dim=1)
        dy2 = self.ydec2(dy2)

        dy1 = self.yup1(dy2)
        if dy1.shape != e1.shape:
            dy1 = F.interpolate(
                dy1, size=e1.shape[2:], mode="trilinear", align_corners=False
            )
        dy1 = torch.cat([dy1, e1], dim=1)
        dy1 = self.ydec1(dy1)

        recon_out = torch.sigmoid(self.yfinal(dy1))

        return seg_logits, recon_out


if __name__ == "__main__":
    model = SpineResUNet()
    test_input = torch.randn(1, 2, 128, 128, 64)
    print(f"Model created. Testing with input {test_input.shape}...")
    seg, recon = model(test_input)
    print(f"Seg shape: {seg.shape}")
    print(f"Recon shape: {recon.shape}")
