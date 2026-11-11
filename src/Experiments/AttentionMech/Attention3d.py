import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt3D(nn.Module):

    def __init__(self, channels, reduction=32, pooling="avg", gate_bias=0.0):
        super(CoordAtt3D, self).__init__()
        assert pooling in ("avg", "avgmax"), pooling
        self.pooling = pooling

        # keep one axis, collapse the other two
        self.avg_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.avg_w = nn.AdaptiveAvgPool3d((1, 1, None))
        if pooling == "avgmax":
            self.max_d = nn.AdaptiveMaxPool3d((None, 1, 1))
            self.max_h = nn.AdaptiveMaxPool3d((1, None, 1))
            self.max_w = nn.AdaptiveMaxPool3d((1, 1, None))

        in_c = channels * (2 if pooling == "avgmax" else 1)
        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv3d(in_c, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_d = nn.Conv3d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, channels, kernel_size=1, stride=1, padding=0)

        for conv in (self.conv_d, self.conv_h, self.conv_w):
            nn.init.constant_(conv.bias, gate_bias)

    def _descriptors(self, x):
        x_d = self.avg_d(x)  # (B, C, D, 1, 1)
        x_h = self.avg_h(x)  # (B, C, 1, H, 1)
        x_w = self.avg_w(x)  # (B, C, 1, 1, W)
        if self.pooling == "avgmax":
            x_d = torch.cat([x_d, self.max_d(x)], dim=1)  # (B, 2C, D, 1, 1)
            x_h = torch.cat([x_h, self.max_h(x)], dim=1)
            x_w = torch.cat([x_w, self.max_w(x)], dim=1)
        return x_d, x_h, x_w

    def forward(self, x):
        identity = x

        n, c, d, h, w = x.size()
        x_d, x_h, x_w = self._descriptors(x)

        # move each descriptor's long axis to dim 2 so they can be concatenated
        x_h = x_h.permute(0, 1, 3, 2, 4)  # (B, C', H, 1, 1)
        x_w = x_w.permute(0, 1, 4, 3, 2)  # (B, C', W, 1, 1)

        y = torch.cat([x_d, x_h, x_w], dim=2)  # (B, C', D+H+W, 1, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_d, y_h, y_w = torch.split(y, [d, h, w], dim=2)

        # put the H and W axes back where they belong (both permutes are their own inverse)
        y_h = y_h.permute(0, 1, 3, 2, 4)  # (B, mip, 1, H, 1)
        y_w = y_w.permute(0, 1, 4, 3, 2)  # (B, mip, 1, 1, W)

        a_d = self.conv_d(y_d).sigmoid()  # (B, C, D, 1, 1)
        a_h = self.conv_h(y_h).sigmoid()  # (B, C, 1, H, 1)
        a_w = self.conv_w(y_w).sigmoid()  # (B, C, 1, 1, W)

        return identity * a_d * a_h * a_w


class SE3D(nn.Module):
    """Squeeze-and-Excitation for 3D, the 'any attention helps' control."""

    def __init__(self, channels, reduction=32):
        super(SE3D, self).__init__()
        mip = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mip, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mip, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


def make_attn(kind, channels, reduction=32):
    """kind: None | "se" | "ca" | "ca_max"."""
    if kind is None:
        return nn.Identity()
    if kind == "se":
        return SE3D(channels, reduction)
    if kind == "ca":
        return CoordAtt3D(channels, reduction, pooling="avg")
    if kind == "ca_max":
        return CoordAtt3D(channels, reduction, pooling="avgmax")
    raise ValueError(f"Unknown attention kind: {kind}")