import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Pre-activation residual block used in WideResNet.

    Structure:
        BN -> ReLU -> Conv
        BN -> ReLU -> Dropout(optional) -> Conv
        + shortcut
    """

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_planes)

        # Second 3x3 convolution.
        # No further spatial downsampling here.
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.drop_rate = drop_rate

        self.equal_in_out = (
            in_planes == out_planes and stride == 1
        )

        self.shortcut = (
            None
            if self.equal_in_out
            else nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )
        )

    def forward(self, x):

        out = F.relu(self.bn1(x))

        shortcut = (
            x
            if self.equal_in_out
            else self.shortcut(out)
        )

        out = self.conv1(out)

        out = F.relu(self.bn2(out))

        if self.drop_rate > 0:
            out = F.dropout(
                out,
                p=self.drop_rate,
                training=self.training
            )

        out = self.conv2(out)

        out = out + shortcut

        return out


class WideResNet(nn.Module):
    """
    WideResNet for small images.

    Example:
        depth = 28
        widen_factor = 10

    gives WRN-28-10.
    """

    def __init__(
        self,
        depth=28,
        widen_factor=10,
        num_classes=10,
        in_channels=3,
        drop_rate=0.0
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0, \
            "WideResNet depth must satisfy depth = 6n + 4"

        n = (depth - 4) // 6


        k = widen_factor

        widths = [
            16,
            16 * k,
            32 * k,
            64 * k
        ]

        self.conv1 = nn.Conv2d(
            in_channels,
            widths[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.group1 = self._make_group(
            in_planes=widths[0],
            out_planes=widths[1],
            n=n,
            stride=1,
            drop_rate=drop_rate
        )

        self.group2 = self._make_group(
            in_planes=widths[1],
            out_planes=widths[2],
            n=n,
            stride=2,
            drop_rate=drop_rate
        )

        self.group3 = self._make_group(
            in_planes=widths[2],
            out_planes=widths[3],
            n=n,
            stride=2,
            drop_rate=drop_rate
        )

        self.bn = nn.BatchNorm2d(widths[3])

        self.fc = nn.Linear(
            widths[3],
            num_classes
        )

        self._init_weights()

    @staticmethod
    def _make_group(
        in_planes,
        out_planes,
        n,
        stride,
        drop_rate
    ):
        """
        Create one WideResNet residual group.

        Each group contains n BasicBlocks.

        Only the first block:
          - may change the number of channels
          - may perform spatial downsampling

        The remaining blocks keep the same dimensions.
        """

        layers = []

        for i in range(n):

            block_in_planes = (
                in_planes
                if i == 0
                else out_planes
            )

            block_stride = (
                stride
                if i == 0
                else 1
            )

            layers.append(
                BasicBlock(
                    in_planes=block_in_planes,
                    out_planes=out_planes,
                    stride=block_stride,
                    drop_rate=drop_rate
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Initialize Conv, BatchNorm, and Linear layers.
        """

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.conv1(x)

        x = self.group1(x)

        x = self.group2(x)

        x = self.group3(x)

        x = F.relu(self.bn(x))

        x = F.adaptive_avg_pool2d(x, output_size=1)

        x = x.flatten(1)

        x = self.fc(x)

        return x

def wrn_28_10(
    in_channels,
    num_classes=10,
    drop_rate=0.0
):
    """
    Convenience constructor for WRN-28-10.
    """

    return WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        in_channels=in_channels,
        drop_rate=drop_rate
    )


if __name__ == "__main__":

    model_gray = wrn_28_10(
        in_channels=1,
        num_classes=10
    )

    x_gray = torch.randn(
        2,  
        1,      
        28,
        28
    )

    y_gray = model_gray(x_gray)

    print(
        "Grayscale:",
        x_gray.shape,
        "->",
        y_gray.shape
    )

    model_rgb = wrn_28_10(
        in_channels=3,
        num_classes=10
    )

    x_rgb = torch.randn(
        2,
        3,
        32,
        32
    )

    y_rgb = model_rgb(x_rgb)

    print(
        "RGB:",
        x_rgb.shape,
        "->",
        y_rgb.shape
    )

    n_params = sum(
        p.numel()
        for p in model_rgb.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {n_params / 1e6:.2f}M"
    )