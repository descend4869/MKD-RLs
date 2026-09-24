"""ImageNet-style RegNetX-200MF."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """A convolution followed by batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class RegNetXBlock(nn.Module):
    """RegNet bottleneck block without squeeze-and-excitation (RegNetX)."""

    def __init__(self, in_channels, out_channels, stride, group_width=8):
        super().__init__()

        # RegNet uses a bottleneck ratio of 1.0 for this configuration.
        bottleneck_channels = out_channels
        if bottleneck_channels % group_width != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"group_width ({group_width})"
            )

        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=bottleneck_channels // group_width,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, return_preact=False):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        preact = out + self.shortcut(x)
        # Keep ``preact`` intact because it can be returned to distillation
        # code when ``preact=True``.
        out = F.relu(preact, inplace=False)

        if return_preact:
            return out, preact
        return out


class RegNetXImageNet(nn.Module):
    """RegNetX-200MF adapted for ImageNet-sized inputs."""

    def __init__(self, num_classes=1000):
        super().__init__()

        # RegNetX-200MF design parameters.
        widths = [24, 56, 152, 368]
        depths = [1, 1, 4, 7]
        strides = [2, 2, 2, 2]
        group_width = 8
        stem_width = 32

        self.stem = ConvBNReLU(3, stem_width, kernel_size=3, stride=2)
        self.stages = nn.ModuleList()

        in_channels = stem_width
        for width, depth, stride in zip(widths, depths, strides):
            blocks = [
                RegNetXBlock(
                    in_channels,
                    width,
                    stride=stride,
                    group_width=group_width,
                )
            ]
            blocks.extend(
                RegNetXBlock(
                    width,
                    width,
                    stride=1,
                    group_width=group_width,
                )
                for _ in range(depth - 1)
            )
            self.stages.append(nn.Sequential(*blocks))
            in_channels = width

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(widths[-1], num_classes)

        self._initialize_weights()

    def get_feat_modules(self):
        """Return the main feature modules for compatibility with the repo."""
        return nn.ModuleList([self.stem, *self.stages])

    def distill_seq(self):
        """Return a simple sequential view used by older distillation code."""
        return nn.ModuleList(
            [
                self.stages[0],
                self.stages[1],
                self.stages[2],
                self.stages[3],
                nn.Sequential(self.avgpool, nn.Flatten(), self.classifier),
            ]
        )

    def forward(self, x, is_feat=False, preact=False, is_lr_adaptive=False):
        features = []
        preact_features = []

        x = self.stem(x)
        features.append(x)

        for stage in self.stages:
            stage_preact = None
            for block in stage:
                if preact:
                    x, stage_preact = block(x, return_preact=True)
                else:
                    x = block(x)
            features.append(x)
            if preact:
                preact_features.append(stage_preact)

        pool = self.avgpool(x)
        embedding = torch.flatten(pool, 1)
        logits = self.classifier(embedding)

        if not is_feat:
            return logits

        # Keep the same convention as the existing RegNet implementation:
        # features[-2] is a feature map and features[-1] is a vector.
        output_features = [features[0], *features[1:], embedding]
        if preact:
            output_features = [features[0], *preact_features, embedding]
        elif is_lr_adaptive:
            output_features = [features[0], *features[1:], pool, embedding]

        return output_features, logits

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


def RegNetX_200MF_ImageNet(num_classes=1000):
    """Construct an ImageNet-style RegNetX-200MF student."""
    return RegNetXImageNet(num_classes=num_classes)


if __name__ == "__main__":
    model = RegNetX_200MF_ImageNet(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    features, logits = model(x, is_feat=True)
    print("logits:", logits.shape)
    print("features:", [feature.shape for feature in features])
