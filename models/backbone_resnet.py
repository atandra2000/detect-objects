"""From-scratch ResNet-50 + FPN backbone for Detect-Objects.

Channels_last memory format is applied by the caller (utils.memory) before
first forward.  No ImageNet pretraining — the backbone is trained jointly on
full COCO 2017.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """ResNet-50 bottleneck block with expansion=4."""

    expansion: int = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """ResNet-50 trunk; returns multi-scale feature maps C3..C5."""

    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class FPN(nn.Module):
    """Lightweight top-down FPN: C3/C4/C5 → P3/P4/P5 with shared output channels."""

    def __init__(self, in_channels: list[int] = (512, 1024, 2048), out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(32, out_channels),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, out_channels),
                )
            )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # features ordered high-res → low-res: [C3, C4, C5]
        laterals = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            prev = laterals[i]
            target_h, target_w = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(prev, size=(target_h, target_w), mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled
        outs = [f_conv(lat) for lat, f_conv in zip(laterals, self.fpn_convs)]
        return outs


class ResNet50FPN(nn.Module):
    """Convenience wrapper: ResNet-50 + FPN → three 256-ch feature levels."""

    def __init__(
        self,
        out_channels: int = 256,
        return_layers: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.backbone = ResNetBackbone(Bottleneck, (3, 4, 6, 3))
        self.return_layers = return_layers
        in_channels = [512, 1024, 2048]
        self.fpn = FPN(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(x)
        selected = [features[i] for i in self.return_layers]
        return self.fpn(selected)


def build_resnet50_fpn(cfg: dict) -> ResNet50FPN:
    backbone_cfg = cfg.get("model", {}).get("backbone", {})
    fpn_cfg = cfg.get("model", {}).get("fpn", {})
    return ResNet50FPN(
        out_channels=fpn_cfg.get("out_channels", 256),
        return_layers=tuple(backbone_cfg.get("out_indices", [1, 2, 3])),
    )
