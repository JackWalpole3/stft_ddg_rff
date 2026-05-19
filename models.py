import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=1, norm: bool = True):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=not norm)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STFTIdentityEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, z_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=1),
            ConvBlock(256, 256, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, z_dim), nn.BatchNorm1d(z_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.fc(self.pool(h))


class STFTVariationEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, out_channels, stride=2),
            ConvBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArcMarginProduct(nn.Module):
    """Cosine/L2Softmax classifier. With m=0.0 this is DR-RFF style L2Softmax."""

    def __init__(self, in_features: int, out_features: int, s: float = 10.0, m: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None or self.m == 0.0:
            return cosine * self.s
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, z_id_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=False)
        self.film = nn.Linear(z_id_dim, channels * 4)

    def forward(self, x: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.film(z_id).chunk(4, dim=1)
        gamma1 = gamma1[:, :, None, None]
        beta1 = beta1[:, :, None, None]
        gamma2 = gamma2[:, :, None, None]
        beta2 = beta2[:, :, None, None]

        h = self.conv1(x)
        h = self.norm1(h) * (1.0 + gamma1) + beta1
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv2(h)
        h = self.norm2(h) * (1.0 + gamma2) + beta2
        return F.leaky_relu(x + h, 0.2, inplace=True)


class STFTFiLMGenerator(nn.Module):
    def __init__(self, z_var_channels: int = 128, z_id_dim: int = 256, out_channels: int = 2):
        super().__init__()
        self.res1 = FiLMResBlock(z_var_channels, z_id_dim)
        self.up1 = ConvBlock(z_var_channels, 128, stride=1)
        self.res2 = FiLMResBlock(128, z_id_dim)
        self.up2 = ConvBlock(128, 64, stride=1)
        self.res3 = FiLMResBlock(64, z_id_dim)
        self.up3 = ConvBlock(64, 32, stride=1)
        self.out = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, z_var: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:
        h = self.res1(z_var, z_id)
        h = F.interpolate(h, size=(16, 7), mode="bilinear", align_corners=False)
        h = self.up1(h)
        h = self.res2(h, z_id)
        h = F.interpolate(h, size=(32, 13), mode="bilinear", align_corners=False)
        h = self.up2(h)
        h = self.res3(h, z_id)
        h = F.interpolate(h, size=(64, 25), mode="bilinear", align_corners=False)
        h = self.up3(h)
        return self.out(h)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 32, num_scales: int = 2):
        super().__init__()
        self.num_scales = num_scales
        self.blocks = nn.ModuleList([self._make_block(in_channels, base) for _ in range(num_scales)])

    @staticmethod
    def _make_block(in_channels: int, base: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, base, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, 1, 3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        outputs = []
        cur = x
        for i, block in enumerate(self.blocks):
            outputs.append(block(cur))
            if i + 1 < self.num_scales:
                cur = F.avg_pool2d(cur, kernel_size=3, stride=2, padding=1)
        return outputs


class STFTDDGModel(nn.Module):
    def __init__(self, num_classes: int = 6, z_id_dim: int = 256, z_var_channels: int = 128):
        super().__init__()
        self.eid = STFTIdentityEncoder(z_dim=z_id_dim)
        self.evar = STFTVariationEncoder(out_channels=z_var_channels)
        self.generator = STFTFiLMGenerator(z_var_channels=z_var_channels, z_id_dim=z_id_dim)
        self.classifier = ArcMarginProduct(z_id_dim, num_classes, s=10.0, m=0.0)

    def classify(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.eid(x), labels)

    def encode_pair(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.evar(x), self.eid(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z_var, z_id = self.encode_pair(x)
        return self.generator(z_var, z_id)

