import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=0.25):
        super().__init__()
        hidden = max(8, int(c * r))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class UIB(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=6.0):
        super().__init__()
        hidden = max(16, int(round(c1 * e)))
        self.use_res = (s == 1 and c1 == c2)

        self.block = nn.Sequential(
            ConvBNAct(c1, hidden, k=1, s=1, act=True),
            ConvBNAct(hidden, hidden, k=k, s=s, g=hidden, act=True),
            SqueezeExcite(hidden, r=0.25),
            ConvBNAct(hidden, c2, k=1, s=1, act=False),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class UIBDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, e=6.0):
        super().__init__()
        hidden = max(16, int(round(c1 * e)))

        self.block = nn.Sequential(
            ConvBNAct(c1, hidden, k=1, s=1, act=True),
            ConvBNAct(hidden, hidden, k=k, s=s, g=hidden, act=True),
            SqueezeExcite(hidden, r=0.25),
            ConvBNAct(hidden, c2, k=1, s=1, act=False),
        )

    def forward(self, x):
        return self.block(x)