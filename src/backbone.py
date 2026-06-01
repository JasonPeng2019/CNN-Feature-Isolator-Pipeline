"""CIFAR ResNet (He et al. 2016) with named section taps Q1..Q5.

Widths 16/32/64 over 3 stages of `n` blocks. ResNet-20: n=3, ResNet-56: n=9.
Taps are post-activation hidden maps at 5 block-group boundaries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, n=9, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, n, 1)
        self.stage2 = self._make_stage(32, n, 2)
        self.stage3 = self._make_stage(64, n, 2)
        self.fc = nn.Linear(64, num_classes)
        self.n = n

    def _make_stage(self, planes, n, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(n - 1):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _run_stage(self, stage, x, mid_idx=None):
        """Run a stage; optionally also return the activation after block `mid_idx`."""
        mid = None
        for i, block in enumerate(stage):
            x = block(x)
            if mid_idx is not None and i == mid_idx:
                mid = x
        return x, mid

    def forward_with_taps(self, x):
        """Returns (logits, {Q1..Q5}) — post-activation hidden maps.
        Q1 post-stem (16,32x32), Q2 end-stage1 (16,32x32),
        Q3 end-stage2 (32,16x16), Q4 mid-stage3 (64,8x8), Q5 end-stage3 (64,8x8).
        """
        taps = {}
        x = F.relu(self.bn1(self.conv1(x)))
        taps["Q1"] = x
        x, _ = self._run_stage(self.stage1, x)
        taps["Q2"] = x
        x, _ = self._run_stage(self.stage2, x)
        taps["Q3"] = x
        x, mid = self._run_stage(self.stage3, x, mid_idx=self.n // 2)
        taps["Q4"] = mid
        taps["Q5"] = x
        logits = self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
        return logits, taps

    def forward(self, x):
        return self.forward_with_taps(x)[0]

    def forward_from(self, section, h):
        """Continue the frozen forward pass from `section` given a (possibly
        reconstructed) hidden map `h`. Used for downstream-preservation eval."""
        order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        idx = order.index(section)
        x = h
        if idx < order.index("Q2"):
            x, _ = self._run_stage(self.stage1, x)
        if idx < order.index("Q3"):
            x, _ = self._run_stage(self.stage2, x)
        if section == "Q4":
            # resume stage3 after its mid block
            for block in list(self.stage3)[self.n // 2 + 1:]:
                x = block(x)
        elif idx < order.index("Q4"):
            x, _ = self._run_stage(self.stage3, x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


    def forward_between(self, src, dst, h):
        """Run the frozen blocks mapping the activation at tap `src` to tap `dst`
        (must be adjacent in Q1..Q5). Used by the frozen-block-hybrid transition."""
        pairs = {
            ("Q1", "Q2"): lambda x: self._run_stage(self.stage1, x)[0],
            ("Q2", "Q3"): lambda x: self._run_stage(self.stage2, x)[0],
            ("Q3", "Q4"): lambda x: self._run_blocks(self.stage3, x, 0, self.n // 2 + 1),
            ("Q4", "Q5"): lambda x: self._run_blocks(self.stage3, x, self.n // 2 + 1, self.n),
        }
        return pairs[(src, dst)](h)

    def _run_blocks(self, stage, x, lo, hi):
        for block in list(stage)[lo:hi]:
            x = block(x)
        return x


SECTION_INFO = {  # (channels, spatial) for ResNet-56/20 CIFAR widths
    "Q1": (16, 32), "Q2": (16, 32), "Q3": (32, 16), "Q4": (64, 8), "Q5": (64, 8),
}


def resnet20(num_classes=100):
    return CifarResNet(n=3, num_classes=num_classes)


def resnet56(num_classes=100):
    return CifarResNet(n=9, num_classes=num_classes)


def resnet110(num_classes=100):
    return CifarResNet(n=18, num_classes=num_classes)
