"""Ball heatmap detector: a TrackNetV2-style U-Net.

Input: three consecutive RGB frames stacked on the channel axis (9 x H x W, values 0..1), causal
(frames t-2, t-1, t). Output: one heatmap of logits (1 x H x W) for the ball position in frame t.

The shipped model uses PP_WIDTH=0.5 (channels 32/64/128/256, 2.83 M parameters) at 640 x 360.
"""
import os
import torch, torch.nn as nn, torch.nn.functional as F

W, H = int(os.environ.get("PP_W", 640)), int(os.environ.get("PP_H", 360))   # model resolution
WIDTH = float(os.environ.get("PP_WIDTH", 1.0))                               # channel width multiplier


def cbr(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class TrackNet(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        c = c or tuple(int(x * WIDTH) for x in (64, 128, 256, 512))
        self.e1 = nn.Sequential(cbr(9, c[0]), cbr(c[0], c[0]))
        self.e2 = nn.Sequential(cbr(c[0], c[1]), cbr(c[1], c[1]))
        self.e3 = nn.Sequential(cbr(c[1], c[2]), cbr(c[2], c[2]), cbr(c[2], c[2]))
        self.b = nn.Sequential(cbr(c[2], c[3]), cbr(c[3], c[3]), cbr(c[3], c[3]))
        self.d3 = nn.Sequential(cbr(c[3] + c[2], c[2]), cbr(c[2], c[2]), cbr(c[2], c[2]))
        self.d2 = nn.Sequential(cbr(c[2] + c[1], c[1]), cbr(c[1], c[1]))
        self.d1 = nn.Sequential(cbr(c[1] + c[0], c[0]), cbr(c[0], c[0]))
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="nearest")
        d3 = self.d3(torch.cat([up(b, e3), e3], 1)); d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return self.head(d1)  # logits
