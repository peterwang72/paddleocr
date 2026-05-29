from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle.vision.models import mobilenet_v3_small


class MobileNetV3Binary(nn.Layer):
    """
    Minimal binary classifier based on Paddle MobileNetV3-small.
    """

    def __init__(self, num_classes=2, dropout=0.2):
        super().__init__()
        # Build backbone from Paddle vision model zoo.
        base = mobilenet_v3_small(pretrained=False, num_classes=1000)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2D(1)

        in_features = 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x, return_feat=False):
        x = self.features(x)
        feat = self.avgpool(x)
        feat = paddle.flatten(feat, start_axis=1)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        return logits


def MobileNetV3_binary_small(num_classes=2, **kwargs):
    return MobileNetV3Binary(num_classes=num_classes, **kwargs)
