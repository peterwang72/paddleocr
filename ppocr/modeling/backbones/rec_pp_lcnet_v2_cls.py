# Copyright (c) 2025 PaddleOCR Authors. All Rights Reserved.
#
# PP-LCNetV2 骨干 + ClsHead，用于文本框多分类（如手写/印刷/符号）。
# 与 mobile_v5 中说明一致；输出 4D 特征图，由 BaseModel 接 ClsHead。
from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm2D, Conv2D
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

__all__ = ["PPLCNetV2_cls", "PPLCNetV2_cls_base"]


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups=1, use_act=True
    ):
        super().__init__()
        self.use_act = use_act
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False,
        )
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)),
        )
        self.act = nn.ReLU() if self.use_act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(channel, channel // reduction, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(channel // reduction, channel, kernel_size=1, stride=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gate(x)
        return paddle.multiply(identity, x)


class DepthwiseSeparable(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size=3,
        split_pw=False,
        use_se=False,
        use_shortcut=False,
    ):
        super().__init__()
        self.use_se = use_se
        self.split_pw = split_pw
        self.use_shortcut = (
            True
            if use_shortcut and stride == 1 and in_channels == out_channels
            else False
        )
        self.dw_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
        )
        if use_se:
            self.se = SEModule(in_channels)
        if split_pw:
            mid_channels = int(out_channels * 0.5)
            self.pw_conv_1 = ConvBNLayer(
                in_channels, mid_channels, kernel_size=1, stride=1
            )
            self.pw_conv_2 = ConvBNLayer(
                mid_channels, out_channels, kernel_size=1, stride=1
            )
        else:
            self.pw_conv = ConvBNLayer(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        if self.split_pw:
            x = self.pw_conv_1(x)
            x = self.pw_conv_2(x)
        else:
            x = self.pw_conv(x)
        if self.use_shortcut:
            x = x + identity
        return x


class PPLCNetV2_cls(nn.Layer):
    """
    PP-LCNetV2 分类骨干：输出最后一层特征图 (N,C,H,W)，供 ClsHead 使用。
    结构与 mobile_v5 中原 PPLCNetV2Binary 卷积部分一致，不包含分类头。
    """

    NET_CONFIG = {
        "stage1": [64, 3, False, False, False],
        "stage2": [128, 3, False, False, False],
        "stage3": [256, 5, True, True, False],
        "stage4": [512, 5, False, False, True],
    }

    def __init__(self, in_channels=3, scale=1.0, depths=(2, 2, 6, 2), **kwargs):
        super().__init__()
        if isinstance(depths, list):
            depths = tuple(depths)
        self.stem = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=make_divisible(32 * scale),
                kernel_size=3,
                stride=2,
            ),
            DepthwiseSeparable(
                in_channels=make_divisible(32 * scale),
                out_channels=make_divisible(64 * scale),
                stride=1,
                dw_size=3,
            ),
        )

        self.stages = nn.LayerList()
        for depth_idx, key in enumerate(self.NET_CONFIG):
            in_ch_cfg, kernel_size, split_pw, use_se, use_shortcut = self.NET_CONFIG[key]
            blocks = []
            for i in range(depths[depth_idx]):
                blocks.append(
                    DepthwiseSeparable(
                        in_channels=make_divisible(
                            (in_ch_cfg if i == 0 else in_ch_cfg * 2) * scale
                        ),
                        out_channels=make_divisible(in_ch_cfg * 2 * scale),
                        stride=2 if i == 0 else 1,
                        dw_size=kernel_size,
                        split_pw=split_pw,
                        use_se=use_se,
                        use_shortcut=use_shortcut,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))

        last_channels = make_divisible(self.NET_CONFIG["stage4"][0] * 2 * scale)
        self.out_channels = last_channels

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


def PPLCNetV2_cls_base(in_channels=3, **kwargs):
    """与检测配置中 PPLCNetV2_base 同深宽规格，用于 OCR 配套三分类训练。"""
    kwargs.pop("return_all_feats", None)
    scale = kwargs.pop("scale", 1.0)
    depths = kwargs.pop("depths", (2, 2, 6, 2))
    if isinstance(depths, list):
        depths = tuple(depths)
    return PPLCNetV2_cls(in_channels=in_channels, scale=scale, depths=depths, **kwargs)
