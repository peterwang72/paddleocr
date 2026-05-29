# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DB 检测头（det_db_head.py）— 中文说明
====================================
本文件实现 Differentiable Binarization (DB) 的检测输出头，与配置中 Head.name: DBHead 对应。

- Head：单分支子网络，经卷积/反卷积将融合特征映射为 1 通道概率图（sigmoid）。
- DBHead：包含 binarize 分支与 thresh 分支；训练时输出 shrink/threshold/binary 拼接图供 DBLoss；
  推理时仅返回 shrink_maps 作为 "maps"，再由 DBPostProcess 转为文字框。
- PFHeadLocal / LocalModule：DB 的变体/加强结构，用于部分实验配置。
====================================
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer


def get_bias_attr(k):
    """为转置卷积等生成均匀分布初始化的 bias ParamAttr。"""
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class Head(nn.Layer):
    """
    DB 中单支预测头：Conv -> BN+ReLU -> 两次 Conv2DTranspose 上采样 -> 1 通道 sigmoid。
    输出可理解为文字区域的概率图（或训练用 shrink/threshold 相关图）。
    """

    def __init__(self, in_channels, kernel_list=[3, 2, 2], fix_nan=False, **kwargs):
        super(Head, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False,
        )
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu",
        )

        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4),
        )
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu",
        )
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4),
        )

        self.fix_nan = fix_nan

    def forward(self, x, return_f=False):
        # return_f=True 时返回中间特征 f，供 PFHeadLocal 等结构使用
        x = self.conv1(x)
        x = self.conv_bn1(x)
        if self.fix_nan and self.training:
            x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        if self.fix_nan and self.training:
            x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class DBHead(nn.Layer):
    """
    Differentiable Binarization (DB) 检测头 — 中文摘要
    论文：https://arxiv.org/abs/1911.08947
    - binarize：近似文字区域概率图 P
    - thresh：阈值图 T
    - 训练时 binary = sigmoid(k*(P-T)) 与 P、T 拼接供损失监督
    - 推理时只返回 {"maps": shrink_maps}，后处理用 maps 生成多边形/四边形框
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)

    def step_function(self, x, y):
        """可微近似阶跃：将 P 与 T 组合成二值化图，k 越大越陡。"""
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {"maps": shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        return {"maps": y}


class LocalModule(nn.Layer):
    """局部细化模块：拼接初始图与特征再卷积，用于 PFHeadLocal。"""

    def __init__(self, in_c, mid_c, use_distance=True):
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act="relu")
        self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        outf = paddle.concat([init_map, x], axis=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out


class PFHeadLocal(DBHead):
    """DBHead 扩展：在 binarize 中间特征上增加局部 cbn 分支，融合基础图与细化图。"""

    def __init__(self, in_channels, k=50, mode="small", **kwargs):
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        self.mode = mode

        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest", align_mode=1)
        if self.mode == "large":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif self.mode == "small":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)

    def forward(self, x, targets=None):
        shrink_maps, f = self.binarize(x, return_f=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None)
        cbn_maps = F.sigmoid(cbn_maps)
        if not self.training:
            return {"maps": 0.5 * (base_maps + cbn_maps)}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([cbn_maps, threshold_maps, binary_maps], axis=1)
        return {"maps": y, "distance_maps": cbn_maps, "cbn_maps": binary_maps}
