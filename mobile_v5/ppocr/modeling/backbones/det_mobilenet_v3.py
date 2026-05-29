import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppocr.modeling.backbones.rec_hgnet import MeanPool2D

__all__ = ["MobileNetV3"]


def make_divisible(v, divisor=8, min_value=None):
    """
    将通道数调整为 divisor 的整数倍（通常是8的倍数）
    👉 目的：更适合硬件（GPU/NPU）并行计算，提高效率
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 防止向下取整太多（比如从32变成16）
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Layer):
    def __init__(
        self, 
        in_channels=3,         # 输入通道（RGB=3）
        model_name="large",    # large / small 两种结构
        scale=0.5,             # 通道缩放倍率（控制模型大小）
        disable_se=False,      # 是否关闭 SE 模块
        **kwargs
    ):
        """
        MobileNetV3 backbone（用于检测）

        输出：
        👉 多尺度特征列表 [C2, C3, C4, C5]
        👉 供后续 FPN 使用
        """
        super(MobileNetV3, self).__init__()

        self.disable_se = disable_se

        # =========================
        # 1️⃣ 定义网络结构配置 cfg
        # 每一行：k, exp, c, se, 激活函数, stride
        # =========================
        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],  # ↓ 下采样
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True,  "relu", 2],  # ↓ 下采样
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hardswish", 2],  # ↓ 下采样
                [3, 200, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 480, 112, True, "hardswish", 1],
                [3, 672, 112, True, "hardswish", 1],
                [5, 672, 160, True, "hardswish", 2],  # ↓ 下采样
                [5, 960, 160, True, "hardswish", 1],
                [5, 960, 160, True, "hardswish", 1],
            ]
            cls_ch_squeeze = 960  # 最后一层通道数
        else:
            raise NotImplementedError

        # 支持的 scale（控制模型大小）
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale

        # =========================
        # 2️⃣ 第一层卷积（输入层）
        # =========================
        inplanes = 16

        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,     # ↓ H,W / 2
            padding=1,
            groups=1,
            if_act=True,
            act="hardswish",
        )

        # =========================
        # 3️⃣ 构建多个 stage（核心）
        # =========================
        self.stages = []        # 每个 stage（给 FPN 用）
        self.out_channels = []  # 每个 stage 的通道数

        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)

        for k, exp, c, se, nl, s in cfg:

            # 是否使用 SE
            se = se and not self.disable_se

            # large 模型从第2层开始切 stage
            start_idx = 2

            # =========================
            # 当 stride=2 → 新 stage
            # =========================
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []

            # 添加一个倒残差块
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),  # 扩展通道
                    out_channels=make_divisible(scale * c),    # 输出通道
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                )
            )

            inplanes = make_divisible(scale * c)
            i += 1

        # =========================
        # 最后一层 1x1 卷积（增强语义）
        # =========================
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                if_act=True,
                act="hardswish",
            )
        )

        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

        # 注册子模块
        for i, stage in enumerate(self.stages):
            self.add_sublayer(sublayer=stage, name="stage{}".format(i))

    def forward(self, x):
        """
        前向传播

        输入：
        x → 原图

        输出：
        out_list → 多尺度特征
        """
        x = self.conv(x)  # 初始下采样

        out_list = []

        for stage in self.stages:
            x = stage(x)
            out_list.append(x)  # 保存每一层特征

        return out_list


# =========================
# 基础模块：Conv + BN + 激活
# =========================
class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        if_act=True,
        act=None,
    ):
        super(ConvBNLayer, self).__init__()

        self.if_act = if_act
        self.act = act

        # 卷积
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,  # groups>1 → depthwise
            bias_attr=False,
        )

        # BN
        self.bn = nn.BatchNorm(num_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        # 激活函数
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)

        return x


# =========================
# 倒残差块（核心）
# =========================
class ResidualUnit(nn.Layer):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
        use_se,
        act=None,
    ):
        super(ResidualUnit, self).__init__()

        # 是否使用残差连接
        self.if_shortcut = stride == 1 and in_channels == out_channels

        self.if_se = use_se

        # 1️⃣ 扩展通道（1x1）
        self.expand_conv = ConvBNLayer(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
        )

        # 2️⃣ 深度卷积（空间特征）
        self.bottleneck_conv = ConvBNLayer(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=mid_channels,  # depthwise
            if_act=True,
            act=act,
        )

        # 3️⃣ SE 注意力（可选）
        if self.if_se:
            self.mid_se = SEModule(mid_channels)

        # 4️⃣ 投影回低维（1x1）
        self.linear_conv = ConvBNLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
        )

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)

        if self.if_se:
            x = self.mid_se(x)

        x = self.linear_conv(x)

        # 残差连接
        if self.if_shortcut:
            x = paddle.add(inputs, x)

        return x


# =========================
# SE 注意力模块
# =========================
class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        # 两层 1x1 conv
        self.conv1 = nn.Conv2D(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2D(in_channels // reduction, in_channels, 1)

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        # 生成权重（0~1）
        x = F.hardsigmoid(x)

        # 通道加权
        return inputs * x