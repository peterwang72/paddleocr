"""
IntraCL（Intra-stage Context Localization）模块 — 中文说明
================================================================
作用：在特征图上用多尺度条形卷积（横向/纵向）与方形卷积提取上下文，
      经 1x1 降维/升维与残差连接，增强 Neck（如 RSEFPN）融合特征时的局部结构表达。
来源参考：ViTAE / I3CL 中的 intra_cl 思想（见原仓库注释链接）。
本文件被 db_fpn.py 中 RSEFPN 在 intracl=True 时选用。
================================================================
"""
import paddle
from paddle import nn

# refer from: https://github.com/ViTAE-Transformer/I3CL/blob/736c80237f66d352d488e83b05f3e33c55201317/mmdet/models/detectors/intra_cl_module.py


class IntraCLBlock(nn.Layer):
    """单块 IntraCL：多方向多尺度卷积 + BN + ReLU + 与输入残差。"""

    def __init__(self, in_channels=96, reduce_factor=4):
        super(IntraCLBlock, self).__init__()
        self.channels = in_channels
        self.rf = reduce_factor
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.conv1x1_reduce_channel = nn.Conv2D(
            self.channels, self.channels // self.rf, kernel_size=1, stride=1, padding=0
        )
        self.conv1x1_return_channel = nn.Conv2D(
            self.channels // self.rf, self.channels, kernel_size=1, stride=1, padding=0
        )

        self.v_layer_7x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(3, 0),
        )
        self.v_layer_5x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        self.v_layer_3x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
        )

        self.q_layer_1x7 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(0, 3),
        )
        self.q_layer_1x5 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 5),
            stride=(1, 1),
            padding=(0, 2),
        )
        self.q_layer_1x3 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )

        # base
        self.c_layer_7x7 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(3, 3),
        )
        self.c_layer_5x5 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )
        self.c_layer_3x3 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.bn = nn.BatchNorm2D(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 先压缩通道，减少条形/方形卷积计算量
        x_new = self.conv1x1_reduce_channel(x)

        # 三个尺度（7/5/3）：每个尺度上同时做方形卷积与水平条、垂直条，再相加
        x_7_c = self.c_layer_7x7(x_new)
        x_7_v = self.v_layer_7x1(x_new)
        x_7_q = self.q_layer_1x7(x_new)
        x_7 = x_7_c + x_7_v + x_7_q

        x_5_c = self.c_layer_5x5(x_7)
        x_5_v = self.v_layer_5x1(x_7)
        x_5_q = self.q_layer_1x5(x_7)
        x_5 = x_5_c + x_5_v + x_5_q

        x_3_c = self.c_layer_3x3(x_5)
        x_3_v = self.v_layer_3x1(x_5)
        x_3_q = self.q_layer_1x3(x_5)
        x_3 = x_3_c + x_3_v + x_3_q

        # 恢复通道数并归一化、激活
        x_relation = self.conv1x1_return_channel(x_3)

        x_relation = self.bn(x_relation)
        x_relation = self.relu(x_relation)

        # 残差：原特征 + 上下文增强分支
        return x + x_relation


def build_intraclblock_list(num_block):
    """堆叠多个 IntraCLBlock，返回 LayerList（供 Neck 按层使用）。"""
    IntraCLBlock_list = nn.LayerList()
    for i in range(num_block):
        IntraCLBlock_list.append(IntraCLBlock())

    return IntraCLBlock_list
