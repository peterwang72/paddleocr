# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PaddleOCR 训练入口脚本。

典型用法（在仓库根目录）：
    python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml
    python -m paddle.distributed.launch --gpus 0,1 tools/train.py -c <配置.yml> -o Global.pretrained_model=...

流程概览：解析配置 → 构建数据与模型 → 损失/优化器/指标 → 可选 AMP 与 SyncBN →
加载预训练或断点 → 分布式包装 → 调用 tools.program.train 执行 epoch 循环。
配置项说明见文档与 configs 下各 yml 文件。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# 添加项目根目录到 Python 路径，以便导入 ppocr 模块
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import yaml
import paddle
import paddle.distributed as dist

# 导入 PaddleOCR 内部模块
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
import tools.program as program
import tools.naive_sync_bn as naive_sync_bn

# 尽早访问分布式环境（单卡时 world_size 为 1），与后续 DataParallel 初始化衔接
# 这行代码会触发分布式环境的初始化检查
dist.get_world_size()


def main(config, device, logger, vdl_writer, seed):
    """
    训练主函数：根据配置完成组网与训练准备，最终交给 program.train 循环迭代。
    
    Args:
        config (dict): 完整的配置字典，包含 Global, Architecture, Loss, Optimizer 等
        device (paddle.CUDAPlace or paddle.CPUPlace): 计算设备
        logger (logging.Logger): 日志记录器
        vdl_writer: VisualDL 日志写入器，用于可视化训练曲线
        seed (int): 随机种子，保证可复现性
    """
    
    # ==================== 1. 分布式环境初始化 ====================
    # 多卡时初始化进程组；是否分布式在 program.preprocess 里会按实际进程数覆盖 Global.distributed
    if config["Global"]["distributed"]:
        dist.init_parallel_env()  # 初始化并行环境，设置通信方式等

    global_config = config["Global"]

    # ==================== 2. 数据加载器构建 ====================
    # 设置信号处理器，用于优雅退出（如 Ctrl+C 时保存模型）
    set_signal_handlers()
    
    # 构建训练数据加载器
    train_dataloader = build_dataloader(config, "Train", device, logger, seed)
    
    # 检查训练集是否为空（常见问题：标签文件路径错误或图片数量不足）
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n"
            + "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            + "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    # 构建验证数据加载器（如果配置了 Eval 部分）
    if config["Eval"]:
        valid_dataloader = build_dataloader(config, "Eval", device, logger, seed)
    else:
        valid_dataloader = None
    
    # 每个 epoch 的迭代步数（用于学习率调度器）
    step_pre_epoch = len(train_dataloader)

    # ==================== 3. 后处理模块构建 ====================
    # 后处理模块负责将模型原始输出转换为可读结果（如检测框坐标、识别文字）
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # ==================== 4. 识别模型：动态调整输出通道数 ====================
    # 根据字符集大小自动调整模型最后一层的输出维度
    # 这是关键步骤：因为不同数据集的字符集大小不同（中文约6625，英文36）
    if hasattr(post_process_class, "character"):
        # 获取字符集大小
        char_num = len(getattr(post_process_class, "character"))
        
        # 4.1 蒸馏模型处理（Distillation）
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            for key in config["Architecture"]["Models"]:
                # 处理多头模型（MultiHead）
                if config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead":
                    # 根据不同解码器调整字符数偏移
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2  # SAR 需要排除 SOS/EOS
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3  # NRTR 需要特殊标记
                    
                    # 配置不同解码器的输出通道数
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
                    
                    # 更新 SARLoss 的参数（忽略索引）
                    if list(config["Loss"]["loss_config_list"][-1].keys())[0] == "DistillationSARLoss":
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = (char_num + 1)
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    # 更新 NRTRLoss 的参数
                    elif any("DistillationNRTRLoss" in d for d in config["Loss"]["loss_config_list"]):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3
                    
                    # 将输出通道配置写入模型配置
                    config["Architecture"]["Models"][key]["Head"]["out_channels_list"] = out_channels_list
                else:
                    # 普通蒸馏模型，直接设置输出通道数
                    config["Architecture"]["Models"][key]["Head"]["out_channels"] = char_num
        
        # 4.2 单模型多头处理
        elif config["Architecture"]["Head"]["name"] == "MultiHead":
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            
            # 更新 SARLoss 的 ignore_index
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {"ignore_index": char_num + 1}
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = char_num + 1
                out_channels_list["SARLabelDecode"] = char_num + 2
            # 更新 NRTRLoss 的配置
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        
        # 4.3 基础识别模型
        else:
            config["Architecture"]["Head"]["out_channels"] = char_num

        # 4.4 SAR 模型的特殊处理：设置损失函数忽略索引
        if config["PostProcess"]["name"] == "SARLabelDecode":
            config["Loss"]["ignore_index"] = char_num - 1

    # ==================== 5. 构建神经网络模型 ====================
    model = build_model(config["Architecture"])

    # ==================== 6. 同步批归一化处理（SyncBN） ====================
    # 多卡训练时，同步所有卡上的 BN 统计信息，提升训练稳定性
    use_sync_bn = config["Global"].get("use_sync_bn", False)
    if use_sync_bn:
        # NPU（华为昇腾）或 XPU（百度昆仑）使用自定义的同步 BN 实现
        if config["Global"].get("use_npu", False) or config["Global"].get("use_xpu", False):
            naive_sync_bn.convert_syncbn(model)
        else:
            # GPU 使用 Paddle 官方的 SyncBatchNorm
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("convert_sync_batchnorm")

    # ==================== 7. 动态图转静态图（可选） ====================
    # 转换为静态图可以提升推理速度，但训练时通常保持动态图
    model = apply_to_static(model, config, logger)

    # ==================== 8. 构建损失函数、优化器、评估指标 ====================
    # 损失函数（如 CTCLoss、SARLoss、检测损失等）
    loss_class = build_loss(config["Loss"])

    # 优化器（如 Adam、SGD）和学习率调度器（如余弦退火、阶梯下降）
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    # 评估指标（如准确率、召回率、Hmean）
    eval_class = build_metric(config["Metric"])

    # 打印数据加载器信息
    logger.info("train dataloader has {} iters".format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))

    # ==================== 9. 混合精度训练（AMP）配置 ====================
    # 混合精度可以加速训练并减少显存占用
    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")  # O1: 混合精度, O2: 纯半精度
    amp_dtype = config["Global"].get("amp_dtype", "float16")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])  # 强制用 FP32 的算子
    amp_custom_white_list = config["Global"].get("amp_custom_white_list", [])  # 强制用 FP16 的算子
    
    # 删除可能存在的旧训练结果文件
    if os.path.exists(os.path.join(config["Global"]["save_model_dir"], "train_result.json")):
        try:
            os.remove(os.path.join(config["Global"]["save_model_dir"], "train_result.json"))
        except:
            pass
    
    if use_amp:
        # 设置 AMP 相关的 Paddle 标志位（优化性能）
        AMP_RELATED_FLAGS_SETTING = {}
        if paddle.is_compiled_with_cuda():
            AMP_RELATED_FLAGS_SETTING.update({
                "FLAGS_cudnn_batchnorm_spatial_persistent": 1,  # 优化 BN 层
                "FLAGS_gemm_use_half_precision_compute_type": 0,  # GEMM 使用半精度
            })
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        
        # 梯度缩放器：防止 FP16 训练时梯度下溢
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get("use_dynamic_loss_scaling", False)
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        
        # O2 级别：模型参数和大部分计算都转为 FP16
        if amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                master_weight=True,  # 保留 FP32 的主权重用于更新
                dtype=amp_dtype,
            )
    else:
        scaler = None

    # ==================== 10. 加载预训练模型或断点 ====================
    # 优先级：Global.checkpoints > Global.pretrained_model
    # 返回之前的最佳模型指标（如 best_accuracy），用于续训恢复最佳状态
    pre_best_model_dict = load_model(
        config, model, optimizer, config["Architecture"]["model_type"]
    )

    # ==================== 11. 分布式包装 ====================
    # 多卡训练时用 DataParallel 自动处理梯度同步
    if config["Global"]["distributed"]:
        # find_unused_parameters：是否检测未使用的参数（帮助调试）
        find_unused_parameters = config["Global"].get("find_unused_parameters", False)
        model = paddle.DataParallel(
            model, find_unused_parameters=find_unused_parameters
        )

    # ==================== 12. 启动训练循环 ====================
    # program.train 包含实际的训练逻辑：epoch 循环、前向传播、反向传播、
    # 保存模型、输出日志、验证集评估等
    program.train(
        config,
        train_dataloader,
        valid_dataloader,
        device,
        model,
        loss_class,
        optimizer,
        lr_scheduler,
        post_process_class,
        eval_class,
        pre_best_model_dict,
        logger,
        step_pre_epoch,
        vdl_writer,
        scaler,
        amp_level,
        amp_custom_black_list,
        amp_custom_white_list,
        amp_dtype,
    )


def test_reader(config, device, logger):
    """
    调试数据管道：仅遍历训练集并打印 batch 耗时。
    用于检查数据加载是否正常，不进行实际训练。
    
    Args:
        config (dict): 配置字典
        device: 计算设备
        logger: 日志记录器
    """
    # 构建训练数据加载器
    loader = build_dataloader(config, "Train", device, logger, seed=1024)
    import time
    
    starttime = time.time()
    count = 0
    try:
        # 遍历所有数据
        for data in loader():
            count += 1
            if count % 1 == 0:  # 每个 batch 都打印
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == "__main__":
    """
    脚本入口点：
    1. 解析命令行参数（-c 配置文件, -o 覆盖配置）
    2. 合并 YAML 配置
    3. 设置设备（CPU/GPU/NPU/XPU）
    4. 初始化日志和 VisualDL
    5. 设置随机种子
    6. 调用 main() 开始训练
    """
    # 预处理：解析配置、设置设备、初始化日志等
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    
    # 获取随机种子（默认 1024）
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    
    # 设置随机种子，保证结果可复现
    set_seed(seed)
    
    # 启动训练
    main(config, device, logger, vdl_writer, seed)
    
    # 如需测试数据加载器，取消下面一行的注释
    # test_reader(config, device, logger)