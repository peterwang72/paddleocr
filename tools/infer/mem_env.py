# Copyright (c) 2025 PaddleOCR Authors. All Rights Reserved.
"""推理前设置显存相关环境变量，须在 import paddle / torch 之前调用。"""

import os


def setup_infer_memory_env():
    """
    - PYTORCH_CUDA_ALLOC_CONF: 仅对 PyTorch 有效，减轻 CUDA 碎片 OOM。
    - FLAGS_allocator_strategy: Paddle 动态图按需申请显存（与 predict_det 一致）。

    Paddle Inference Predictor 主要仍依赖 enable_memory_optim()；
    本函数不能替代「使用空闲 GPU、batch=1」等做法。
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
