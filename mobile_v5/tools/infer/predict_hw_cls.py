# Copyright (c) 2025 PaddleOCR Authors. All Rights Reserved.
#
# 文本框三分类推理（PP-LCNetV2 + ClsHead 导出模型），与方向分类独立：
# 不做 180° 旋转，不校验官方 angle_cls 的 model_name。
# 预处理与 tools/infer/predict_cls.TextClassifier 一致（与训练 ClsResizeImg 对齐）。
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
repo_root_dir = os.path.abspath(os.path.join(__dir__, "../../.."))
if repo_root_dir not in sys.path:
    sys.path.insert(0, repo_root_dir)

from tools.infer.mem_env import setup_infer_memory_env

setup_infer_memory_env()

import copy
import importlib.util
import math
import time
import traceback

import cv2
import numpy as np

# 必须加载本目录 utility（含 hw3_cls_* 参数）；勿用仓库根 tools/infer/utility.py
_utility_path = os.path.join(__dir__, "utility.py")
_spec = importlib.util.spec_from_file_location("mobile_v5_infer_utility", _utility_path)
utility = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utility)
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read

logger = get_logger()


def _has_pdiparams(model_dir):
    for name in ("inference", "model"):
        if os.path.isfile(os.path.join(model_dir, f"{name}.pdiparams")):
            return True
    return False


def resolve_hw3_cls_model_dir(model_dir):
    """Paddle 导出目录可能是 .../inference 或 .../inference/inference。"""
    candidates = [
        model_dir,
        os.path.join(model_dir, "inference"),
        os.path.dirname(model_dir.rstrip("/")),
        os.path.join(os.path.dirname(model_dir.rstrip("/")), "inference"),
    ]
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if _has_pdiparams(path):
            if path != os.path.abspath(model_dir):
                logger.info("Resolved hw3_cls_model_dir: {} -> {}".format(model_dir, path))
            return path
    raise FileNotFoundError(
        "Cannot find inference.pdiparams under {}. Run: find {} -name '*.pdiparams'".format(
            model_dir, os.path.dirname(os.path.abspath(model_dir))
        )
    )


def _parse_hw3_label_list(args):
    raw = getattr(args, "hw3_cls_label_list", None)
    if raw is None:
        return ["printed", "handwriting_text", "handwriting_symbol"]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    s = str(raw).strip()
    if not s:
        return ["printed", "handwriting_text", "handwriting_symbol"]
    return [x.strip() for x in s.split(",") if x.strip()]


class TextBoxHW3Classifier(object):
    """对检测裁切条带做三分类；返回与 img_list 顺序一致的 (label, score) 列表。"""

    def __init__(self, args, logger=None):
        if logger is None:
            logger = get_logger()
        self.hw3_cls_image_shape = [
            int(v) for v in args.hw3_cls_image_shape.replace(" ", "").split(",")
        ]
        self.hw3_cls_batch_num = args.hw3_cls_batch_num
        label_list = _parse_hw3_label_list(args)
        postprocess_params = {"name": "ClsPostProcess", "label_list": label_list}
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            _,
        ) = utility.create_predictor(args, "hw3_cls", logger)
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.hw3_cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        if self.hw3_cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype="float32")
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))

        cls_res = [["", 0.0]] * img_num
        batch_num = self.hw3_cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            starttime = time.time()
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            if self.use_onnx:
                input_dict = {self.input_tensor.name: norm_img_batch}
                outputs = self.predictor.run(self.output_tensors, input_dict)
                prob_out = outputs[0]
            else:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.run()
                prob_out = self.output_tensors[0].copy_to_cpu()
                self.predictor.try_shrink_memory()
            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
        return cls_res, elapse


def main(args):
    if not args.hw3_cls_model_dir:
        raise SystemExit("请设置 --hw3_cls_model_dir 为导出的推理模型目录")
    args.hw3_cls_model_dir = resolve_hw3_cls_model_dir(args.hw3_cls_model_dir)
    image_file_list = get_image_file_list(args.image_dir)
    clf = TextBoxHW3Classifier(args)
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        try:
            cls_res, predict_time = clf([img])
        except Exception as E:
            logger.info(traceback.format_exc())
            logger.info(E)
            exit()
        logger.info("Predicts of {}:{}".format(image_file, cls_res[0]))


if __name__ == "__main__":
    main(utility.parse_args())
