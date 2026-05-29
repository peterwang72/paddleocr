# Copyright (c) 2025 PaddleOCR Authors. All Rights Reserved.
#
# PP-OCRv5 Mobile 检测 + PP-LCNetV2 三分类联调示例（无识别）。
# 在仓库根目录执行:
#   python mobile_v5/tools/infer/run_det_hw3_demo.py ^
#     --image_dir ./doc/imgs/ ^
#     --det_model_dir <PP-OCRv5_mobile_det 推理目录> ^
#     --hw3_cls_model_dir <pp_lcnet_v2_hw3 导出推理目录>
import copy
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
mobile_v5_dir = os.path.abspath(os.path.join(__dir__, "../.."))
if mobile_v5_dir not in sys.path:
    sys.path.insert(0, mobile_v5_dir)
repo_root_dir = os.path.abspath(os.path.join(__dir__, "../../.."))
if repo_root_dir not in sys.path:
    sys.path.append(repo_root_dir)

from tools.infer.mem_env import setup_infer_memory_env

setup_infer_memory_env()

import cv2
import numpy as np

import tools.infer.utility as utility
from tools.infer.predict_det import TextDetector
from tools.infer.predict_hw_cls import TextBoxHW3Classifier
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read, get_image_file_list


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_b = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_b)
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def main():
    args = utility.parse_args()
    if not args.hw3_cls_model_dir:
        raise SystemExit("请设置 --hw3_cls_model_dir 为三分类导出推理模型目录")
    logger = get_logger()
    detector = TextDetector(args, logger)
    hw_clf = TextBoxHW3Classifier(args, logger)

    for image_file in get_image_file_list(args.image_dir):
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.warning("skip load fail: %s", image_file)
            continue
        ori = img.copy()
        dt_boxes, t_det = detector(img)
        if dt_boxes is None or (hasattr(dt_boxes, "size") and dt_boxes.size == 0):
            logger.info("%s: 0 boxes, det %.3fs", image_file, t_det)
            continue
        dt_boxes = np.array(sorted_boxes(dt_boxes))
        crops = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            crops.append(utility.get_rotate_crop_image(ori, tmp_box))
        cls_res, t_cls = hw_clf(crops)
        logger.info(
            "%s: det %.3fs hw3 %.3fs %s",
            image_file,
            t_det,
            t_cls,
            list(zip(cls_res, [b.tolist() for b in dt_boxes])),
        )


if __name__ == "__main__":
    main()
