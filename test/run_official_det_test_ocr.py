#!/usr/bin/env python3
"""
在 PaddleOCR-main 内调用官方 PP-OCRv5_mobile 检测推理模型。

示例（在仓库根目录执行，GPU7）:
  cd /mnt/hdd/wb/shared/PaddleOCR-main
  CUDA_VISIBLE_DEVICES=7 python test/run_official_det_test_ocr.py \\
    --det_model_dir inference/PP-OCRv5_mobile_det_infer \\
    --image_dir test/test_ocr \\
    --output_dir test/test_ocr/det_vis_official
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.infer.mem_env import setup_infer_memory_env

setup_infer_memory_env()

import cv2

from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read
from tools.infer import utility
from tools.infer.predict_det import TextDetector


def _repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def list_images_recursive(image_dir: str) -> list[str]:
    """get_image_file_list 只扫一层；图片在 data/ 等子目录时需递归。"""
    files = []
    for root, _, names in os.walk(image_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, name))
    return sorted(files)


def find_pdiparams_dir(root: str) -> str | None:
    for dirpath, _, filenames in os.walk(root):
        if "inference.pdiparams" in filenames or "model.pdiparams" in filenames:
            return dirpath
    return None


def parse_args():
    p = argparse.ArgumentParser(description="PP-OCRv5_mobile 官方检测推理")
    p.add_argument(
        "--det_model_dir",
        type=str,
        default="inference/PP-OCRv5_mobile_det_infer",
        help="含 inference.pdiparams 的目录（相对 PaddleOCR-main 根目录）",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        default="test/test_ocr/data",
        help="待检测图片目录（相对 PaddleOCR-main 根目录，会递归子目录）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="test/test_ocr/det_vis_official",
        help="可视化与 det_results.txt 输出目录（相对 PaddleOCR-main 根目录）",
    )
    p.add_argument("--use_gpu", action="store_true", default=True)
    p.add_argument("--gpu_id", type=int, default=0, help="CUDA_VISIBLE_DEVICES 映射后的 id，一般为 0")
    p.add_argument("--det_limit_side_len", type=int, default=960)
    return p.parse_args()


def build_det_args(cli, det_model_dir: str, output_dir: str):
    """从 tools.infer.utility 默认参数构造完整 Namespace，避免缺字段。"""
    argv_bak = sys.argv
    try:
        sys.argv = [argv_bak[0]]
        args = utility.parse_args()
    finally:
        sys.argv = argv_bak

    args.det_model_dir = det_model_dir
    args.det_algorithm = "DB"
    args.det_limit_side_len = cli.det_limit_side_len
    args.det_limit_type = "max"
    args.det_box_type = "quad"
    args.det_db_thresh = 0.3
    args.det_db_box_thresh = 0.6
    args.det_db_unclip_ratio = 1.5
    args.use_gpu = cli.use_gpu
    args.gpu_id = cli.gpu_id
    args.draw_img_save_dir = output_dir
    return args


def main():
    cli = parse_args()
    logger = get_logger()

    det_dir = _repo_path(cli.det_model_dir)
    if not os.path.isfile(os.path.join(det_dir, "inference.pdiparams")) and not os.path.isfile(
        os.path.join(det_dir, "model.pdiparams")
    ):
        found = find_pdiparams_dir(det_dir)
        if found:
            logger.info("Resolved det_model_dir: {} -> {}".format(det_dir, found))
            det_dir = found
        else:
            raise FileNotFoundError(
                "未找到 inference.pdiparams，请检查 --det_model_dir: {}".format(cli.det_model_dir)
            )

    image_dir = _repo_path(cli.image_dir)
    output_dir = _repo_path(cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    images = list_images_recursive(image_dir)
    if not images:
        raise FileNotFoundError(
            "目录中无图片（已递归搜索）: {}，请改用 --image_dir test/test_ocr/data".format(
                image_dir
            )
        )

    logger.info("det_model_dir={}".format(det_dir))
    logger.info("images={} from {}".format(len(images), image_dir))
    logger.info("output_dir={}".format(output_dir))

    det_args = build_det_args(cli, det_dir, output_dir)
    detector = TextDetector(det_args, logger)

    results_path = os.path.join(output_dir, "det_results.txt")
    lines = []
    total_time = 0.0

    for idx, image_file in enumerate(images):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if flag_gif or flag_pdf:
            logger.warning("skip: {}".format(image_file))
            continue
        if img is None:
            img = cv2.imread(image_file)
        if img is None:
            logger.warning("load fail: {}".format(image_file))
            continue

        t0 = time.time()
        dt_boxes, elapse = detector(img)
        total_time += time.time() - t0

        n_box = 0 if dt_boxes is None else len(dt_boxes)
        box_json = json.dumps([x.tolist() for x in dt_boxes]) if n_box else "[]"
        lines.append("{}\t{}\n".format(os.path.basename(image_file), box_json))

        vis = utility.draw_text_det_res(dt_boxes, img)
        out_img = os.path.join(output_dir, "det_res_{}".format(os.path.basename(image_file)))
        cv2.imwrite(out_img, vis)

        logger.info(
            "[{}/{}] {} boxes={} time={:.3f}s".format(
                idx + 1, len(images), os.path.basename(image_file), n_box, elapse
            )
        )

    with open(results_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info("done. avg_time={:.3f}s results={}".format(total_time / max(len(lines), 1), results_path))


if __name__ == "__main__":
    main()
