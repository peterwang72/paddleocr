#!/usr/bin/env python3
"""
可视化 DB 检测 probability map / 二值图，用于区分：
  - 漏检行有亮区 → 后处理(box_thresh/thresh)滤掉
  - 漏检行全暗   → 模型没响应（训练/标注/域问题）

单张:
  CUDA_VISIBLE_DEVICES=7 python test/vis_db_prob_map.py \\
    --image_path test/test_ocr/test/bg_00492.jpg \\
    --det_model_dir output/.../best_model/inference \\
    --output_dir output/debug_prob_map/bg_00492

文件夹（递归）:
  CUDA_VISIBLE_DEVICES=7 python test/vis_db_prob_map.py \\
    --image_dir test/test_ocr/test \\
    --det_model_dir output/.../best_model/inference \\
    --output_dir output/debug_prob_map/test
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.infer.mem_env import setup_infer_memory_env

setup_infer_memory_env()

import cv2
import numpy as np

from ppocr.data import transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read
from tools.infer import utility
from tools.infer.predict_det import TextDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def parse_args():
    p = argparse.ArgumentParser(description="Visualize DB probability map")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image_path", help="单张图片路径")
    g.add_argument("--image_dir", help="图片目录（递归搜索）")
    p.add_argument(
        "--det_model_dir",
        required=True,
        help="含 inference.pdiparams 的目录",
    )
    p.add_argument("--output_dir", required=True, help="可视化输出根目录")
    p.add_argument("--use_gpu", action="store_true", default=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--det_limit_side_len", type=int, default=1280)
    p.add_argument("--det_db_thresh", type=float, default=0.3)
    p.add_argument("--det_db_box_thresh", type=float, default=0.6)
    p.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    p.add_argument(
        "--compare_box_thresh",
        type=float,
        default=0.3,
        help="额外用更宽松的 box_thresh 出框对比",
    )
    return p.parse_args()


def repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def list_images_recursive(image_dir: str) -> list[str]:
    files = []
    for root, _, names in os.walk(image_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, name))
    return sorted(files)


def build_args(cli, output_dir: str):
    argv_bak = sys.argv
    try:
        sys.argv = [argv_bak[0]]
        args = utility.parse_args()
    finally:
        sys.argv = argv_bak
    args.det_model_dir = repo_path(cli.det_model_dir)
    args.det_algorithm = "DB"
    args.det_limit_side_len = cli.det_limit_side_len
    args.det_limit_type = "max"
    args.det_box_type = "quad"
    args.det_db_thresh = cli.det_db_thresh
    args.det_db_box_thresh = cli.det_db_box_thresh
    args.det_db_unclip_ratio = cli.det_db_unclip_ratio
    args.use_gpu = cli.use_gpu
    args.gpu_id = cli.gpu_id
    args.draw_img_save_dir = output_dir
    return args


def load_image(image_path: str):
    img, flag_gif, flag_pdf = check_and_read(image_path)
    if flag_gif or flag_pdf or img is None:
        img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    return img


def run_forward(detector: TextDetector, img: np.ndarray):
    data = {"image": img.copy()}
    data = transform(data, detector.preprocess_op)
    img_in, shape_list = data
    img_in = np.expand_dims(img_in, axis=0)
    shape_list = np.expand_dims(shape_list, axis=0)

    detector.input_tensor.copy_from_cpu(img_in)
    detector.predictor.run()
    outputs = [t.copy_to_cpu() for t in detector.output_tensors]
    maps = outputs[0]
    prob = maps[0, 0]
    return prob, shape_list, maps


def prob_to_original(prob: np.ndarray, shape_list: np.ndarray) -> np.ndarray:
    src_h, src_w, _, _ = shape_list[0]
    src_h, src_w = int(src_h), int(src_w)
    return cv2.resize(prob, (src_w, src_h), interpolation=cv2.INTER_LINEAR)


def save_prob_vis(prob_orig: np.ndarray, bgr: np.ndarray, out_dir: str, thresh: float):
    os.makedirs(out_dir, exist_ok=True)

    prob_u8 = np.clip(prob_orig * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "prob_gray.jpg"), prob_u8)

    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(out_dir, "prob_heatmap.jpg"), heat)

    overlay = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)
    cv2.imwrite(os.path.join(out_dir, "prob_overlay.jpg"), overlay)

    binary = (prob_orig > thresh).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(out_dir, f"binary_thresh_{thresh:.2f}.jpg"), binary)

    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_overlay = cv2.addWeighted(bgr, 0.55, binary_color, 0.45, 0)
    cv2.imwrite(os.path.join(out_dir, f"binary_overlay_{thresh:.2f}.jpg"), binary_overlay)


def postprocess_boxes(maps, shape_list, thresh, box_thresh, unclip_ratio):
    postprocess = build_post_process(
        {
            "name": "DBPostProcess",
            "thresh": thresh,
            "box_thresh": box_thresh,
            "max_candidates": 2000,
            "unclip_ratio": unclip_ratio,
            "score_mode": "fast",
            "box_type": "quad",
        }
    )
    return postprocess({"maps": maps}, shape_list)[0]["points"]


def process_one(detector, cli, image_path: str, out_dir: str, logger):
    img = load_image(image_path)
    prob, shape_list, maps = run_forward(detector, img)
    prob_orig = prob_to_original(prob, shape_list)

    save_prob_vis(prob_orig, img, out_dir, cli.det_db_thresh)

    boxes = postprocess_boxes(
        maps,
        shape_list,
        cli.det_db_thresh,
        cli.det_db_box_thresh,
        cli.det_db_unclip_ratio,
    )
    vis = utility.draw_text_det_res(boxes, img)
    cv2.imwrite(
        os.path.join(out_dir, f"boxes_boxthresh_{cli.det_db_box_thresh:.2f}.jpg"),
        vis,
    )

    boxes_loose = postprocess_boxes(
        maps,
        shape_list,
        cli.det_db_thresh,
        cli.compare_box_thresh,
        cli.det_db_unclip_ratio,
    )
    vis_loose = utility.draw_text_det_res(boxes_loose, img)
    cv2.imwrite(
        os.path.join(out_dir, f"boxes_boxthresh_{cli.compare_box_thresh:.2f}.jpg"),
        vis_loose,
    )

    logger.info(
        "[{}] prob min={:.4f} max={:.4f} mean={:.4f} | boxes {}@{}  {}@{}".format(
            os.path.basename(image_path),
            float(prob_orig.min()),
            float(prob_orig.max()),
            float(prob_orig.mean()),
            cli.det_db_box_thresh,
            len(boxes),
            cli.compare_box_thresh,
            len(boxes_loose),
        )
    )


def output_subdir(output_root: str, image_path: str, image_root: str | None) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if image_root:
        rel = os.path.relpath(os.path.dirname(image_path), image_root)
        if rel == ".":
            return os.path.join(output_root, stem)
        return os.path.join(output_root, rel, stem)
    return os.path.join(output_root, stem)


def main():
    cli = parse_args()
    logger = get_logger()
    output_root = repo_path(cli.output_dir)
    os.makedirs(output_root, exist_ok=True)

    if cli.image_path:
        image_list = [repo_path(cli.image_path)]
        image_root = None
    else:
        image_root = repo_path(cli.image_dir)
        image_list = list_images_recursive(image_root)
        if not image_list:
            raise FileNotFoundError("目录中无图片: {}".format(image_root))

    args = build_args(cli, output_root)
    detector = TextDetector(args, logger)

    logger.info("images={} model={}".format(len(image_list), args.det_model_dir))
    for idx, image_path in enumerate(image_list, start=1):
        out_dir = output_subdir(output_root, image_path, image_root)
        logger.info("[{}/{}] {} -> {}".format(idx, len(image_list), image_path, out_dir))
        process_one(detector, cli, image_path, out_dir, logger)

    logger.info("done. output_root={}".format(output_root))


if __name__ == "__main__":
    main()
