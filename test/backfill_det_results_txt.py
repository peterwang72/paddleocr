#!/usr/bin/env python3
"""
为仅有 det_res_*.jpg、没有 det_results.txt 的自训推理目录补写坐标文件。
overlay 对比依赖该文件；profile_det_folder 旧版未写入时可运行本脚本。

示例:
  cd PaddleOCR-main
  CUDA_VISIBLE_DEVICES=7 python test/backfill_det_results_txt.py \\
    --image_dir test/test_ocr \\
    --vis_dir output/output_1/PP-OCRv5_mobile_det_ocr_det_dataset_bg/test_infer_vis \\
    --det_model_dir output/output_1/PP-OCRv5_mobile_det_ocr_det_dataset_bg/best_model/inference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from ppocr.utils.logging import get_logger
from tools.infer.predict_det import TextDetector
from tools.infer.profile_det_folder import resolve_det_model_dir
from tools.infer import utility

from compare_det_models_vis import IMG_EXTS, index_vis_images, vis_key_from_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--vis_dir", required=True, help="含 det_res_*.jpg 的目录，将写入 det_results.txt")
    p.add_argument("--det_model_dir", required=True)
    p.add_argument("--merge", action="store_true", help="与已有 det_results.txt 合并（默认覆盖重写）")
    p.add_argument("--use_gpu", action="store_true", default=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--det_limit_side_len", type=int, default=960)
    return p.parse_args()


def build_det_args(cli, det_model_dir: str, vis_dir: str):
    argv_bak = sys.argv
    try:
        sys.argv = [argv_bak[0]]
        det_args = utility.parse_args()
    finally:
        sys.argv = argv_bak

    det_args.det_model_dir = det_model_dir
    det_args.det_algorithm = "DB"
    det_args.det_limit_side_len = cli.det_limit_side_len
    det_args.det_limit_type = "max"
    det_args.det_box_type = "quad"
    det_args.det_db_thresh = 0.3
    det_args.det_db_box_thresh = 0.6
    det_args.det_db_unclip_ratio = 1.5
    det_args.use_gpu = cli.use_gpu
    det_args.gpu_id = cli.gpu_id
    det_args.draw_img_save_dir = vis_dir
    det_args.save_crop_res = False
    det_args.benchmark = False
    return det_args


def repo_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


def find_image(image_root: Path, key: str) -> str | None:
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem.lower() == key:
            return str(p)
    return None


def main():
    args = parse_args()
    image_root = repo_path(args.image_dir)
    vis_dir = repo_path(args.vis_dir)
    det_dir = resolve_det_model_dir(str(repo_path(args.det_model_dir)), get_logger())

    keys = sorted(index_vis_images(vis_dir).keys())
    if not keys:
        raise SystemExit(f"无 det_res 图: {vis_dir}")

    existing = {}
    out_txt = vis_dir / "det_results.txt"
    if args.merge and out_txt.is_file():
        for line in out_txt.read_text(encoding="utf-8").splitlines():
            if "\t" not in line.strip():
                continue
            name, payload = line.split("\t", 1)
            k = vis_key_from_name(Path(name).name)
            try:
                existing[k] = payload.strip()
            except Exception:
                pass

    logger = get_logger()
    det_args = build_det_args(args, det_dir, str(vis_dir))
    detector = TextDetector(det_args, logger)

    lines = []
    for i, key in enumerate(keys):
        img_path = find_image(image_root, key)
        if not img_path:
            logger.warning("skip (no source image): {}".format(key))
            continue
        if args.merge and key in existing and existing[key] not in ("[]", ""):
            lines.append("{}\t{}\n".format(os.path.basename(img_path), existing[key]))
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        dt_boxes, _ = detector(img)
        n_box = 0 if dt_boxes is None else len(dt_boxes)
        box_json = json.dumps([x.tolist() for x in dt_boxes]) if n_box else "[]"
        lines.append("{}\t{}\n".format(os.path.basename(img_path), box_json))
        if (i + 1) % 50 == 0:
            logger.info("processed {}/{}".format(i + 1, len(keys)))

    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Wrote {} lines -> {}".format(len(lines), out_txt))


if __name__ == "__main__":
    main()
