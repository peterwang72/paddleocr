#!/usr/bin/env python3
# Profile exported DB det model on a folder (batch=1 per image).
# Usage (PaddleOCR-main root, e.g. GPU6 only):
#   CUDA_VISIBLE_DEVICES=6 python tools/infer/profile_det_folder.py \
#     --image_dir test/test_ocr \
#     --det_model_dir output/PP-OCRv5_mobile_det_ocr_det_dataset_bg/best_model/inference \
#     --profile_csv output/PP-OCRv5_mobile_det_ocr_det_dataset_bg/test_infer_profile.csv

import csv
import json
import os
import subprocess
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from tools.infer.mem_env import setup_infer_memory_env

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
setup_infer_memory_env()

import cv2
import numpy as np

from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read
from tools.infer import utility
from tools.infer.predict_det import TextDetector
from tools.infer.utility import str2bool

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def list_images_recursive(image_dir: str) -> list[str]:
    """递归收集图片；get_image_file_list 只扫目录顶层一层。"""
    if not os.path.isdir(image_dir):
        return []
    files = []
    for root, _, names in os.walk(image_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, name))
    return sorted(files)


def _has_pdiparams(model_dir):
    for name in ("inference", "model"):
        if os.path.isfile(os.path.join(model_dir, f"{name}.pdiparams")):
            return True
    return False


def resolve_det_model_dir(det_model_dir, logger):
    """Find directory that contains inference.pdiparams (Paddle export layout varies)."""
    candidates = [
        det_model_dir,
        os.path.join(det_model_dir, "inference"),
        os.path.dirname(det_model_dir.rstrip("/")),
        os.path.join(os.path.dirname(det_model_dir.rstrip("/")), "inference"),
    ]
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if _has_pdiparams(path):
            if path != os.path.abspath(det_model_dir):
                logger.info("Resolved det_model_dir: {} -> {}".format(det_model_dir, path))
            return path
    raise FileNotFoundError(
        "Cannot find inference.pdiparams under {} (tried parent/inference subdirs). "
        "Run: find {} -name '*.pdiparams'".format(
            det_model_dir, os.path.dirname(os.path.abspath(det_model_dir))
        )
    )


def gpu_mem_used_mb(physical_gpu_id):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={physical_gpu_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return -1


def main():
    parser = utility.init_args()
    parser.add_argument("--profile_csv", type=str, required=True)
    parser.add_argument(
        "--physical_gpu_id",
        type=int,
        default=None,
        help="nvidia-smi GPU index (default: first id in CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument("--save_vis", type=str2bool, default=True)
    args = parser.parse_args()

    if args.physical_gpu_id is None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        args.physical_gpu_id = int(vis) if vis else 0

    profile_dir = os.path.dirname(os.path.abspath(args.profile_csv))
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(args.draw_img_save_dir, exist_ok=True)

    logger = get_logger()
    image_file_list = list_images_recursive(args.image_dir)
    if not image_file_list:
        logger.error(
            "No images under {} (searched recursively). "
            "Try e.g. --image_dir test/test_ocr or test/test_ocr/data".format(
                args.image_dir
            )
        )
        sys.exit(1)

    args.det_model_dir = resolve_det_model_dir(args.det_model_dir, logger)
    logger.info(
        "Profile det infer | images={} | batch=1 | physical_gpu={} | model={}".format(
            len(image_file_list), args.physical_gpu_id, args.det_model_dir
        )
    )

    mem_before = gpu_mem_used_mb(args.physical_gpu_id)
    text_detector = TextDetector(args, logger)
    mem_after_load = gpu_mem_used_mb(args.physical_gpu_id)
    logger.info(
        "GPU{} mem: before_load={}MB after_load={}MB (delta={}MB)".format(
            args.physical_gpu_id,
            mem_before,
            mem_after_load,
            max(0, mem_after_load - mem_before) if mem_before >= 0 and mem_after_load >= 0 else -1,
        )
    )

    warmup_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(args.warmup_iters):
        text_detector(warmup_img)

    rows = []
    times = []
    peak_mem = mem_before
    det_result_lines = []

    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if flag_pdf or flag_gif:
            logger.warning("Skip unsupported: {}".format(image_file))
            continue
        if img is None:
            img = cv2.imread(image_file)
        if img is None:
            logger.warning("Failed to load: {}".format(image_file))
            continue

        mem_pre = gpu_mem_used_mb(args.physical_gpu_id)
        t0 = time.time()
        dt_boxes, elapse = text_detector(img)
        wall_s = time.time() - t0
        mem_post = gpu_mem_used_mb(args.physical_gpu_id)
        peak_mem = max(peak_mem, mem_pre, mem_post)

        times.append(wall_s)
        n_box = 0 if dt_boxes is None else len(dt_boxes)
        box_json = json.dumps([x.tolist() for x in dt_boxes]) if n_box else "[]"
        det_result_lines.append(
            "{}\t{}\n".format(os.path.basename(image_file), box_json)
        )
        rows.append(
            {
                "index": idx,
                "image": os.path.basename(image_file),
                "batch_size": 1,
                "num_boxes": len(dt_boxes) if dt_boxes is not None else 0,
                "time_s": f"{wall_s:.6f}",
                "detector_elapse_s": f"{elapse:.6f}",
                "gpu_mem_used_mb": mem_post,
            }
        )
        logger.info(
            "[{}/{}] {} boxes={} time={:.3f}s mem={}MB".format(
                idx + 1,
                len(image_file_list),
                os.path.basename(image_file),
                len(dt_boxes) if dt_boxes is not None else 0,
                wall_s,
                mem_post,
            )
        )

        if args.save_vis and dt_boxes is not None:
            vis = utility.draw_text_det_res(dt_boxes, img)
            out_path = os.path.join(
                args.draw_img_save_dir,
                "det_res_{}".format(os.path.basename(image_file)),
            )
            cv2.imwrite(out_path, vis)

    if args.save_vis and det_result_lines:
        results_path = os.path.join(args.draw_img_save_dir, "det_results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.writelines(det_result_lines)
        logger.info("Wrote det_results.txt ({} lines) -> {}".format(
            len(det_result_lines), results_path
        ))

    if not times:
        logger.error("No image processed.")
        sys.exit(1)

    mem_after_infer = gpu_mem_used_mb(args.physical_gpu_id)
    peak_mem = max(peak_mem, mem_after_infer)
    summary = {
        "batch_size": 1,
        "num_images": len(times),
        "total_time_s": sum(times),
        "avg_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "mem_load_delta_mb": max(0, mem_after_load - mem_before)
        if mem_before >= 0 and mem_after_load >= 0
        else -1,
        "peak_gpu_mem_used_mb": peak_mem,
        "peak_above_load_mb": max(0, peak_mem - mem_after_load)
        if mem_after_load >= 0
        else -1,
        "physical_gpu_id": args.physical_gpu_id,
    }

    fieldnames = list(rows[0].keys())
    with open(args.profile_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        f.write("\n")
        w.writerow({k: "" for k in fieldnames})
        w.writerow(
            {
                "index": "SUMMARY",
                "image": "",
                "batch_size": 1,
                "num_boxes": "",
                "time_s": f"{summary['avg_time_s']:.6f}",
                "detector_elapse_s": f"total={summary['total_time_s']:.3f}",
                "gpu_mem_used_mb": summary["peak_gpu_mem_used_mb"],
            }
        )

    summary_path = args.profile_csv.replace(".csv", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    logger.info("Profile CSV: {}".format(args.profile_csv))
    logger.info("Summary: {}".format(summary_path))
    logger.info(
        "batch=1 | images={} | avg={:.3f}s | total={:.1f}s | peak_gpu_mem={}MB".format(
            summary["num_images"],
            summary["avg_time_s"],
            summary["total_time_s"],
            summary["peak_gpu_mem_used_mb"],
        )
    )


if __name__ == "__main__":
    main()
