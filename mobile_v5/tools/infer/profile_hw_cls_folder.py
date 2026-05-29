#!/usr/bin/env python3
# Profile exported PP-LCNetV2 hw3 cls model on a folder (batch=1 per image).
# Usage (PaddleOCR-main root):
#   CUDA_VISIBLE_DEVICES=3 python mobile_v5/tools/infer/profile_hw_cls_folder.py \
#     --image_dir test/test_cls/data \
#     --hw3_cls_model_dir output/cls/pp_lcnet_v2_hw3_ocr_det_dataset_all/best_model/inference \
#     --profile_csv output/cls/pp_lcnet_v2_hw3_ocr_det_dataset_all/test_cls_infer_profile.csv

import csv
import importlib.util
import os
import subprocess
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
repo_root_dir = os.path.abspath(os.path.join(__dir__, "../../.."))
if repo_root_dir not in sys.path:
    sys.path.insert(0, repo_root_dir)

from tools.infer.mem_env import setup_infer_memory_env

setup_infer_memory_env()

import cv2
import numpy as np

_utility_path = os.path.join(__dir__, "utility.py")
_spec_u = importlib.util.spec_from_file_location("mobile_v5_infer_utility", _utility_path)
utility = importlib.util.module_from_spec(_spec_u)
_spec_u.loader.exec_module(utility)

_predict_path = os.path.join(__dir__, "predict_hw_cls.py")
_spec_p = importlib.util.spec_from_file_location("predict_hw_cls", _predict_path)
_predict_mod = importlib.util.module_from_spec(_spec_p)
_spec_p.loader.exec_module(_predict_mod)

TextBoxHW3Classifier = _predict_mod.TextBoxHW3Classifier
resolve_hw3_cls_model_dir = _predict_mod.resolve_hw3_cls_model_dir

from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read, get_image_file_list


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
    parser.add_argument("--physical_gpu_id", type=int, default=None)
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument(
        "--save_results_txt",
        type=str,
        default=None,
        help="Optional: save per-image label,score lines",
    )
    args = parser.parse_args()

    if args.physical_gpu_id is None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        args.physical_gpu_id = int(vis) if vis else 0

    if not args.hw3_cls_model_dir:
        raise SystemExit("请设置 --hw3_cls_model_dir")

    profile_dir = os.path.dirname(os.path.abspath(args.profile_csv))
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)

    logger = get_logger()
    image_file_list = get_image_file_list(args.image_dir)
    if not image_file_list:
        logger.error("No images in {}".format(args.image_dir))
        sys.exit(1)

    args.hw3_cls_model_dir = resolve_hw3_cls_model_dir(args.hw3_cls_model_dir)
    args.hw3_cls_batch_num = 1

    logger.info(
        "Profile hw3 cls | images={} | batch=1 | physical_gpu={} | model={}".format(
            len(image_file_list), args.physical_gpu_id, args.hw3_cls_model_dir
        )
    )

    mem_before = gpu_mem_used_mb(args.physical_gpu_id)
    classifier = TextBoxHW3Classifier(args, logger)
    mem_after_load = gpu_mem_used_mb(args.physical_gpu_id)
    logger.info(
        "GPU{} mem: before_load={}MB after_load={}MB (delta={}MB)".format(
            args.physical_gpu_id,
            mem_before,
            mem_after_load,
            max(0, mem_after_load - mem_before)
            if mem_before >= 0 and mem_after_load >= 0
            else -1,
        )
    )

    warmup_img = np.random.randint(0, 255, (48, 192, 3), dtype=np.uint8)
    for _ in range(args.warmup_iters):
        classifier([warmup_img])

    rows = []
    times = []
    peak_mem = mem_before
    result_lines = []

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
        cls_res, elapse = classifier([img])
        wall_s = time.time() - t0
        mem_post = gpu_mem_used_mb(args.physical_gpu_id)
        peak_mem = max(peak_mem, mem_pre, mem_post)

        label, score = cls_res[0]
        times.append(wall_s)
        rows.append(
            {
                "index": idx,
                "image": os.path.basename(image_file),
                "batch_size": 1,
                "label": label,
                "score": f"{score:.6f}",
                "time_s": f"{wall_s:.6f}",
                "classifier_elapse_s": f"{elapse:.6f}",
                "gpu_mem_used_mb": mem_post,
            }
        )
        result_lines.append(
            "{}\t{}\t{:.6f}\t{:.6f}".format(
                os.path.basename(image_file), label, score, wall_s
            )
        )
        logger.info(
            "[{}/{}] {} {} {:.3f} time={:.3f}s mem={}MB".format(
                idx + 1,
                len(image_file_list),
                os.path.basename(image_file),
                label,
                score,
                wall_s,
                mem_post,
            )
        )

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
                "label": "",
                "score": "",
                "time_s": f"{summary['avg_time_s']:.6f}",
                "classifier_elapse_s": f"total={summary['total_time_s']:.3f}",
                "gpu_mem_used_mb": summary["peak_gpu_mem_used_mb"],
            }
        )

    summary_path = args.profile_csv.replace(".csv", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    if args.save_results_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_results_txt)) or ".", exist_ok=True)
        with open(args.save_results_txt, "w", encoding="utf-8") as f:
            f.write("image\tlabel\tscore\ttime_s\n")
            f.write("\n".join(result_lines) + "\n")

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
