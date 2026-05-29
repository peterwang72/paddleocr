#!/usr/bin/env python3
"""
对比官方 PP-OCRv5_mobile_det 与自训模型在同一张图上的检测可视化。

默认目录（相对 PaddleOCR-main 根目录）:
  官方: test/test_ocr/det_vis_official
  自训: test_infer_vis + data_infer_vis（同一批图分在两个目录，脚本会合并后与官方逐张对比）

示例:
  cd PaddleOCR-main
  python test/compare_det_models_vis.py \\
    --image_dir test/test_ocr \\
    --official_dir test/test_ocr/det_vis_official \\
    --custom_dirs output/output_1/PP-OCRv5_mobile_det_ocr_det_dataset_bg/test_infer_vis,output/output_1/PP-OCRv5_mobile_det_ocr_det_dataset_bg/data_infer_vis \\
    --output_dir test/test_ocr/det_compare_official_vs_custom
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="官方 vs 自训 检测框对比可视化")
    p.add_argument(
        "--image_dir",
        type=str,
        default="test/test_ocr",
        help="原图根目录（递归搜索，用于三列对比时的左图）",
    )
    p.add_argument(
        "--official_dir",
        type=str,
        default="test/test_ocr/det_vis_official",
    )
    p.add_argument(
        "--custom_dir",
        type=str,
        default="",
        help="单个自训可视化目录（与 --custom_dirs 二选一）",
    )
    p.add_argument(
        "--custom_dirs",
        type=str,
        default="output/output_1/PP-OCRv5_mobile_det_ocr_det_dataset_bg/test_infer_vis,output/output_1/PP-OCRv5_mobile_det_ocr_det_dataset_bg/data_infer_vis",
        help="逗号分隔的自训目录，会合并为一套结果（图分散在多个目录时）",
    )
    p.add_argument(
        "--custom_label",
        type=str,
        default="Custom (bg)",
        help="并排图中自训侧标题",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="test/test_ocr/det_compare_official_vs_custom",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["side_by_side", "overlay", "both"],
        default="both",
        help="side_by_side=并排原图+红框; overlay=原图+双色框; both=两种都输出",
    )
    p.add_argument(
        "--side_color",
        type=str,
        default="0,0,255",
        help="compare_side 左右两侧检测框颜色（BGR，默认红）",
    )
    p.add_argument("--official_color", type=str, default="0,255,0", help="overlay 官方框 BGR 颜色")
    p.add_argument("--custom_color", type=str, default="0,0,255", help="overlay 自训框 BGR 颜色")
    p.add_argument("--thickness", type=int, default=2)
    return p.parse_args()


def repo_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


def parse_bgr(s: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    return tuple(parts[:3])


def vis_key_from_name(name: str) -> str:
    """det_res_foo.jpg -> foo；用于跨目录匹配。"""
    stem = Path(name).stem
    if stem.startswith("det_res_"):
        stem = stem[len("det_res_") :]
    return stem.lower()


def index_vis_images(vis_dir: Path) -> dict[str, Path]:
    out = {}
    if not vis_dir.is_dir():
        return out
    for p in vis_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[vis_key_from_name(p.name)] = p
    return out


def merge_maps(
    maps: list[dict[str, object]], names: list[str]
) -> tuple[dict[str, object], list[str]]:
    """合并多个目录索引；同一 key 重复时保留先出现的并记录警告。"""
    merged: dict[str, object] = {}
    dup_warnings: list[str] = []
    for name, m in zip(names, maps):
        for k, v in m.items():
            if k in merged and merged[k] != v:
                dup_warnings.append(f"  {k}: 已在其他目录，忽略 {name} 中的重复")
            elif k not in merged:
                merged[k] = v
    return merged, dup_warnings


def box_count(boxes) -> int:
    return len(boxes) if boxes else 0


def merge_box_maps(
    maps: list[dict[str, list]], names: list[str]
) -> tuple[dict[str, list], list[str]]:
    """合并 det_results；同一 key 优先保留框更多的条目（避免先写入空 [] 挡住有效结果）。"""
    merged: dict[str, list] = {}
    dup_warnings: list[str] = []
    for name, m in zip(names, maps):
        for k, v in m.items():
            if k not in merged:
                merged[k] = v
            elif box_count(v) > box_count(merged[k]):
                dup_warnings.append(
                    f"  {k}: 用 {name} 的 {box_count(v)} 框覆盖先前的 {box_count(merged[k])} 框"
                )
                merged[k] = v
    return merged, dup_warnings


def load_det_results_txt(txt_path: Path) -> dict[str, list]:
    """det_results.txt: basename \\t json_boxes（键与 det_res_*.jpg 一致，会去掉 det_res_ 前缀）"""
    result = {}
    if not txt_path.is_file():
        return result
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        name, payload = line.split("\t", 1)
        key = vis_key_from_name(Path(name).name)
        try:
            boxes = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(boxes, list):
            continue
        if key not in result or box_count(boxes) >= box_count(result[key]):
            result[key] = boxes
    return result


def lookup_boxes(box_map: dict[str, list], key: str) -> list:
    if key in box_map and box_map[key]:
        return box_map[key]
    alt = f"det_res_{key}"
    if alt in box_map and box_map[alt]:
        return box_map[alt]
    return box_map.get(key, [])


def find_original_image(image_root: Path, key: str) -> Path | None:
    candidates = []
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if p.stem.lower() == key:
                candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: len(str(x)))[0]


def load_base_image(image_root: Path, key: str, fallback: np.ndarray) -> np.ndarray:
    orig_path = find_original_image(image_root, key)
    if orig_path is not None:
        base = cv2.imread(str(orig_path))
        if base is not None:
            return base
    return fallback.copy()


def draw_boxes(img: np.ndarray, boxes: list, color, thickness: int) -> np.ndarray:
    vis = img.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        pts = pts.astype(np.int32)
        cv2.polylines(vis, [pts], True, color, thickness)
    return vis


def put_label(img: np.ndarray, text: str, y: int = 28) -> None:
    cv2.putText(
        img,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    w = max(1, int(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h))


def hstack_images(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    padded = []
    for im, lb in zip(images, labels):
        r = resize_to_height(im, h)
        put_label(r, lb)
        padded.append(r)
    return cv2.hconcat(padded)


def build_color_legend(
    width: int, off_color: tuple[int, int, int], cus_color: tuple[int, int, int], height: int = 40
) -> np.ndarray:
    """与下方图像同宽，便于 vconcat。"""
    bar = np.zeros((height, max(1, width), 3), dtype=np.uint8)
    cv2.rectangle(bar, (10, 10), (30, 30), off_color, -1)
    cv2.putText(bar, "Official", (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.rectangle(bar, (150, 10), (170, 30), cus_color, -1)
    cv2.putText(bar, "Custom", (180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return bar


def main():
    args = parse_args()
    image_root = repo_path(args.image_dir)
    official_dir = repo_path(args.official_dir)
    if args.custom_dir:
        custom_dir_list = [repo_path(args.custom_dir)]
    else:
        custom_dir_list = [
            repo_path(x.strip()) for x in args.custom_dirs.split(",") if x.strip()
        ]
    if not custom_dir_list:
        raise SystemExit("请指定 --custom_dir 或 --custom_dirs")

    custom_dir_names = [str(d) for d in custom_dir_list]

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    side_color = parse_bgr(args.side_color)
    off_color = parse_bgr(args.official_color)
    cus_color = parse_bgr(args.custom_color)

    off_vis = index_vis_images(official_dir)
    cus_vis_list = [index_vis_images(d) for d in custom_dir_list]
    cus_vis, vis_dup_warn = merge_maps(cus_vis_list, custom_dir_names)

    off_boxes_map = load_det_results_txt(official_dir / "det_results.txt")
    cus_boxes_list = [
        load_det_results_txt(d / "det_results.txt") for d in custom_dir_list
    ]
    cus_boxes_map, boxes_dup_warn = merge_box_maps(cus_boxes_list, custom_dir_names)

    missing_txt_dirs = [d for d in custom_dir_list if not (d / "det_results.txt").is_file()]
    if missing_txt_dirs:
        print(
            "WARN: 自训目录缺少 det_results.txt，overlay 无法画红框（仅有 det_res_*.jpg 可视化图）：\n"
            + "\n".join(f"  - {d}" for d in missing_txt_dirs)
            + "\n  请用 tools/infer/predict_det.py 或 profile_det_folder（已支持写 txt）重新推理。"
        )

    off_keys = set(off_vis)
    cus_keys = set(cus_vis)
    common_keys = sorted(off_keys & cus_keys)
    only_off = sorted(off_keys - cus_keys)
    only_cus = sorted(cus_keys - off_keys)

    if not common_keys:
        raise SystemExit(
            "官方与自训（合并后）无交集。请检查 det_res_*.jpg 命名。\n"
            f"  official: {len(off_vis)} in {official_dir}\n"
            + "\n".join(
                f"  custom part{i + 1}: {len(m)} in {d}"
                for i, (m, d) in enumerate(zip(cus_vis_list, custom_dir_list))
            )
            + f"\n  custom merged: {len(cus_vis)}"
        )

    n_done = 0
    n_custom_boxes_missing = 0
    for key in common_keys:
        off_img = cv2.imread(str(off_vis[key]))
        cus_img = cv2.imread(str(cus_vis[key]))
        if off_img is None or cus_img is None:
            continue

        stem_out = key

        boxes_o = lookup_boxes(off_boxes_map, key)
        boxes_c = lookup_boxes(cus_boxes_map, key)

        if args.mode in ("side_by_side", "both"):
            base = load_base_image(image_root, key, off_img)
            left = base.copy()
            right = base.copy()
            if boxes_o:
                left = draw_boxes(left, boxes_o, side_color, args.thickness)
            else:
                left = off_img.copy()
            if boxes_c:
                right = draw_boxes(right, boxes_c, side_color, args.thickness)
            else:
                right = cus_img.copy()
            panel = hstack_images(
                [left, right],
                ["Official mobile", args.custom_label],
            )
            out_path = output_dir / f"compare_side_{stem_out}.jpg"
            cv2.imwrite(str(out_path), panel)

        if args.mode in ("overlay", "both"):
            base = load_base_image(image_root, key, off_img)
            if boxes_o or boxes_c:
                if not boxes_c and cus_img is not None:
                    n_custom_boxes_missing += 1
                vis = base.copy()
                if boxes_o:
                    vis = draw_boxes(vis, boxes_o, off_color, args.thickness)
                if boxes_c:
                    vis = draw_boxes(vis, boxes_c, cus_color, args.thickness)
                label = "Green=Official  Red=Custom"
                if not boxes_c:
                    label += "  (no custom coords in det_results.txt)"
                put_label(vis, label, 32)
                legend = build_color_legend(vis.shape[1], off_color, cus_color)
                vis = cv2.vconcat([legend, vis])
            elif cus_img is not None:
                n_custom_boxes_missing += 1
                left = draw_boxes(base.copy(), boxes_o, off_color, args.thickness) if boxes_o else base
                vis = hstack_images(
                    [left, cus_img],
                    ["Official overlay", "Custom vis (no det_results.txt)"],
                )
                legend = build_color_legend(vis.shape[1], off_color, cus_color)
                vis = cv2.vconcat([legend, vis])
            else:
                vis = hstack_images(
                    [off_img, cus_img],
                    ["Official (no txt)", "Custom (no txt)"],
                )
            out_path = output_dir / f"compare_overlay_{stem_out}.jpg"
            cv2.imwrite(str(out_path), vis)

        n_done += 1

    print(f"Official: {len(off_vis)}  Custom (merged): {len(cus_vis)}  Matched: {len(common_keys)}  Wrote: {n_done}")
    print(f"  Custom box coords in det_results.txt: {len(cus_boxes_map)} images")
    if n_custom_boxes_missing:
        print(
            f"  Overlay 缺自训坐标（仅绿框）: {n_custom_boxes_missing} 张。"
            f" 见 compare_side_* 或重新推理生成 det_results.txt"
        )
    if only_off:
        print(f"  Only in official ({len(only_off)}), e.g. {only_off[:5]}")
    if only_cus:
        print(f"  Only in custom ({len(only_cus)}), e.g. {only_cus[:5]}")
    for w in (vis_dup_warn + boxes_dup_warn)[:10]:
        print(w)
    if len(vis_dup_warn + boxes_dup_warn) > 10:
        print(f"  ... and {len(vis_dup_warn + boxes_dup_warn) - 10} more duplicate-key warnings")
    print(f"Output: {output_dir}")
    print(f"  compare_side_*.jpg   : 左官方 | 右自训（均为红框，BGR={args.side_color}）")
    print("  compare_overlay_*.jpg: 原图叠框（绿=官方 红=自训）")


if __name__ == "__main__":
    main()
