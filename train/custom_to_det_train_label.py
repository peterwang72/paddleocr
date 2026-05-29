# Copyright (c) 2026 PaddleOCR Authors. SPDX-License-Identifier: Apache-2.0
"""
将常见自定义标注转为 PaddleOCR 文本检测训练格式（SimpleDataSet + DetLabelEncode）。

目标：每行
    <相对 data_dir 的图片路径>\\t<JSON 列表>
JSON 列表元素：
    {"transcription": "文字或###", "points": [[x1,y1], [x2,y2], ...]}
- 框须为顺时针或逆时针多边形顶点；4 点矩形即可。
- transcription 为 \"*\" 或 \"###\" 时表示忽略该区域（不参与 loss）。

用法示例见文件末尾 __main__。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Union

Number = Union[int, float]


def instances_to_json_line(
    image_rel_path: str, instances: List[Dict[str, Any]], ensure_ascii: bool = False
) -> str:
    """单张图 -> 一行标注（不含换行符后的多余字符）。"""
    norm: List[Dict[str, Any]] = []
    for inst in instances:
        pts = inst["points"]
        if len(pts) < 3:
            continue
        row = {
            "transcription": str(inst.get("transcription", "")),
            "points": [[float(p[0]), float(p[1])] for p in pts],
        }
        norm.append(row)
    if not norm:
        raise ValueError(f"no valid boxes: {image_rel_path}")
    return image_rel_path + "\t" + json.dumps(norm, ensure_ascii=ensure_ascii)


def bbox_xywh_to_quad(x: Number, y: Number, w: Number, h: Number) -> List[List[float]]:
    """水平矩形 (x,y,w,h) -> 四点逆时针。"""
    return [
        [float(x), float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x), float(y + h)],
    ]


def _collect_labelme_shapes(data: dict) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sh in data.get("shapes", []):
        pts = sh.get("points") or []
        if len(pts) < 3:
            continue
        label = sh.get("label", "")
        out.append({"transcription": label, "points": pts})
    return out


def convert_labelme_pair_dir(
    image_json_dir: str,
    output_txt: str,
    image_rel_prefix: str = "",
) -> None:
    """
    目录下每张图旁有同名 .json（LabelMe 导出）。
    image_rel_prefix: 写入 txt 时路径前缀，如 \"images/\"（相对训练 data_dir）。
    """
    lines: List[str] = []
    for name in sorted(os.listdir(image_json_dir)):
        if not name.lower().endswith(".json"):
            continue
        stem = name[:-5]
        json_path = os.path.join(image_json_dir, name)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_file = data.get("imagePath") or (stem + ".jpg")
        # 若 json 里带子路径，只取文件名再与 stem 尝试匹配常见后缀
        img_file = os.path.basename(img_file)
        base = os.path.join(image_json_dir, os.path.splitext(img_file)[0])
        img_rel = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            if os.path.isfile(base + ext):
                img_rel = os.path.splitext(img_file)[0] + ext
                break
        if img_rel is None:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                if os.path.isfile(os.path.join(image_json_dir, stem + ext)):
                    img_rel = stem + ext
                    break
        if img_rel is None:
            raise FileNotFoundError(f"no image for labelme json: {json_path}")

        rel = (
            os.path.join(image_rel_prefix, img_rel).replace("\\", "/")
            if image_rel_prefix
            else img_rel.replace("\\", "/")
        )
        inst = _collect_labelme_shapes(data)
        if not inst:
            continue
        lines.append(instances_to_json_line(rel, inst))

    os.makedirs(os.path.dirname(os.path.abspath(output_txt)) or ".", exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} lines -> {output_txt}")


def convert_icdar_polygon_csv(
    csv_path: str,
    output_txt: str,
    image_col_is_first: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
) -> None:
    """
    每行一张框：首列为图片相对路径，随后为 x1,y1,x2,y2,...（偶数个坐标）,最后一列为 transcription。
    若 image_col_is_first=False，则格式为：x1,y1,...,xn,yn,transcription,image_rel_path
    """
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(csv_path, "r", encoding=encoding) as f:
        for raw in f:
            line = raw.strip("\n\r").replace("\ufeff", "")
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(delimiter)]
            if image_col_is_first:
                img_rel = parts[0]
                rest = parts[1:]
            else:
                img_rel = parts[-1]
                rest = parts[:-1]
            if len(rest) < 9:
                continue
            coords = rest[:-1]
            txt = rest[-1]
            pts: List[List[float]] = []
            for i in range(0, len(coords), 2):
                if i + 1 >= len(coords):
                    break
                pts.append([float(coords[i]), float(coords[i + 1])])
            if len(pts) < 3:
                continue
            by_image[img_rel.replace("\\", "/")].append(
                {"transcription": txt, "points": pts}
            )

    lines = [
        instances_to_json_line(img, insts)
        for img, insts in sorted(by_image.items())
        if insts
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_txt)) or ".", exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} images -> {output_txt}")


def convert_coco_instances(
    coco_json_path: str,
    output_txt: str,
    image_rel_prefix: str = "",
) -> None:
    """COCO detection：bbox 为 [x,y,w,h]；segmentation 为多边形时优先用 segmentation。"""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        im_id = ann["image_id"]
        if im_id not in id_to_file:
            continue
        fname = id_to_file[im_id].replace("\\", "/")
        rel = (
            os.path.join(image_rel_prefix, fname).replace("\\", "/")
            if image_rel_prefix
            else fname
        )
        seg = ann.get("segmentation")
        if seg and isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
            poly = seg[0]
            pts = [[float(poly[i]), float(poly[i + 1])] for i in range(0, len(poly), 2)]
            if len(pts) >= 3:
                txt = ann.get("text", "") or ann.get("rec", "") or "###"
                by_image[rel].append({"transcription": str(txt), "points": pts})
                continue
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            pts = bbox_xywh_to_quad(bbox[0], bbox[1], bbox[2], bbox[3])
            txt = ann.get("text", "") or "###"
            by_image[rel].append({"transcription": str(txt), "points": pts})

    lines = [
        instances_to_json_line(img, insts)
        for img, insts in sorted(by_image.items())
        if insts
    ]
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} images -> {output_txt}")


def convert_voc_style_txts(
    gt_dir: str,
    image_root: str,
    output_txt: str,
    image_rel_prefix: str = "",
    exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> None:
    """
    gt_dir 下 gt_*.txt（与 ppocr/utils/gen_label.py 相同）：每行
        x1,y1,...,x8,transcription
    文件名 gt_img123.txt -> 图片 img123.<ext>
    """
    lines: List[str] = []
    for name in sorted(os.listdir(gt_dir)):
        if not (name.startswith("gt_") and name.endswith(".txt")):
            continue
        stem = name[3:-4]
        img_rel = None
        for ext in exts:
            cand = os.path.join(image_root, stem + ext)
            if os.path.isfile(cand):
                img_rel = stem + ext
                break
        if img_rel is None:
            continue
        rel = (
            os.path.join(image_rel_prefix, img_rel).replace("\\", "/")
            if image_rel_prefix
            else img_rel.replace("\\", "/")
        )
        inst: List[Dict[str, Any]] = []
        with open(os.path.join(gt_dir, name), "r", encoding="utf-8-sig") as f:
            for line in f:
                tmp = line.strip("\n\r").replace("\ufeff", "").split(",")
                if len(tmp) < 9:
                    continue
                points = tmp[:8]
                s: List[List[float]] = []
                for i in range(0, len(points), 2):
                    s.append([float(points[i]), float(points[i + 1])])
                inst.append({"transcription": tmp[8], "points": s})
        if inst:
            lines.append(instances_to_json_line(rel, inst))

    os.makedirs(os.path.dirname(os.path.abspath(output_txt)) or ".", exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} lines -> {output_txt}")


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("labelme", help="目录：图片与同名 LabelMe .json 同层")
    p1.add_argument("--dir", required=True, help="含 .jpg/.png 与同名 .json 的目录")
    p1.add_argument("--out", required=True, help="输出 train.txt / val.txt")
    p1.add_argument(
        "--rel-prefix",
        default="",
        help="写入标注中的相对路径前缀（相对 yaml 里的 data_dir）",
    )

    p2 = sub.add_parser("csv", help="扁平 CSV：每行一个框，多行可同图")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument(
        "--image-last",
        action="store_true",
        help="若指定则末列为图片相对路径（默认首列）",
    )
    p2.add_argument("--delimiter", default=",")

    p3 = sub.add_parser("coco", help="COCO json（bbox 或 segmentation）")
    p3.add_argument("--json", required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--rel-prefix", default="")

    p4 = sub.add_parser("voc_gt", help="gt_*.txt 目录 + 图片根目录（ICDAR 风格）")
    p4.add_argument("--gt-dir", required=True)
    p4.add_argument("--image-root", required=True)
    p4.add_argument("--out", required=True)
    p4.add_argument("--rel-prefix", default="")

    args = p.parse_args()
    if args.cmd == "labelme":
        convert_labelme_pair_dir(args.dir, args.out, args.rel_prefix)
    elif args.cmd == "csv":
        convert_icdar_polygon_csv(
            args.csv,
            args.out,
            image_col_is_first=not args.image_last,
            delimiter=args.delimiter,
        )
    elif args.cmd == "coco":
        convert_coco_instances(args.json, args.out, args.rel_prefix)
    elif args.cmd == "voc_gt":
        convert_voc_style_txts(args.gt_dir, args.image_root, args.out, args.rel_prefix)


if __name__ == "__main__":
    _main()
