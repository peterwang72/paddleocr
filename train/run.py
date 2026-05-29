import os
from pathlib import Path
from paddleocr import PaddleOCR

# =========================
# 路径配置
# =========================
INPUT_DIR = Path("data")
TEST_IMAGE = INPUT_DIR / "7_1785x2500_bing_online.jpg"
OUTPUT_ROOT = Path("output_exp")

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# PP-OCRv5 检测参数实验组
# =========================
EXPERIMENTS = [
    {
        "name": "A_baseline_960",
        "det": {
            "text_det_limit_side_len": 960,
            "text_det_limit_type": "max",
            "text_det_thresh": 0.3,
            "text_det_box_thresh": 0.6,
            "text_det_unclip_ratio": 1.5,
        },
    },
    {
        "name": "B_recall_1216",
        "det": {
            "text_det_limit_side_len": 1216,
            "text_det_limit_type": "max",
            "text_det_thresh": 0.25,
            "text_det_box_thresh": 0.5,
            "text_det_unclip_ratio": 1.8,
        },
    },
    {
        "name": "C_thinstroke_1216",
        "det": {
            "text_det_limit_side_len": 1216,
            "text_det_limit_type": "max",
            "text_det_thresh": 0.2,
            "text_det_box_thresh": 0.45,
            "text_det_unclip_ratio": 2.0,
        },
    },
    {
        "name": "D_precision_960",
        "det": {
            "text_det_limit_side_len": 960,
            "text_det_limit_type": "max",
            "text_det_thresh": 0.35,
            "text_det_box_thresh": 0.65,
            "text_det_unclip_ratio": 1.3,
        },
    },
]

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS

def main():
    if not is_image_file(TEST_IMAGE):
        print(f"[ERROR] 测试图片不存在或不是图片: {TEST_IMAGE.resolve()}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        exp_dir = OUTPUT_ROOT / exp_name
        vis_dir = exp_dir / "vis"
        json_dir = exp_dir / "json"
        txt_dir = exp_dir / "txt"
        vis_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[EXP] {exp_name}")
        print(f"[EXP] det params: {exp['det']}")

        ocr = PaddleOCR(
            ocr_version="PP-OCRv5",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            return_word_box=True,
            lang="ch",
            **exp["det"],
        )

        try:
            results = ocr.predict(input=str(TEST_IMAGE))
            for res in results:
                stem = TEST_IMAGE.stem
                res.save_to_img(str(vis_dir))
                res.save_to_json(str(json_dir))

                txt_path = txt_dir / f"{stem}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    data = res.json
                    rec_texts = data.get("rec_texts", [])
                    rec_scores = data.get("rec_scores", [])
                    rec_boxes = data.get("rec_boxes", [])
                    for i, text in enumerate(rec_texts):
                        score = rec_scores[i] if i < len(rec_scores) else None
                        box = rec_boxes[i] if i < len(rec_boxes) else None
                        f.write(f"text: {text}\n")
                        f.write(f"score: {score}\n")
                        f.write(f"box: {box}\n")
                        f.write("-" * 50 + "\n")

            print(f"[EXP] done. outputs in: {exp_dir.resolve()}")
        except Exception as e:
            print(f"[EXP][ERROR] failed: {e}")

if __name__ == "__main__":
    main()