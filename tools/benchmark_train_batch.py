#!/usr/bin/env python3
"""
Sweep Train.loader.batch_size_per_card and record GPU memory + step time.

Run from PaddleOCR-main root (det or cls configs):

  # 检测
  python tools/benchmark_train_batch.py \\
    -c mobile_v5/configs/det/PP-OCRv5/PP-OCRv5_server_det_ocr_det_dataset_all.yml \\
    --batch-sizes 1 2 4 8 --skip-pretrained

  # 三分类
  python tools/benchmark_train_batch.py \\
    -c mobile_v5/configs/cls/pp_lcnet_v2_textbox_hw3_ocr_det_dataset_all.yml \\
    --batch-sizes 1 2 4 8 16 32 64 128 256 --skip-pretrained \\
    --output ./output/cls/batch_benchmark_hw3.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import os
import sys
import time
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.losses import build_loss
from ppocr.modeling.architectures import build_model
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
import tools.program as program


def _cuda_mb() -> tuple[int | None, int | None]:
    if not paddle.device.is_compiled_with_cuda():
        return None, None
    return (
        paddle.device.cuda.max_memory_reserved() // (1024**2),
        paddle.device.cuda.max_memory_allocated() // (1024**2),
    )


def _forward(model, model_type: str, images, batch):
    if model_type in ("table",):
        return model(images, data=batch[1:])
    if model_type in ("kie", "sr"):
        return model(batch)
    return model(images)


def _reset_cuda_peak() -> None:
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.reset_peak_memory_stats()
        paddle.device.cuda.empty_cache()


def _run_steps(
    config: dict,
    device: str,
    logger,
    seed: int,
    warmup: int,
    steps: int,
) -> dict:
    set_signal_handlers()
    train_loader = build_dataloader(config, "Train", device, logger, seed)
    if len(train_loader) == 0:
        raise RuntimeError("Train dataloader is empty for this batch size.")

    post_process_class = build_post_process(config["PostProcess"], config["Global"])
    model = build_model(config["Architecture"])
    loss_class = build_loss(config["Loss"])
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=1,
        step_each_epoch=max(len(train_loader), 1),
        model=model,
    )
    load_model(config, model, optimizer, config["Architecture"]["model_type"])

    model.train()
    model_type = config["Architecture"].get("model_type", "det")
    use_amp = config["Global"].get("use_amp", False)
    scaler = None
    if use_amp:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=config["Global"].get("scale_loss", 1.0),
            use_dynamic_loss_scaling=config["Global"].get(
                "use_dynamic_loss_scaling", False
            ),
        )

    reader_time = 0.0
    batch_time = 0.0
    samples = 0
    step = 0
    reader_start = time.time()

    def train_step(batch) -> None:
        nonlocal reader_time, batch_time, samples
        reader_time += time.time() - reader_start
        t0 = time.time()
        images = batch[0]
        if scaler:
            with paddle.amp.auto_cast(
                level=config["Global"].get("amp_level", "O2"),
                dtype=config["Global"].get("amp_dtype", "float16"),
            ):
                preds = _forward(model, model_type, images, batch)
            from tools.program import to_float32

            preds = to_float32(preds)
            loss = loss_class(preds, batch)
            scaled = scaler.scale(loss["loss"])
            scaled.backward()
            scaler.minimize(optimizer, scaled)
        else:
            preds = _forward(model, model_type, images, batch)
            loss = loss_class(preds, batch)
            loss["loss"].backward()
            optimizer.step()
        optimizer.clear_grad()
        if not isinstance(lr_scheduler, float):
            lr_scheduler.step()
        batch_time += time.time() - t0
        samples += int(images.shape[0]) if hasattr(images, "shape") else len(images)

    data_iter = iter(train_loader)
    total_steps = warmup + steps
    for _ in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        if step == warmup:
            reader_time = 0.0
            batch_time = 0.0
            samples = 0
            _reset_cuda_peak()
        train_step(batch)
        step += 1
        reader_start = time.time()

    n = max(steps, 1)
    reserved_mb, allocated_mb = _cuda_mb()
    return {
        "avg_reader_s": reader_time / n,
        "avg_batch_s": batch_time / n,
        "ips": samples / batch_time if batch_time > 0 else 0.0,
        "samples_per_step": samples / n,
        "peak_mem_reserved_mb": reserved_mb,
        "peak_mem_allocated_mb": allocated_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--opt", nargs="*", default=[])
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="batch_size_per_card values to test",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/batch_benchmark.csv"),
    )
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="Do not load Global.pretrained_model (faster, random weights)",
    )
    args = parser.parse_args()

    # Reuse PaddleOCR config parser
    sys.argv = [sys.argv[0], "-c", args.config] + sum(
        ([ "-o", x] for x in args.opt), []
    )
    config, device, logger, _ = program.preprocess(is_train=True)
    seed = config["Global"].get("seed", 1024)
    set_seed(seed)

    if args.skip_pretrained:
        config["Global"]["pretrained_model"] = None
        config["Global"]["checkpoints"] = None

    config["Global"]["distributed"] = False
    if paddle.device.is_compiled_with_cuda():
        paddle.device.set_device("gpu:0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for bs in args.batch_sizes:
        cfg = copy.deepcopy(config)
        cfg["Train"]["loader"]["batch_size_per_card"] = bs
        logger.info(f"===== batch_size_per_card={bs} =====")
        _reset_cuda_peak()
        gc.collect()
        try:
            stats = _run_steps(cfg, device, logger, seed, args.warmup, args.steps)
            status = "ok"
            err = ""
        except RuntimeError as exc:
            stats = {
                "avg_reader_s": "",
                "avg_batch_s": "",
                "ips": "",
                "samples_per_step": "",
                "peak_mem_reserved_mb": "",
                "peak_mem_allocated_mb": "",
            }
            status = "oom" if "memory" in str(exc).lower() else "error"
            err = str(exc)
            logger.error(f"batch={bs} failed: {exc}")
        except Exception as exc:
            stats = {
                "avg_reader_s": "",
                "avg_batch_s": "",
                "ips": "",
                "samples_per_step": "",
                "peak_mem_reserved_mb": "",
                "peak_mem_allocated_mb": "",
            }
            status = "error"
            err = str(exc)
            logger.error(f"batch={bs} failed: {exc}")

        row = {
            "batch_size_per_card": bs,
            "warmup_steps": args.warmup,
            "bench_steps": args.steps,
            "status": status,
            "error": err,
            **stats,
        }
        rows.append(row)
        logger.info(
            f"batch={bs} status={status} "
            f"reader={stats.get('avg_reader_s', '')}s "
            f"batch={stats.get('avg_batch_s', '')}s "
            f"mem_reserved={stats.get('peak_mem_reserved_mb', '')}MB "
            f"mem_allocated={stats.get('peak_mem_allocated_mb', '')}MB"
        )
        del cfg
        gc.collect()
        _reset_cuda_peak()

    fieldnames = list(rows[0].keys()) if rows else []
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved benchmark -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
