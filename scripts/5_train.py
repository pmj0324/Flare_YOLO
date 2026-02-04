#!/usr/bin/env python3
"""
5. YOLO 모델 학습 및 성능지표 출력
"""
import argparse
import configparser
from pathlib import Path


def get_metrics(model_path, data_yaml):
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    v = m.val(data=str(data_yaml), split="val", verbose=False)
    return {
        "mAP@50-95": round(float(v.box.map), 4),
        "mAP@50": round(float(v.box.map50), 4),
        "Precision": round(float(v.box.mp), 4),
        "Recall": round(float(v.box.mr), 4),
    }


def main():
    p = argparse.ArgumentParser(description="YOLO 학습")
    p.add_argument("--config", type=Path, help="config.ini (선택)")
    p.add_argument("--data", type=Path, default=Path("datasets/yolo/data.yaml"))
    p.add_argument("--weight", type=str, default="yolo11m.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=Path, default=Path("runs/detect"))
    p.add_argument("--name", type=str, default="train")
    args = p.parse_args()

    cfg = {}
    if args.config and args.config.exists():
        cp = configparser.ConfigParser()
        cp.read(args.config, encoding="utf-8")
        if cp.has_section("Model"):
            cfg["weight"] = cp.get("Model", "weight", fallback="yolo11s.pt")
        if cp.has_section("Data"):
            cfg["yaml_path"] = cp.get("Data", "yaml_path", fallback="")
        if cp.has_section("Training"):
            cfg["epochs"] = cp.getint("Training", "epochs", fallback=100)
            cfg["imgsz"] = cp.getint("Training", "imgsz", fallback=640)
            cfg["batch"] = cp.getint("Training", "batch", fallback=16)
        if cp.has_section("Output"):
            cfg["project"] = cp.get("Output", "project_path", fallback="runs/detect")
            cfg["name"] = cp.get("Output", "run_name", fallback="train")

    data = str(args.data or cfg.get("yaml_path", ""))
    if not data or not Path(data).exists():
        raise SystemExit("--data 또는 config yaml_path 필요")

    from ultralytics import YOLO
    model = YOLO(args.weight or cfg.get("weight", "yolo11s.pt"))
    model.train(
        data=data,
        epochs=args.epochs or cfg.get("epochs", 100),
        imgsz=args.imgsz or cfg.get("imgsz", 640),
        batch=args.batch or cfg.get("batch", 16),
        device=args.device,
        project=str(args.project or cfg.get("project", "runs/detect")),
        name=args.name or cfg.get("name", "train"),
    )

    best = Path(args.project or cfg.get("project", "runs/detect")) / (args.name or cfg.get("name", "train")) / "weights" / "best.pt"
    if best.exists():
        print("\n=== 성능지표 ===")
        for k, v in get_metrics(best, data).items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
