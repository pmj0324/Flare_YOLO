#!/usr/bin/env python3
"""
5. YOLO 모델 학습 및 성능지표 출력

학습 설정 (Ultralytics 기본값, model.train()에서 변경 가능):
  - Optimizer: SGD (momentum=0.937, weight_decay=0.0005)
  - Learning rate: lr0=0.01, lrf=0.01 (최종 lr = lr0 * lrf)
  - Scheduler: cosine LR (OneCycleLR 스타일)
  - Early stopping: patience epoch 동안 val mAP 개선 없으면 학습 중단 (기본 50)
"""
import argparse
import configparser
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # val_vis용; pip install pyyaml


def get_iou(box1, box2):
    """IoU 계산. box format: [x1, y1, x2, y2] (torch tensor)."""
    import torch
    # 교집합 좌표
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union


def get_metrics(model_path, data_yaml):
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    v = m.val(data=str(data_yaml), split="val", verbose=False)
    return {
        "mAP@50-95": round(float(v.box.map), 4),
        "mAP@50": round(float(v.box.map50), 4),
        "mAP@75": round(float(v.box.map75), 4),
        "Precision": round(float(v.box.mp), 4),
        "Recall": round(float(v.box.mr), 4),
    }


def save_metrics_txt(metrics: dict, filepath: Path) -> None:
    """성능 지표를 보기 좋게 출력하고 txt 파일로 저장."""
    lines = [
        "=" * 40,
        "성능 지표 (Validation)",
        "=" * 40,
        "",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    lines.extend(["", "=" * 40])
    text = "\n".join(lines)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(text, encoding="utf-8")
    print(text)


def draw_val_gt_vs_pred(model_path: str, data_yaml_path: str, out_dir: Path) -> None:
    """Val 이미지마다 실제(빨강) vs 예측(파랑) bbox를 한 장에 그려 저장."""
    if yaml is None:
        print("val_vis: PyYAML 없음, 건너뜀. pip install pyyaml")
        return
    with open(data_yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    dataset_path = Path(data_cfg.get("path", ".")).resolve()
    val_rel = data_cfg.get("val", "images/val").strip()
    val_img_dir = dataset_path / val_rel
    val_lbl_dir = dataset_path / "labels" / (val_rel.split("/")[-1] if "/" in val_rel else "val")
    if not val_img_dir.is_dir():
        print(f"val_vis: 이미지 폴더 없음 {val_img_dir}, 건너뜀")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from PIL import Image, ImageDraw
    model = YOLO(model_path)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for img_path in sorted(val_img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix not in exts:
            continue
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        # GT (빨강)
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        # 예측 (파랑)
        results = model.predict(source=str(img_path), conf=0.25, iou=0.7, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0]
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=2)
        out_path = out_dir / img_path.name
        img.save(out_path)
    print(f"Val 비교 이미지 저장: {out_dir} (빨강=실제, 파랑=예측)")


def main():
    p = argparse.ArgumentParser(description="YOLO 학습")
    p.add_argument("--config", type=Path, help="config.ini (선택)")
    p.add_argument("--data", type=Path, default=Path("datasets/yolo/data.yaml"))
    p.add_argument("--weight", type=str, default="yolo8l.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=50, help="Early stopping: stop if no improvement for N epochs")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=Path, default=Path("runs/detect"))
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--metrics_file", type=Path, default=None, help="성능 지표 저장 txt 경로 (없으면 project/name/metrics.txt)")
    p.add_argument("--val_vis_dir", type=Path, default=None, help="Val 실제vs예측 이미지 저장 폴더 (없으면 project/name/val_vis)")
    args = p.parse_args()

    cfg = {}
    if args.config and args.config.exists():
        cp = configparser.ConfigParser()
        cp.read(args.config, encoding="utf-8")
        if cp.has_section("Model"):
            cfg["weight"] = cp.get("Model", "weight", fallback="yolo8l.pt")
        if cp.has_section("Data"):
            cfg["yaml_path"] = cp.get("Data", "yaml_path", fallback="")
        if cp.has_section("Training"):
            cfg["epochs"] = cp.getint("Training", "epochs", fallback=100)
            cfg["patience"] = cp.getint("Training", "patience", fallback=50)
            cfg["imgsz"] = cp.getint("Training", "imgsz", fallback=640)
            cfg["batch"] = cp.getint("Training", "batch", fallback=16)
        if cp.has_section("Output"):
            cfg["project"] = cp.get("Output", "project_path", fallback="runs/detect")
            cfg["name"] = cp.get("Output", "run_name", fallback="train")
            cfg["metrics_file"] = cp.get("Output", "metrics_file", fallback="").strip()
            cfg["val_vis_dir"] = cp.get("Output", "val_vis_dir", fallback="").strip()

    data = str(args.data or cfg.get("yaml_path", ""))
    if not data or not Path(data).exists():
        raise SystemExit("--data 또는 config yaml_path 필요")

    from ultralytics import YOLO
    model = YOLO(args.weight or cfg.get("weight", "yolo8l.pt"))
    model.train(
        data=data,
        epochs=args.epochs or cfg.get("epochs", 100),
        patience=args.patience or cfg.get("patience", 50),
        imgsz=args.imgsz or cfg.get("imgsz", 640),
        batch=args.batch or cfg.get("batch", 16),
        device=args.device,
        project=str(args.project or cfg.get("project", "runs/detect")),
        name=args.name or cfg.get("name", "train"),
    )

    project = str(args.project or cfg.get("project", "runs/detect"))
    name = args.name or cfg.get("name", "train")
    best = Path(project) / name / "weights" / "best.pt"
    if best.exists():
        metrics = get_metrics(best, data)
        metrics_file = args.metrics_file or cfg.get("metrics_file") or ""
        metrics_file = Path(metrics_file) if str(metrics_file).strip() else Path(project) / name / "metrics.txt"
        save_metrics_txt(metrics, metrics_file)
        val_vis_dir = args.val_vis_dir or cfg.get("val_vis_dir") or ""
        val_vis_dir = Path(val_vis_dir) if str(val_vis_dir).strip() else Path(project) / name / "val_vis"
        draw_val_gt_vs_pred(str(best), data, val_vis_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
