#!/usr/bin/env python3
"""
5. YOLO 모델 학습 및 성능지표 출력

학습 설정 (Ultralytics 기본값, config/CLI로 변경 가능):
  - Optimizer: 기본 auto(=SGD). AdamW 쓰려면 config [Training] optimizer=AdamW, lr0=0.001 권장.
  - SGD: momentum=0.937, lr0=0.01. Adam/AdamW: lr0=1e-3 권장.
  - Scheduler: cosine LR. Early stopping: patience epoch.
"""
import argparse
import configparser
import os
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
    p.add_argument("--weight", type=str, default=None, help="사용할 모델 가중치 (config 없을 때만 필요)")
    p.add_argument("--optimizer", type=str, default=None, help="SGD, Adam, AdamW, NAdam, RAdam, RMSProp, auto")
    p.add_argument("--lr0", type=float, default=None, help="Initial LR (SGD 0.01, AdamW 0.001 권장)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=50, help="Early stopping: stop if no improvement for N epochs")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--output_dir", type=Path, default=None, help="아웃풋 저장 경로 (weights, metrics.txt, val_vis 모두 여기 아래)")
    args = p.parse_args()

    # 실행 시 작업 디렉터리를 스크립트 기준 프로젝트 루트(Flare_YOLO)로 고정 → config·다운로드 경로 일치
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # config 경로: 지정한 경로가 없거나 없으면 프로젝트 루트의 config.ini 사용
    config_path = Path(args.config).resolve() if args.config else None
    if not config_path or not config_path.exists():
        config_path = Path(__file__).resolve().parent.parent / "config.ini"
    cfg = {}
    if config_path.exists():
        cp = configparser.ConfigParser()
        cp.read(config_path, encoding="utf-8")
        if cp.has_section("Model"):
            cfg["weight"] = cp.get("Model", "weight", fallback="yolo8l.pt").strip()
        if cp.has_section("Data"):
            cfg["yaml_path"] = cp.get("Data", "yaml_path", fallback="").strip()
        if cp.has_section("Training"):
            cfg["optimizer"] = cp.get("Training", "optimizer", fallback="auto").strip()
            cfg["lr0"] = cp.getfloat("Training", "lr0", fallback=0.01)
            cfg["epochs"] = cp.getint("Training", "epochs", fallback=100)
            cfg["patience"] = cp.getint("Training", "patience", fallback=50)
            cfg["imgsz"] = cp.getint("Training", "imgsz", fallback=640)
            cfg["batch"] = cp.getint("Training", "batch", fallback=16)
        if cp.has_section("Output"):
            raw_out = cp.get("Output", "output_dir", fallback="").strip()
            cfg["output_dir"] = str(Path(raw_out).resolve()) if raw_out else ""

    data = str(args.data or cfg.get("yaml_path", ""))
    if not data or not Path(data).exists():
        raise SystemExit("--data 또는 config yaml_path 필요")

    # 아웃풋 저장 경로 하나 (CLI 우선, 없으면 config, 없으면 runs/detect/train)
    output_dir = args.output_dir or cfg.get("output_dir") or "runs/detect/train"
    output_dir = Path(output_dir).resolve()
    # Ultralytics는 project/name 형태로 받으므로 분리
    project = str(output_dir.parent)
    name = output_dir.name
    print(f"Config: {config_path}")
    print(f"저장 경로: {output_dir}")

    # 사용할 가중치: config 또는 CLI에 지정된 것만 사용. 없으면 에러 후 종료
    weight = (args.weight or cfg.get("weight") or "").strip()
    if not weight:
        raise SystemExit(
            "config에 weight가 없거나 입력이 잘못되었습니다. "
            "사용할 모델 가중치를 config.ini [Model] weight= 또는 --weight 로 지정해 주세요."
        )
    print(f"사용할 가중치 (config/CLI): {weight}")

    # 지정한 모델만 사용: 로컬에 있으면 그대로, 없으면 지정 이름으로만 다운로드 시도 (yolo8* → yolov8*.pt 변환)
    weight_path = Path(weight)
    if not weight_path.is_absolute():
        weight_path = project_root / weight_path.name
    if not weight_path.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        download_name = weight.replace("yolo8", "yolov8", 1) if "yolo8" in weight and weight.endswith(".pt") else weight
        try:
            weight = attempt_download_asset(download_name)
        except Exception as e:
            raise SystemExit(
                f"지정한 모델을 찾을 수 없고 다운로드에도 실패했습니다: {weight}\n  오류: {e}"
            ) from e
        if not weight or not Path(weight).exists():
            raise SystemExit(
                f"지정한 모델을 찾을 수 없고 다운로드에도 실패했습니다: {weight}"
            )
    else:
        weight = str(weight_path)
    print(f"모델 가중치 (로드): {weight}")

    from ultralytics import YOLO
    model = YOLO(weight)
    train_kw = dict(
        data=data,
        epochs=args.epochs or cfg.get("epochs", 100),
        patience=args.patience or cfg.get("patience", 50),
        imgsz=args.imgsz or cfg.get("imgsz", 640),
        batch=args.batch or cfg.get("batch", 16),
        device=args.device,
        project=project,
        name=name,
    )
    opt = args.optimizer if args.optimizer is not None else cfg.get("optimizer")
    if opt is not None:
        train_kw["optimizer"] = opt
    lr0 = args.lr0 if args.lr0 is not None else cfg.get("lr0")
    if lr0 is not None:
        train_kw["lr0"] = lr0
    model.train(**train_kw)

    best = output_dir / "weights" / "best.pt"
    if best.exists():
        metrics = get_metrics(best, data)
        save_metrics_txt(metrics, output_dir / "metrics.txt")
        draw_val_gt_vs_pred(str(best), data, output_dir / "val_vis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
