#!/usr/bin/env python3
"""
4. 라벨 검증: YOLO bbox를 이미지에 그려 저장
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes(img_path: Path, lbl_path: Path, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x, y, w, h = int(float(parts[0])), *map(float, parts[1:])
        x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, img.width, img.height)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    p = argparse.ArgumentParser(description="YOLO 라벨 시각화 검증")
    p.add_argument("--labels_dir", type=Path, help="라벨 디렉토리")
    p.add_argument("--images_dir", type=Path, help="이미지 디렉토리")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/verify"))
    args = p.parse_args()

    if not args.labels_dir or not args.images_dir:
        raise SystemExit("--labels_dir, --images_dir 필요")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for lbl in sorted(args.labels_dir.glob("*.txt")):
        img = None
        for e in exts:
            p = args.images_dir / f"{lbl.stem}{e}"
            if p.exists():
                img = p
                break
        if img:
            out = args.out_dir / img.name
            draw_boxes(img, lbl, out)
    print(f"저장: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
