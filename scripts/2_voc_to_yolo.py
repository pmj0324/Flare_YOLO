#!/usr/bin/env python3
"""
2. VOC → YOLO 변환: XML 라벨을 YOLO txt로 변환하고 train/val 분할
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"No size in {xml_path}")
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return w, h, boxes


def voc_to_yolo_line(xmin, ymin, xmax, ymax, img_w, img_h):
    xc = (xmin + xmax) / 2 / img_w
    yc = (ymin + ymax) / 2 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def iter_voc_items(voc_dir: Path):
    xmls = list(voc_dir.glob("*.xml")) + list((voc_dir / "annotations").glob("*.xml"))
    seen = set()
    for xml_path in xmls:
        stem = xml_path.stem
        if stem in seen:
            continue
        for base in (voc_dir, voc_dir / "images"):
            if not base.is_dir():
                continue
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                img = base / f"{stem}{ext}"
                if img.exists():
                    seen.add(stem)
                    yield stem, img, xml_path
                    break
            else:
                continue
            break


def main():
    p = argparse.ArgumentParser(description="VOC → YOLO 변환 및 train/val 분할")
    p.add_argument("--voc_dir", type=Path, default=Path("data"), help="VOC 데이터 (images/, annotations/)")
    p.add_argument("--out_dir", type=Path, default=Path("datasets/yolo"))
    p.add_argument("--val_ratio", type=float, default=0.1, help="val 비율 (기본 9:1)")
    p.add_argument("--classes", nargs="+", default=["row"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items = list(iter_voc_items(args.voc_dir))
    if not items:
        raise SystemExit(f"VOC 데이터 없음: {args.voc_dir}")

    cls2id = {c: i for i, c in enumerate(args.classes)}
    random.seed(args.seed)
    random.shuffle(items)
    n_val = max(0, min(len(items) - 1, int(len(items) * args.val_ratio)))
    val_items, train_items = items[:n_val], items[n_val:]

    for d in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out_dir / d).mkdir(parents=True, exist_ok=True)

    def process(stem, img_path, xml_path, split):
        img_out = args.out_dir / "images" / split / img_path.name
        lbl_out = args.out_dir / "labels" / split / f"{stem}.txt"
        shutil.copy2(img_path, img_out)
        w, h, boxes = parse_voc(xml_path)
        lines = []
        for name, xmin, ymin, xmax, ymax in boxes:
            if name in cls2id:
                line = voc_to_yolo_line(xmin, ymin, xmax, ymax, w, h)
                lines.append(f"{cls2id[name]} {line}")
        lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    for item in train_items:
        process(*item, "train")
    for item in val_items:
        process(*item, "val")

    yaml = (
        f"path: {args.out_dir.resolve()}\n"
        "train: images/train\nval: images/val\n"
        f"nc: {len(args.classes)}\nnames:\n" + "\n".join(f"  - {c}" for c in args.classes) + "\n"
    )
    (args.out_dir / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"출력: {args.out_dir} | train {len(train_items)}, val {len(val_items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
