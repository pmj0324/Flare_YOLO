#!/usr/bin/env python3
"""
3. train:val 비율 재분할 (기본 9:1)
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="YOLO 데이터셋 train:val 비율 분할")
    p.add_argument("--input", type=Path, default=Path("datasets/yolo"))
    p.add_argument("--output", type=Path, default=None, help="없으면 input 덮어쓰기")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.output or args.input

    img_dir = args.input / "images"
    lbl_dir = args.input / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SystemExit(f"images/ 또는 labels/ 없음: {args.input}")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    items = []
    for sub in ("", "train", "val"):
        pi = img_dir / sub if sub else img_dir
        pl = lbl_dir / sub if sub else lbl_dir
        if pi.is_dir() and pl.is_dir():
            for f in pi.iterdir():
                if f.is_file() and f.suffix in exts:
                    lbl = pl / f"{f.stem}.txt"
                    if lbl.exists():
                        items.append((f.stem, f, lbl))
    by_stem = {s: (s, i, l) for s, i, l in items}
    items = list(by_stem.values())
    if not items:
        raise SystemExit("이미지-라벨 쌍 없음")

    random.seed(args.seed)
    random.shuffle(items)
    n_val = max(0, min(len(items) - 1, int(len(items) * args.val_ratio)))
    val_items, train_items = items[:n_val], items[n_val:]

    for d in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / d).mkdir(parents=True, exist_ok=True)

    for stem, img, lbl in train_items:
        shutil.copy2(img, out / "images" / "train" / img.name)
        shutil.copy2(lbl, out / "labels" / "train" / f"{stem}.txt")
    for stem, img, lbl in val_items:
        shutil.copy2(img, out / "images" / "val" / img.name)
        shutil.copy2(lbl, out / "labels" / "val" / f"{stem}.txt")

    src_yaml = args.input / "data.yaml"
    dst_yaml = out / "data.yaml"
    if src_yaml.exists():
        t = src_yaml.read_text(encoding="utf-8").replace(
            str(args.input.resolve()), str(out.resolve())
        )
        dst_yaml.write_text(t, encoding="utf-8")
    else:
        dst_yaml.write_text(
            f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  - row\n",
            encoding="utf-8",
        )

    print(f"출력: {out} | train {len(train_items)}, val {len(val_items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
