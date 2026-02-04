#!/usr/bin/env python3
"""
1. 데이터 전처리: VOC 형식 데이터를 images/, annotations/, unlabeled/ 로 분류
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="데이터 정리: images, annotations, unlabeled 분류")
    p.add_argument("--data_dir", type=Path, default=Path("data"), help="data 폴더 경로")
    args = p.parse_args()

    data = args.data_dir
    for sub in ("images", "annotations", "unlabeled"):
        (data / sub).mkdir(parents=True, exist_ok=True)

    ext = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    xml_files = list(data.glob("*.xml")) + list((data / "annotations").glob("*.xml"))
    xml_stems = {f.stem for f in xml_files}

    moved = 0
    for f in data.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            stem = f.stem
            if stem not in xml_stems and not (data / "annotations" / f"{stem}.xml").exists():
                dest = data / "unlabeled" / f.name
                f.rename(dest)
                moved += 1
                print(f"  {f.name} → unlabeled/")
            else:
                dest = data / "images" / f.name
                if f.parent != data / "images":
                    f.rename(dest)
                    moved += 1
        elif f.suffix.lower() == ".xml":
            dest = data / "annotations" / f.name
            if f.parent != data / "annotations":
                f.rename(dest)
                moved += 1

    n_img = len(list((data / "images").glob("*.*")))
    n_ann = len(list((data / "annotations").glob("*.xml")))
    n_unl = len(list((data / "unlabeled").glob("*.*")))
    print(f"images: {n_img}, annotations: {n_ann}, unlabeled: {n_unl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
