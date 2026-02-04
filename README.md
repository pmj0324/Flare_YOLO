# Flare_YOLO

YOLO 객체 탐지용 데이터 전처리 및 학습 파이프라인

## 구조

```
Flare_YOLO/
├── scripts/          # 역할별 Python 스크립트 (1개 = 1역할)
│   ├── 1_preprocess.py   # 데이터 정리
│   ├── 2_voc_to_yolo.py  # VOC → YOLO 변환 + 분할
│   ├── 3_split.py        # train:val 비율 재분할
│   ├── 4_verify.py       # 라벨 시각화 검증
│   └── 5_train.py        # YOLO 학습
├── data/             # 원본 데이터 (Git 포함)
│   ├── images/
│   ├── annotations/
│   └── unlabeled/
├── config.ini.example
└── requirements.txt
```

## 사용법

```bash
pip install -r requirements.txt
cd Flare_YOLO
```

### 1. 데이터 정리
```bash
python scripts/1_preprocess.py --data_dir data
```

### 2. VOC → YOLO 변환 (train:val 9:1)
```bash
python scripts/2_voc_to_yolo.py --voc_dir data --out_dir datasets/yolo --val_ratio 0.1
```

### 3. 비율 재분할 (선택)
```bash
python scripts/3_split.py --input datasets/yolo --output datasets/yolo_9_1 --val_ratio 0.1
```

### 4. 라벨 검증
```bash
python scripts/4_verify.py --labels_dir datasets/yolo/labels/train --images_dir datasets/yolo/images/train --out_dir outputs/verify
```

### 5. 학습
```bash
python scripts/5_train.py --data datasets/yolo/data.yaml --weight yolo11m.pt
```

또는 config.ini 사용:
```bash
cp config.ini.example config.ini
# config.ini 수정 후
python scripts/5_train.py --config config.ini
```
