# Flare_YOLO

YOLO 객체 탐지용 데이터 전처리 및 학습 파이프라인 (기본: YOLOv8l)

## 구조

```
Flare_YOLO/
├── scripts/
│   ├── 1_preprocess.py   # 데이터 정리 (images / annotations / unlabeled)
│   ├── 2_voc_to_yolo.py  # VOC → YOLO 변환 + train/val 분할
│   ├── 3_split.py        # train:val 비율 재분할 (선택)
│   ├── 4_verify.py       # 라벨 시각화 검증
│   └── 5_train.py        # YOLO 학습 + 지표 저장 + Val 비교 이미지
├── data/
│   ├── images/           # 라벨 있는 이미지
│   ├── annotations/      # Pascal VOC XML
│   └── unlabeled/        # 라벨 없는 이미지
├── config.ini.example    # 설정 예시 (복사 후 config.ini 로 사용)
└── requirements.txt
```

## 설치

```bash
pip install -r requirements.txt
cd Flare_YOLO
```

## 사용법

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

### 4. 라벨 검증 (선택)

```bash
python scripts/4_verify.py --labels_dir datasets/yolo/labels/train --images_dir datasets/yolo/images/train --out_dir outputs/verify
```

### 5. 학습

**CLI만 사용:**

```bash
python scripts/5_train.py --data datasets/yolo/data.yaml
```

기본 모델: `yolo8l.pt`. 저장 경로 기본값: `runs/detect/train/`.

**config.ini 사용 (저장 경로·학습 설정 한 곳에서 관리):**

```bash
cp config.ini.example config.ini
# config.ini [Output] 에서 output_dir, [Data] 에서 yaml_path 등 수정
python scripts/5_train.py --config config.ini
```

- **저장 경로:** `[Output]` 의 `output_dir` 하나만 지정.  
  weights, metrics.txt, val_vis 등 **모든 아웃풋이 이 경로 아래**에 저장됨. 폴더 없으면 자동 생성.

- **학습 종료 후 자동으로:**
  - 성능 지표가 터미널에 출력되고 `{output_dir}/metrics.txt` 에 저장됨.
  - Val 비교 이미지가 `{output_dir}/val_vis/` 에 저장됨. (빨강 = 실제 GT, 파랑 = 예측)

- **Early stopping:** `patience` epoch 동안 val mAP 개선이 없으면 학습 중단.

## 모델 가중치 사용법

**사용할 모델은 config 또는 CLI로 하나만 지정하면 됩니다.** 지정한 이름만 다운로드·사용됩니다.

### 지정 방법

| 방법 | 예시 |
|------|------|
| **config.ini** | `[Model]` 섹션에 `weight = yolo11l.pt` |
| **CLI** | `python scripts/5_train.py --config config.ini --weight yolo11l.pt` |

config에 `weight = yolo11l.pt` 가 있으면 11l만 쓰고, 26n 등 다른 모델은 받지 않습니다.  
(26n이 받아진 경우는 Ultralytics를 다른 명령·경로로 쓸 때일 수 있음. 이 스크립트는 `config` / `--weight` 값만 사용합니다.)

### 사용 가능한 이름 (예시)

| 시리즈 | 이름 예시 | 비고 |
|--------|-----------|------|
| **8** | `yolo8n`, `yolo8s`, `yolo8m`, `yolo8l`, `yolo8x` | 다운로드 시 `yolov8*.pt` 로 저장됨 (같은 모델) |
| **11** | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` | 그대로 `yolo11*.pt` 로 다운로드 |
| **12** | `yolo12n` ~ `yolo12x` | 그대로 다운로드 |

- **n**=nano, **s**=small, **m**=medium, **l**=large, **x**=extra-large  
- 로컬에 해당 `.pt` 가 없으면 자동 다운로드. 있으면 그 파일 사용.

### 11l만 쓰고 싶을 때

- **config.ini** 에서 `weight = yolo11l.pt` 로 두고 `python scripts/5_train.py --config config.ini` 실행.  
- 실행 시 터미널에 `모델 가중치: ...` 가 찍히므로, 실제로 어떤 파일이 쓰이는지 확인할 수 있습니다.

## 출력 디렉터리 (5_train.py)

`output_dir` 예: `/home/.../Flare_YOLO/yolo_11l` 또는 `runs/detect/yolo_8l`

| 경로 | 내용 |
|------|------|
| `weights/best.pt` | 최적 가중치 |
| `metrics.txt` | 성능 지표 (mAP@50-95, mAP@50, mAP@75, Precision, Recall) |
| `val_vis/` | Val 이미지별 실제(빨강) vs 예측(파랑) bbox |

## 의존성

- Python 3.10+
- PyTorch, torchvision, ultralytics, Pillow, PyYAML (requirements.txt 참고)
