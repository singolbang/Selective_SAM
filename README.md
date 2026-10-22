# Selective SAM: RGB–Depth Fusion on DeLiVER

SAM 2.1 Tiny를 고정하고 depth adapter를 학습한 뒤, 입력에 따라 adapter 사용 여부를 결정하는 선택기를 학습했습니다. 선택기는 원본 SAM 마스크와 RGB·depth 정보를 사용합니다. DeLiVER 전방 시점의 box prompt 이진 분할을 대상으로 합니다.

## 준비

| 항목 | 다운로드 | 저장 위치 |
|---|---|---|
| SAM 2 소스 | [공식 저장소](https://github.com/facebookresearch/sam2) | `third_party/sam2/` |
| SAM 2.1 Tiny | [공식 checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | `checkpoints/sam2.1_hiera_tiny.pt` |
| DeLiVER | [공식 저장소](https://github.com/InSAI-Lab/DELIVER) · [데이터 다운로드](https://huggingface.co/datasets/InSAI-Lab/DELIVER) | `data/DELIVER/` |
| 학습한 adapter | [GitHub Release](https://github.com/singolbang/Selective_SAM/releases/download/v1.0.1/adapter_best.pt) | `checkpoints/adapter_best.pt` |
| 학습한 선택기 | [GitHub Release](https://github.com/singolbang/Selective_SAM/releases/download/v1.0.1/gate_best.pt) | `checkpoints/gate_best.pt` |

DeLiVER 아카이브에서는 `img`, `depth`, `semantic` 폴더를 사용합니다. box는 데이터셋에 포함된 별도 주석이 아닙니다. `selective_sam/data.py`의 `targets_from_labels()`가 semantic label에서 만들며, `selective_sam/manifest.py`로 목록을 생성할 수 있습니다.

가중치 SHA256은 adapter `b7d437c3383dcd8e872895eb10d2847e4524a9da2ce3ff3001dd92bd22dbe82b`, 선택기(`GateBaselineSAM_v1`) `e66a376b8570a198e540305e6efd1eef744e5b24c38718290e4a34cd9ee1df4a`입니다.

Python 3.10, PyTorch 2.5.1/CUDA 12.1 환경에서 실행했습니다. SAM 2 소스는 커밋 `2b90b9f5ceec907a1c18123530e92e794ad901a4`를 사용했습니다.

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
SAM2_BUILD_CUDA=0 pip install --no-deps -e third_party/sam2
```

## 결과

공식 train 3,983장으로 학습하고 validation 2,005장으로 checkpoint와 선택 임계값을 정했습니다. test 1,897장의 대상별 평균 IoU는 다음과 같습니다. 세 경로에 같은 이미지와 GT에서 만든 box를 사용했습니다.

| Test 입력 | 원본 SAM | 항상 융합 | 선택적 융합 |
|---|---:|---:|---:|
| 원본 depth | 0.7370 | 0.7907 | 0.7962 |
| 손상 depth | 0.7370 | 0.7663 | 0.7823 |

손상 입력은 이미지마다 depth에 배정한 잡음·픽셀 결측·영역 결측 중 한 조건입니다. 표의 IoU는 DeLiVER 공식 semantic mIoU와 다른 box prompt 기반 이진 분할 지표입니다.

## 추론

다운로드한 세 checkpoint와 저장소의 `checkpoints/selection_policy.json`을 함께 사용합니다. 직접 준비한 이미지에는 box 좌표를 입력합니다.

```bash
python -m selective_sam.infer \
  --rgb path/to/rgb.png --depth path/to/depth.png \
  --box 100 80 400 350 --mode selective --output results/prediction.png
```

`--mode`는 `baseline`, `fusion`, `selective` 중 하나입니다. DeLiVER test의 배정된 손상 입력을 재생하려면 manifest를 만든 뒤 `--split test --index 0 --target-index 0 --damaged`를 사용합니다.

## 다시 학습하기

`configs/final.yaml`과 `configs/gate_baseline.yaml`에 실험 설정이 있습니다. SAM은 두 학습 단계에서 고정됩니다. adapter와 선택기는 같은 공식 train을 순서대로 사용합니다.

```bash
python -m selective_sam.manifest
python -m selective_sam.train_adapter
python -m selective_sam.label_generation --split gate_train
python -m selective_sam.label_generation --split val
python -m selective_sam.train_gate_baseline
python -m selective_sam.label_generation --split test
python -m selective_sam.evaluate
```

adapter를 다시 학습했다면 기존 선택기 checkpoint와 정책 대신 새로 학습한 파일을 사용해야 합니다. 생성된 manifest, 라벨 cache, 결과와 학습 로그는 Git에서 제외됩니다.
