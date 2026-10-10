# WM-811K 입력 캐시와 PyTorch Dataset

## 현재 범위

`src/wafer_dl/cache.py`는 확정 분할과 변환 검사를 확인한 뒤 숫자 전용 uint8 NPY 캐시를 생성합니다. `src/wafer_dl/dataset.py`는 선택 구간의 캐시를 읽어 PyTorch 입력을 구성합니다. 구현·합성 테스트에 이어 사용자가 실제 Train·Validation 첫 배치 검사를 완료했습니다. 모델 학습은 하지 않았습니다.

## 처리 흐름과 분할 파일

```text
원본 pickle → 숫자 전용 canonical → 검증·EDA → 그룹 분할
    → Train/Validation 맵 변환·형태 확인 → uint8 캐시
    → Dataset 2채널 Tensor → 첫 배치 검사 (현재 완료)
    → 모델 학습·Validation 비교 → 최종 Test 평가 (이후 단계)
```

Train 약 70%·Validation 약 15%·Test 약 15%의 분할은 이미 확정했습니다. Test가 없는 것이 아니라 Test용 입력 캐시 준비를 미룬 상태입니다. 원본 맵을 세 폴더에 복사해서 나누는 대신, manifest에 각 행의 소속을 기록합니다.

| 파일·컬럼 | 역할 |
| --- | --- |
| 원본 canonical NPZ | 실제 웨이퍼 맵 보관 |
| `split_summary.csv` | 구간별 클래스 표본 수·비율 요약 |
| `split_manifest.jsonl` | 각 행의 그룹과 최종 분할 소속 기록 |
| `split_group_id` | Lot·동일 맵 관계로 함께 묶인 그룹의 식별자 |
| `group_split` | 연결 그룹이 배정된 train·validation·test. 배정되지 않은 그룹은 null |
| `split` | 행의 최종 용도: train·validation·test·unlabeled·quarantined |

같은 그룹의 미라벨 행은 `group_split`이 있어도 `split=unlabeled`이므로 지도학습 캐시에 넣지 않습니다. 충돌로 격리한 행도 캐시에서 제외합니다. Dataset이나 DataLoader에서 데이터를 다시 무작위 분할하지 않습니다.

## 실행 확인 기록 — 2026-10-10

아래는 사용자가 공유한 실행 출력 기준이며 이 기록을 위해 실제 데이터를 재실행하지 않았습니다.

| 구간 | 상태 | 이미지 형태 | 이미지 자료형 | 라벨 자료형 | Test 맵 접근 |
| --- | --- | --- | --- | --- | --- |
| Train | `dataset_batch_passed` | `[32,2,128,128]` | `torch.float32` | `torch.int64` | 없음 |
| Validation | `dataset_batch_passed` | `[32,2,128,128]` | `torch.float32` | `torch.int64` | 없음 |

두 실행의 분할 프로토콜은 동일합니다.

```text
wm_split_d02d993fd73c2cc4bf496d54774401c60a94b0d1737c7c135e7ae4837a6b85d6
```

이 결과는 선택 구간의 캐시 정합성과 첫 배치 입력 계약 확인입니다. 전체 Epoch 실행·GPU 학습·성능 평가 완료를 뜻하지 않습니다. 이후 명령은 재현·전달용이며 이미 완료한 캐시를 다시 생성할 필요는 없습니다.

프로젝트 루트에서 `ml310` 등 PyTorch가 설치된 환경을 사용하세요. NVIDIA CUDA 12.8용 선택 의존성은 다음과 같습니다.

```powershell
python -m pip install -r requirements-wafer-dl.txt --index-url https://download.pytorch.org/whl/cu128
```

이미 설치한 환경에서는 재설치하지 않아도 됩니다. CPU·다른 CUDA 환경은 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)에 맞는 설치 경로를 선택하세요.

## 1. 개발 구간 캐시 생성

변환 검사 결과가 `transform_check_passed`이고 `examples.png`에서 패턴 보존을 직접 확인한 경우에만 `--confirm-visual-review`를 지정합니다. 이 옵션은 자동 시각 검사를 뜻하지 않습니다.

```powershell
python -m src.wafer_dl.cache `
  --split-dir data/splits/wm811k/group_v1 `
  --check-dir reports/wm811k/transform_v1 `
  --output-dir data/processed/wm811k/input_cache/wm_resize_v1 `
  --confirm-visual-review
```

실제 변환 검사 결과 폴더가 다르면 `--check-dir`을 변경하세요. canonical 경로는 분할 보고서에서 읽습니다. 데이터 묶음을 이동했다면 같은 원본 묶음의 현재 위치를 `--canonical-dir`로 지정할 수 있습니다.

완료 기준은 `cache_complete`와 출력 폴더의 `SUCCESS.json`입니다. 기존 출력 폴더는 덮어쓰지 않습니다. 중단된 폴더는 완료 결과로 사용하지 말고, 원인을 확인한 후 다른 새 출력 경로로 실행하세요.

- 기본 대상은 Train·Validation입니다. Test 배열은 읽지 않습니다.
- 미라벨·격리 행은 학습 캐시에 포함하지 않습니다.
- 원본 맵·라벨·분할을 변경하거나 다시 나누지 않습니다.
- 선택한 배열의 무결성을 확인하고 기존 결정적 resize·padding을 재사용합니다.
- 유효 영역 또는 불량 영역이 완전히 사라지면 중단합니다. 일부 형태 변화까지 없다는 보장은 아닙니다.

## 2. 첫 배치 확인

캐시 생성 성공 후 각각 실행합니다.

```powershell
python -m src.wafer_dl.dataset `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v1 `
  --split train `
  --batch-size 32

python -m src.wafer_dl.dataset `
  --cache-dir data/processed/wm811k/input_cache/wm_resize_v1 `
  --split validation `
  --batch-size 32
```

기대 상태는 `dataset_batch_passed`입니다. 기본 크기에서 이미지 형태는 `[32,2,128,128]`, 이미지 자료형은 `torch.float32`, 라벨은 `torch.int64`입니다. 마지막 배치 등 표본 수가 적으면 배치 수는 줄어들 수 있습니다.

초기화 시 선택 구간의 전체 shard 해시·배열 헤더·색인을 검사합니다. 따라서 첫 출력까지 파일 크기에 비례한 시간이 걸릴 수 있습니다. 전체 데이터를 RAM에 적재하지는 않습니다. 채널값 검사는 첫 배치 및 이후 실제 접근한 표본에 적용합니다. 다른 분할 파일은 열지 않습니다.

이 검사는 CPU·`num_workers=0`으로 실행합니다. GPU 연산·다중 worker 처리량·전체 Epoch·모델 성능 검사가 아닙니다. 캐시의 `training_ready=false`를 자동 변경하지 않습니다.

## 저장 계약과 학습 연결

캐시 한 표본은 `[128,128]` uint8 범주형 맵입니다. 상태는 `0=외부`, `1=정상 다이`, `2=불량 다이`입니다. 웨이퍼 전체의 9개 패턴 라벨과 픽셀 상태는 별개입니다.

`index.jsonl`은 표본 ID·클래스·shard 파일·파일 내부 위치를 연결합니다. `cache.json`은 설정·분할 프로토콜·개수·파일 해시를 기록하고 `cache.md`는 사람이 읽는 요약입니다. 해시는 자동 기록하며 안전성 인증이나 사용자 수동 해시 관리 절차가 아닙니다.

```python
from torch.utils.data import DataLoader
from src.wafer_dl.dataset import WaferDataset

dataset = WaferDataset("data/processed/wm811k/input_cache/wm_resize_v1", split="train")
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
try:
    batch = next(iter(loader))
    images = batch["image"]   # [B,2,H,W]: defect, valid 순서
    targets = batch["target"] # [B]: 정수 0~8
    # sample_id는 추적용이며 모델 입력에 포함하지 않습니다.
finally:
    dataset.close()
```

Dataset은 읽기 전용 mmap을 최대 두 파일까지 유지하고 필요한 표본만 두 채널로 변환합니다. 캐시 맵은 한 장 16KiB, 모델용 두 float32 채널은 한 장 128KiB입니다. 현재 개발 구간 147,088장 기준 맵 본체는 약 2.41GB이며 색인·헤더 공간이 추가됩니다.

`configs/wm811k/cache.json`의 `shard_max_rows` 기본값은 4,096, `buffer_max_bytes`는 64MiB입니다. 두 상한 중 작은 기준으로 나눕니다. Train·Validation writer 버퍼는 각각 할당되므로 버퍼 합계는 기본 최대 약 128MiB이고, 검사·변환의 추가 메모리는 별도입니다.

`suggested_train_class_weights`는 Train 빈도만 사용한 `N_train/(9×클래스 표본 수)` 후보입니다. 손실 함수에 자동 적용하지 않으며 가중치 채택은 학습 실험에서 결정합니다. 증강·추가 정규화·통계 재학습도 하지 않습니다.

## Test는 이후 별도 준비

현재 Test 캐시는 준비하지 않았습니다. 아래는 이후 별도 준비용 명령입니다. 동일하게 확정된 변환 규칙으로 Test 캐시를 미리 만들어 두는 것도 가능하며, 캐시 생성 자체는 모델 학습·성능 평가가 아닙니다. 다만 Test 결과를 보고 전처리·모델·문턱을 조정하지 않습니다. Test 표본의 시각·통계 분석을 추가로 수행하면 해당 사용 이력을 기록합니다.

```powershell
python -m src.wafer_dl.cache `
  --split-dir data/splits/wm811k/group_v1 `
  --check-dir reports/wm811k/transform_v1 `
  --output-dir data/processed/wm811k/input_cache/wm_resize_v1_test `
  --splits test `
  --confirm-visual-review `
  --confirm-test-preparation
```

Dataset에서 Test를 여는 경우에도 Python의 `allow_test=True` 또는 CLI의 `--confirm-test-preparation`이 필요합니다. 승인 옵션은 실수 방지 장치이지 운영체제 보안 경계가 아닙니다. Test 준비 중 변환 소실이 발생하면 자동 보정하지 않고 중단하며 별도 검토합니다.

## 합성 테스트

```powershell
python -m unittest tests.test_wafer_cache_dataset tests.test_wafer_transform tests.test_wafer_split -v
```

실제 데이터·외부 pickle·모델 학습 없이 저장·복원 정합성, Test 읽기 제한, 승인·덮어쓰기 거부, 손상·색인 중복 거부, Tensor 독립성과 worker 전달 상태를 검사합니다. 다중 worker 실제 실행은 아직 검증 범위가 아닙니다.

상세 캐시·색인·표본·집계·경로는 로컬 산출물입니다. 향후 전처리 에이전트의 외부 LLM·LangSmith 입력으로 그대로 전달하면 안 됩니다.
