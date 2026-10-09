# ==========================================
# 저장된 후보 모델로 센서 CSV 예측
# - 현재 config·Dataset Profile을 읽거나 모델을 재학습하지 않음
# - 센서명·개수·값을 저장된 추론 계약에 맞춰 검증
# - 전체 결측·누락·추가 센서는 임의로 보정하거나 제외하지 않음
# - 결과는 입력 행 순서대로 저장하며 기존 파일 덮어쓰기 금지
# - 데모 입력 내보내기는 저장된 검증 행의 복사이며 신규 실측 데이터가 아님
# ==========================================

import argparse
import csv
from pathlib import Path

import pandas as pd

from src.sensor_ml.inference.sensor_bundle import load_sensor_bundle


def read_sensor_csv(path):
    """중복 헤더·잘못된 행 길이를 차단한 후 센서 CSV를 읽는다."""
    # pandas는 중복 헤더를 자동 변경하므로 원본 헤더를 먼저 검사해야 한다.
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
            if not header or any(not name for name in header) or len(set(header)) != len(header):
                raise ValueError("입력 CSV 헤더는 비어 있지 않은 중복 없는 센서명이어야 합니다.")
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"입력 CSV {reader.line_num}번째 줄의 값 개수가 헤더와 다릅니다.")
        except StopIteration as error:
            raise ValueError("입력 CSV가 비어 있습니다.") from error
        except csv.Error as error:
            raise ValueError("입력 CSV 형식이 잘못되었습니다.") from error
    # 빈 값·NaN은 pandas 결측으로 읽고 숫자가 아닌 값은 추론 코어가 거부한다.
    return pd.read_csv(path, encoding="utf-8-sig")


def write_new_csv(frame, path):
    """이미 존재하는 파일은 열지 않고 새 CSV만 저장한다."""
    target = Path(path)
    if target.exists():
        raise ValueError(f"출력 파일이 이미 있습니다. 새 경로를 지정하세요: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # 검증 이후에도 파일이 생길 수 있으므로 배타적 생성 모드로 덮어쓰기를 차단한다.
    with target.open("x", encoding="utf-8-sig", newline="") as stream:
        frame.to_csv(stream, index=False)


def predict_sensor_file(bundle_dir, input_path, output_path, *, trusted=False):
    """신뢰한 후보 묶음을 복원해 신규 센서 파일을 예측하고 결과를 저장한다."""
    if Path(output_path).exists():
        raise ValueError("예측 출력 파일이 이미 있습니다. 새 경로를 지정하세요.")
    payload = load_sensor_bundle(bundle_dir, trusted=trusted)
    frame = read_sensor_csv(input_path)
    # 일부 행이 잘못되어도 조용히 건너뛰지 않는다. 전체 입력 검증 후에만 결과를 저장한다.
    result = payload["inference"].predict(frame)
    result.insert(0, "input_row_index", range(len(result)))
    write_new_csv(result, output_path)
    return result


def main():
    """신규 입력 예측 또는 저장된 검증 입력의 명시적 데모 내보내기를 수행한다."""
    parser = argparse.ArgumentParser(description="후보 모델의 고정 센서 CSV 예측 (재학습 없음)")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="고정 센서만 포함한 신규 입력 CSV")
    source.add_argument("--export-demo-input", type=Path,
                        help="묶음에 저장된 검증 입력을 데모 CSV로 내보냅니다 (신규 데이터 아님)")
    parser.add_argument("--output", type=Path, help="예측 결과를 저장할 새 CSV 경로")
    parser.add_argument("--trusted-local-bundle", action="store_true",
                        help="직접 생성한 신뢰한 로컬 joblib 파일임을 확인합니다.")
    args = parser.parse_args()
    if args.export_demo_input is not None:
        if args.output is not None:
            parser.error("데모 입력 내보내기에는 --output을 함께 지정하지 마세요.")
        payload = load_sensor_bundle(args.bundle_dir, trusted=args.trusted_local_bundle)
        write_new_csv(payload["verification_input"], args.export_demo_input)
        print(f"저장된 검증 입력의 데모 복사본: {args.export_demo_input}")
        print("Train 유래 검증 입력입니다. 신규 데이터 성능 평가에 사용하지 마세요.")
        return
    if args.output is None:
        parser.error("신규 입력 예측에는 --output이 필요합니다.")
    result = predict_sensor_file(args.bundle_dir, args.input, args.output,
                                 trusted=args.trusted_local_bundle)
    print(f"예측 행 수: {len(result)}, 위험 대상 선별 행 수: {int(result.predicted_positive.sum())}")
    print(f"예측 결과: {args.output}")
    print("후보 문턱에 따른 위험 대상 선별입니다. 실제 불량 확정·출하 승인·성능 평가·배포 승인을 뜻하지 않습니다.")


if __name__ == "__main__":
    main()
