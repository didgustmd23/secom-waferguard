# ==========================================
# 전체 입력 Pipeline과 축소 센서 추론의 정합성 확인
# - 저장된 단일 OOF 설정·행·정책 문턱을 확인한 뒤 Train만 한 번 학습
# - 같은 분류기를 두 입력 경로로 실행하여 확률·판정·label을 비교
# - 컬럼 순서 변경·일부 NaN·단일 행 입력도 확인
# - Validation/Test 평가나 최종 모델 확정은 수행하지 않음
# - 명시한 경우에만 정합성을 통과한 후보 추론 묶음을 별도 저장
# ==========================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_models import fit_pipeline
    from src.sensor_inference import build_sensor_inference
    from src.top20_time_validation import prepare_candidate
except ModuleNotFoundError:
    from dataset_schema import split_frame_to_xy
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_models import fit_pipeline
    from sensor_inference import build_sensor_inference
    from top20_time_validation import prepare_candidate


def compare_prediction_paths(pipeline, inference, full, *, case, atol=1e-12):
    """동일한 행의 원본 Pipeline 확률과 축소 추론 결과를 비교한다."""
    if full.empty:
        raise ValueError("정합성 확인에 한 행 이상 필요합니다.")
    classes = list(pipeline.named_steps["model"].classes_)
    positive_index = classes.index(inference.positive_label)
    # 전체 경로에는 원본 센서를, 축소 경로에는 역순의 선택 센서만 전달한다.
    # 이 비교에는 label·timestamp를 포함하지 않으므로 추론 계약도 함께 확인된다.
    expected = pipeline.predict_proba(full)[:, positive_index]
    actual = inference.predict(full.loc[:, list(reversed(inference.sensors))])
    positive = expected >= inference.threshold
    labels = np.where(positive, inference.positive_label, classes[1 - positive_index])
    probability_match = bool(np.isfinite(expected).all()
                             and np.isfinite(actual.positive_score).all()
                             and np.allclose(expected, actual.positive_score, rtol=0, atol=atol))
    decision_match = bool(np.array_equal(positive, actual.predicted_positive))
    label_match = bool(np.array_equal(labels, actual.predicted_label))
    return {"case": case, "rows": len(full), "absolute_tolerance": atol,
            "max_probability_difference": float(np.max(np.abs(expected - actual.positive_score))),
            "probability_match": probability_match, "decision_match": decision_match,
            "label_match": label_match,
            "passed": probability_match and decision_match and label_match}


def check_inference_paths(pipeline, features, *, positive_label, threshold, max_sensors):
    """학습된 모델의 입력 경로를 확인하고 결과와 후보 센서 목록을 반환한다."""
    inference = build_sensor_inference(pipeline, positive_label=positive_label,
                                        threshold=threshold, max_sensors=max_sensors)
    # 모든 선택 센서가 NaN인 행은 추론 계약상 거부된다. 조용히 제외하지 않고 수를 기록한다.
    usable = ~features.loc[:, list(inference.sensors)].isna().all(axis=1)
    observed = features.loc[usable].copy()
    checks = [compare_prediction_paths(pipeline, inference, observed, case="original_reordered")]
    # 결측 보정 시험용 복사본만 수정한다. 원본 Train과 학습 통계는 그대로 보존한다.
    if len(inference.sensors) > 1:
        partial = observed.copy()
        partial.loc[:, list(inference.sensors)] = partial.loc[:, list(inference.sensors)].fillna(0.0)
        # 마지막 센서는 관측값을 유지하여 모든 센서 결측 거부 규칙과 구별한다.
        partial.loc[:, inference.sensors[0]] = np.nan
        checks.append(compare_prediction_paths(pipeline, inference, partial, case="partial_nan"))
    single = observed.iloc[[0]]
    checks.append(compare_prediction_paths(pipeline, inference, single, case="single_row"))
    # 같은 행을 배치와 단독으로 처리할 때도 동일한 학습 통계가 적용되어야 한다.
    reduced = observed.loc[:, list(inference.sensors)]
    batch_score = inference.predict(reduced).positive_score.iloc[0]
    single_score = inference.predict(reduced.iloc[[0]]).positive_score.iloc[0]
    checks.append({"case": "batch_independence", "rows": 1,
                   "passed": bool(np.isclose(batch_score, single_score, rtol=0, atol=1e-12)),
                   "max_probability_difference": float(abs(batch_score - single_score))})
    # 전체 결측 행은 확률을 만드는 대신 반드시 명시적인 오류를 반환해야 한다.
    try:
        inference.predict(reduced.iloc[[0]] * np.nan)
    except ValueError as error:
        rejected = "모든 필수 센서가 결측" in str(error)
    else:
        rejected = False
    checks.append({"case": "all_missing_rejected", "rows": 1, "passed": rejected})
    summary = {"status": "passed" if all(row["passed"] for row in checks) else "failed",
               "selected_sensors": list(inference.sensors), "selected_sensor_count": len(inference.sensors),
               "input_rows": len(features), "compared_rows": len(observed),
               "all_missing_rejected_rows": int((~usable).sum()), "checks": checks,
               "is_performance_evaluation": False, "is_final_model": False,
               "validation_used": False, "test_used": False}
    return summary


def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="선택 센서 추론 정합성 확인 (Train만 사용)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--bundle-dir", type=Path,
                        help="정합성 통과 시 후보 모델을 저장할 새 폴더 (모듈 실행 필요)")
    args = parser.parse_args(argv)
    if args.bundle_dir is not None:
        if __package__ != "src":
            raise ValueError("후보 저장은 python -m src.check_sensor_inference로 실행하세요.")
        if args.bundle_dir.exists() and any(args.bundle_dir.iterdir()):
            raise ValueError("모델 묶음 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
        if args.bundle_dir.resolve() == args.output_dir.resolve():
            raise ValueError("확인 보고서 폴더와 모델 묶음 폴더는 분리하세요.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 출력 폴더를 지정하세요.")
    if args.n_jobs < 1:
        raise ValueError("병렬 작업 수는 1 이상의 정수여야 합니다.")
    return args


def save_results(args, summary):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inference_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # 확률 비교는 구조 변환 확인이며 학습 행의 점수를 성능으로 보고하지 않는다.
    report = ["# 선택 센서 추론 정합성 확인", "",
              f"- 확인 상태: `{summary['status']}`",
              f"- 후보 센서 수: {summary['selected_sensor_count']}",
              f"- 비교 행 수: {summary['compared_rows']} / {summary['input_rows']}",
              f"- 선택 센서 전체 결측으로 거부한 원본 행 수: {summary['all_missing_rejected_rows']}",
              f"- OOF 후보 threshold: `{args.threshold!r}`", "",
              "## 확인 항목", ""]
    for check in summary["checks"]:
        report.append(f"- `{check['case']}`: {'일치/정상' if check['passed'] else '실패'}")
    report.extend(["", "## 해석 범위", "",
                   "동일한 학습 모델의 전체 입력·축소 입력 경로를 비교한 구현 검증입니다.",
                   "Train으로 한 번 학습했으며 Validation·Test·OOF 재학습·최종 모델 저장은 수행하지 않았습니다.",
                   "센서 목록과 문턱은 후보입니다. 정합성 통과는 성능 목표 달성이나 배포 승인을 뜻하지 않습니다.",
                   "전체 결측 행은 축소 추론에서 거부하므로 원본 Pipeline의 median 대치 예측과 동일하게 처리하지 않습니다.",
                   "정책·설정·Train ID 확인은 센서 값이나 소스 파일 내용의 동일성을 보증하지 않습니다."])
    (args.output_dir / "inference_check.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    train = pd.read_csv(args.train)
    record = json.loads((args.oof_dir / "oof_run.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(args.oof_dir / "oof_predictions.csv", float_precision="round_trip")
    # 기존 검증기를 재사용하여 다른 설정·Train·문턱을 임의로 연결하지 않는다.
    experiment, template, _ = prepare_candidate(train, oof, record, config,
                                                threshold=args.threshold, n_jobs=args.n_jobs)
    features, labels, _ = split_frame_to_xy(train, config.dataset)
    fitted = fit_pipeline(template, features, labels, config, experiment, n_jobs=args.n_jobs)
    summary = check_inference_paths(fitted, features, positive_label=config.dataset.positive_label,
                                     threshold=args.threshold,
                                     max_sensors=config.reduction_policy.max_sensor_count)
    summary.update({"experiment": experiment, "threshold": args.threshold,
                    "threshold_source": "train_single_oof_policy_candidate",
                    "train_path": str(args.train.resolve()), "oof_dir": str(args.oof_dir.resolve()),
                    "training_protocol_id": record["training_protocol_id"],
                    "model_parameters": fitted.named_steps["model"].get_params(),
                    "rf_selector_parameters": fitted.named_steps["selector"].estimator.get_params()})
    save_results(args, summary)
    print(pd.DataFrame(summary["checks"]).to_string(index=False))
    print(f"후보 센서 수: {summary['selected_sensor_count']}, 확인 상태: {summary['status']}")
    print(f"구현 정합성 보고서: {args.output_dir / 'inference_check.md'}")
    if summary["status"] != "passed":
        raise ValueError("추론 경로 정합성 확인에 실패했습니다. 최종 모델 준비 전에 차이를 확인하세요.")
    if args.bundle_dir is not None:
        from src.sensor_bundle import save_sensor_bundle
        # 방금 비교에 사용한 동일 학습 모델에서 추론 객체를 가져오며 다시 fit하지 않는다.
        inference = build_sensor_inference(fitted, positive_label=config.dataset.positive_label,
                                            threshold=args.threshold,
                                            max_sensors=config.reduction_policy.max_sensor_count)
        save_sensor_bundle(inference, features, args.bundle_dir, provenance=summary)
        print(f"후보 추론 묶음: {args.bundle_dir} (최종 모델 아님)")


if __name__ == "__main__":
    main()
