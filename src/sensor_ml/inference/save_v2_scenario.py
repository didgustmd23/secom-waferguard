# ==========================================
# V2 고정 센서 모델의 90%·80% OOF 목표 시나리오 보존
# - 90%는 기존 검증의 과거 학습 범위로 분류기만 재학습해 저장
# - 80%는 신뢰한 90% 묶음을 복사하고 문턱만 변경 (재학습 없음)
# - 원본 OOF 로그를 수정하지 않고 각 묶음에 사본을 보존
# - 검증된 미래 확률과 정합성을 확인하되 성능을 다시 튜닝하지 않음
# - 배포 승인이나 독립 최종 Test 완료로 표시하지 않음
# ==========================================
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.dataset_schema import split_frame_to_xy
from src.sensor_ml.experiments.fixed_sensor_compare import _json_value
from src.sensor_ml.experiments.fixed_sensor_oof import validate_candidate
from src.sensor_ml.inference.sensor_bundle import save_sensor_bundle, load_sensor_bundle, verify_sensor_bundle
from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_models import build_classifier, fit_pipeline, positive_scores
from src.sensor_inference import SensorInference
from src.split_contract import SOURCE_ROW_ID, temporal_folds, validate_train_role
from src.sensor_ml.verification.check_sensor_inference import compare_prediction_paths


def scenario_source(directory, target):
    """기존 OOF의 목표별 정확한 문턱과 선택 근거를 읽는다."""
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    if execution.get("oof_scope") != "after_selection_only" or execution.get("test_used") is not False:
        raise ValueError("선택 이후 OOF 개발 검증 로그만 사용할 수 있습니다.")
    read = lambda name: pd.read_csv(directory / f"{name}.csv", float_precision="round_trip")
    selected = read("scenario_results")
    selected = selected.loc[selected.target_recall.eq(target)]
    if len(selected) != 1:
        raise ValueError("요청한 Recall 목표의 시나리오가 하나만 있어야 합니다.")
    row = selected.iloc[0]
    candidates = read("threshold_compare")
    candidates = candidates[candidates.recall >= target].sort_values(
        ["alarm_ratio", "precision", "threshold"], ascending=[True, False, False])
    if candidates.empty or float(row.threshold) != float(candidates.iloc[0].threshold):
        raise ValueError("저장된 문턱이 과거 OOF 선택 기준과 일치하지 않습니다.")
    names = read("fixed_sensors").sort_values("sensor_order").feature.tolist()
    return float(row.threshold), names, execution, row.to_dict()


def retarget_inference(payload, names, threshold):
    """센서·통계·분류기는 그대로 복사하고 시나리오 문턱만 바꾼다."""
    original = payload["inference"]
    if list(original.sensors) != names:
        raise ValueError("기존 모델 묶음의 센서 목록·순서가 OOF 후보와 다릅니다.")
    if verify_sensor_bundle(payload)["status"] != "passed":
        raise ValueError("기존 모델 묶음의 복원 검증이 실패했습니다.")
    inference = deepcopy(original)
    inference.threshold = threshold
    inference._validate_contract()
    sample = payload["verification_input"]
    # 문턱만 다르므로 확률은 같아야 한다. 판정의 동일성은 요구하지 않는다.
    if not np.array_equal(original.predict(sample).positive_score,
                          inference.predict(sample).positive_score):
        raise ValueError("시나리오 변경으로 모델 확률이 바뀌었습니다.")
    return inference, sample


def train_inference(frame, config, names, threshold, *, n_estimators, n_jobs):
    """검증과 같은 마지막 과거 범위로 고정 센서 분류기를 학습한다."""
    validate_train_role(frame)
    features, labels, _ = split_frame_to_xy(frame, config.dataset)
    train_i, future_i = list(temporal_folds(frame, config.dataset, 3))[2]
    train_x = features.iloc[train_i].loc[:, names]
    classifier = build_classifier(config, "xgboost", n_estimators=n_estimators, n_jobs=n_jobs)
    classifier.set_params(max_depth=2, reg_lambda=1, class_weight_mode="ratio")
    template = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
        ("model", classifier),
    ])
    fitted = fit_pipeline(template, train_x, labels.iloc[train_i], config, "fixed_topk", n_jobs=n_jobs)
    inference = SensorInference(names, fitted.named_steps["imputer"].statistics_,
                                fitted.named_steps["model"], config.dataset.positive_label, threshold)
    valid = train_x.loc[~train_x.isna().all(axis=1)]
    checks = [compare_prediction_paths(fitted, inference, valid, case="reordered")]
    partial = valid.iloc[:32].fillna(0).copy()
    partial.iloc[:, 0] = np.nan
    checks.append(compare_prediction_paths(fitted, inference, partial, case="partial_nan"))
    if not all(check["passed"] for check in checks):
        raise ValueError("Pipeline과 저장용 고정 센서 추론 경로가 다릅니다.")
    future = features.iloc[future_i].loc[:, names]
    proof = pd.DataFrame({"source_row_id": frame[SOURCE_ROW_ID].iloc[future_i].to_numpy(),
                          "positive_score": positive_scores(fitted, future, config.dataset)})
    return inference, train_x, proof, checks


def main():
    """90% 저장 후 80%를 별도 묶음으로 보존하며 기존 경로는 덮어쓰지 않는다."""
    parser = argparse.ArgumentParser(description="V2 90%·80% 목표 시나리오 모델 저장")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, choices=(0.8, 0.9), required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--base-bundle-dir", type=Path)
    parser.add_argument("--trusted-local-bundle", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.bundle_dir.exists() and (not args.bundle_dir.is_dir() or any(args.bundle_dir.iterdir())):
        raise ValueError("비어 있는 새 모델 묶음 폴더를 지정하세요.")
    threshold, names, execution, scenario = scenario_source(args.oof_dir, args.target_recall)
    checks = []
    if args.target_recall == 0.9:
        if args.train is None or args.base_bundle_dir is not None:
            parser.error("90% 저장에는 --train이 필요하며 --base-bundle-dir는 사용하지 않습니다.")
        config = load_modeling_config(args.config)
        current = json.loads(json.dumps(asdict(config), default=_json_value))
        if execution["config"] != current:
            raise ValueError("OOF 실행 당시 설정과 현재 설정이 다릅니다.")
        frame = pd.read_csv(args.train)
        if args.train.resolve() != Path(execution["train_path"]).resolve():
            raise ValueError("OOF 기록과 같은 Train 파일을 지정하세요.")
        candidate_names, candidate = validate_candidate(frame, config, Path(execution["candidate_dir"]))
        if candidate_names != names:
            raise ValueError("OOF 목록과 최초 고정 후보의 센서 순서가 다릅니다.")
        inference, samples, proof, checks = train_inference(
            frame, config, names, threshold, n_estimators=candidate["n_estimators"], n_jobs=args.n_jobs)
        saved = pd.read_csv(args.oof_dir / "future_predictions.csv", float_precision="round_trip")
        paired = proof.merge(saved[["source_row_id", "positive_score"]], on="source_row_id",
                             suffixes=("_new", "_saved"), validate="one_to_one")
        if (len(paired) != len(proof) or len(paired) != len(saved)
                or not np.allclose(paired.positive_score_new, paired.positive_score_saved, rtol=0, atol=1e-7)):
            raise ValueError("새 학습 확률이 기존 OOF 검증 모델과 다릅니다. 저장을 중단합니다.")
        trained = True
    else:
        if args.base_bundle_dir is None or args.train is not None:
            parser.error("80% 저장에는 --base-bundle-dir를 지정하고 --train은 생략하세요.")
        payload = load_sensor_bundle(args.base_bundle_dir, trusted=args.trusted_local_bundle)
        base = json.loads((args.base_bundle_dir / "manifest.json").read_text(encoding="utf-8"))["provenance"]
        if base.get("target_recall") != 0.9 or base.get("oof_execution") != execution:
            raise ValueError("같은 OOF 기록에서 저장한 V2 90% 묶음이 필요합니다.")
        inference, samples = retarget_inference(payload, names, threshold)
        trained = False
    provenance = {"version": "v2", "scenario": f"recall{int(args.target_recall * 100)}",
                  "target_recall": args.target_recall, "source_oof_dir": str(args.oof_dir.resolve()),
                  "oof_execution": execution, "development_scenario_result": scenario,
                  "trained_this_run": trained, "training_scope": "third_outer_train_only",
                  "base_bundle_dir": str(args.base_bundle_dir.resolve()) if args.base_bundle_dir else None,
                  "inference_checks": checks, "is_performance_evaluation": False,
                  "test_used": False, "deployment_approved": False}
    save_sensor_bundle(inference, samples, args.bundle_dir, provenance=provenance)
    # 원본 로그는 그대로 두고 모든 CSV와 실행 기록을 각 시나리오 폴더에 복사한다.
    archive = args.bundle_dir / "experiment"
    archive.mkdir()
    for source in args.oof_dir.iterdir():
        if source.is_file() and (source.suffix == ".csv" or source.name == "execution.json"):
            shutil.copy2(source, archive / source.name)
    payload = load_sensor_bundle(args.bundle_dir, trusted=True)
    verification = verify_sensor_bundle(payload)
    (args.bundle_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    if verification["status"] != "passed":
        raise ValueError("저장·복원 정합성이 실패했습니다. 해당 묶음을 사용하지 마세요.")
    print(f"V2 Recall 목표 {args.target_recall:.0%} 묶음: {args.bundle_dir}")
    print(f"정확한 OOF 문턱: {threshold!r}, 이번 실행 재학습: {trained}")
    print("연구용 위험 선별 시나리오입니다. 목표 검출률 보장·출하 승인·독립 최종 Test를 뜻하지 않습니다.")


if __name__ == "__main__":
    main()
