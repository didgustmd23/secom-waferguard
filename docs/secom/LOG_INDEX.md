# SECOM 로그 인덱스

전체 로그는 로컬에서 보존하며 GitHub에는 올리지 않는다. 공개 집계 결과는 [결과 요약](../../reports/README.md#secom-공개-결과-요약)을 참고한다. 아래 경로는 로컬 원본을 찾기 위한 기록이다.

## 분류와 보존 원칙

| 위치 | 내용 |
| --- | --- |
| `logs/secom/data/` | 병합·품질 점검·split 로그 |
| `logs/secom/exploratory/` | 초기 탐색·교정된 평가·이상 탐지·과거 정책 비교 |
| `logs/secom/exploratory/legacy/` | 과거 루트에 있던 단일 CSV와 초기 스크립트 기본 출력 |
| `logs/secom/v1/` | S0~S3·M0~M3·시간 검증·drift·최근 학습·V1 후속 평가 |
| `logs/secom/v2/` | V2 가중치·결측·센서 선택·고정 센서 초기 대조·진단 |
| `logs/secom/verification/` | 저장·추론·실행 범위 구현 검증 |
| `logs/secom/demos/` | Train 유래 V1/V2 추론 데모 출력 |

현재 모델 묶음이 출처로 참조하는 아래 두 폴더는 **원래 경로에 유지**한다. 90%/80% 묶음 manifest와 원본 OOF 기록을 임의로 수정하지 않는다.

- 고정 센서 선택 후보 (로컬: `logs/v2_fixed_sensor20_later_selection/`)
- 선택 이후 OOF·문턱·개발 평가 (로컬: `logs/v2_fixed_sensor20_oof/`)

CSV 수치·실행 JSON·모델 객체를 변경하거나 학습을 재실행하지 않았다. 과거 실행 JSON·모델 manifest·노트북 셀 출력의 이전 절대 경로는 당시 기록이다. 문서의 현재 링크는 이동 후 경로를 가리킨다. 과거 기록에 있는 경로를 찾을 때 아래 이동표를 사용한다. 실행 기록을 자동 재작성하거나 임의로 복구했다고 주장하지 않는다.

분석 노트북은 [reports/secom/analysis.ipynb](../../reports/secom/analysis.ipynb), 시각화는 [결과물 목록](../../reports/README.md#secom-노트북과-그림)을 따른다. 새 실행에서는 비어 있는 새 로그 폴더를 명시하고 해당 역할 아래 생성한다. 기존 결과를 덮어쓰지 않는다.

`logs/`는 Git에서 제외되는 로컬 산출물이다. 저장소를 새로 복제하면 실제 로그가 없을 수 있으며 이 문서는 분류·경로 안내다.

## 이동 전후 경로

| 이전 경로 (프로젝트 루트 기준) | 현재 위치 |
| --- | --- |
| `logs/anomaly_20261006` | `logs/secom/exploratory/anomaly_20261006` |
| `logs/baseline_random_result.csv` | `logs/secom/exploratory/legacy/baseline_random_result.csv` |
| `logs/baseline_time_result.csv` | `logs/secom/exploratory/legacy/baseline_time_result.csv` |
| `logs/dataset_log.csv` | `logs/secom/data/dataset_log.csv` |
| `logs/evaluation_scope_check` | `logs/secom/verification/evaluation_scope_check` |
| `logs/feature_compare.csv` | `logs/secom/exploratory/legacy/feature_compare.csv` |
| `logs/feature_time_20261006` | `logs/secom/exploratory/feature_time_20261006` |
| `logs/integrated` | `logs/secom/data/integrated` |
| `logs/lightgbm_oof_predictions.csv` | `logs/secom/exploratory/legacy/lightgbm_oof_predictions.csv` |
| `logs/m1_s0_top20_cv` | `logs/secom/v1/m1_s0_top20_cv` |
| `logs/m1_s0_top20_time` | `logs/secom/v1/m1_s0_top20_time` |
| `logs/m2_s0_top20_cv` | `logs/secom/v1/m2_s0_top20_cv` |
| `logs/m2_s0_top20_time` | `logs/secom/v1/m2_s0_top20_time` |
| `logs/m3_all_vs_top20_time` | `logs/secom/v1/m3_all_vs_top20_time` |
| `logs/m3_fixed_sensor_control` | `logs/secom/v1/m3_fixed_sensor_control` |
| `logs/m3_frozen_time_evaluation` | `logs/secom/v1/m3_frozen_time_evaluation` |
| `logs/m3_m0_oof_compare` | `logs/secom/v1/m3_m0_oof_compare` |
| `logs/m3_policy80_40_diagnostics` | `logs/secom/v1/m3_policy80_40_diagnostics` |
| `logs/m3_policy80_40_time_validation` | `logs/secom/v1/m3_policy80_40_time_validation` |
| `logs/m3_recent_sensor_overlap` | `logs/secom/v1/m3_recent_sensor_overlap` |
| `logs/m3_recent_window_compare` | `logs/secom/v1/m3_recent_window_compare` |
| `logs/m3_recent_window_diagnostics` | `logs/secom/v1/m3_recent_window_diagnostics` |
| `logs/m3_s0_top20_cv` | `logs/secom/v1/m3_s0_top20_cv` |
| `logs/m3_s0_top20_time` | `logs/secom/v1/m3_s0_top20_time` |
| `logs/m3_selected_sensor_drift` | `logs/secom/v1/m3_selected_sensor_drift` |
| `logs/m3_sensor_bundle_check` | `logs/secom/verification/m3_sensor_bundle_check` |
| `logs/m3_sensor_detail` | `logs/secom/v1/m3_sensor_detail` |
| `logs/m3_sensor_inference_check` | `logs/secom/verification/m3_sensor_inference_check` |
| `logs/model_compare.csv` | `logs/secom/exploratory/legacy/model_compare.csv` |
| `logs/policy_compare` | `logs/secom/exploratory/policy_compare` |
| `logs/quality_filtered_20261006` | `logs/secom/exploratory/quality_filtered_20261006` |
| `logs/s0_rf_top20_cv` | `logs/secom/v1/s0_rf_top20_cv` |
| `logs/s0_rf_top20_time` | `logs/secom/v1/s0_rf_top20_time` |
| `logs/s1_rf_top20_cv` | `logs/secom/v1/s1_rf_top20_cv` |
| `logs/s1_rf_top20_time` | `logs/secom/v1/s1_rf_top20_time` |
| `logs/s2_rf_top20_cv` | `logs/secom/v1/s2_rf_top20_cv` |
| `logs/s2_rf_top20_time` | `logs/secom/v1/s2_rf_top20_time` |
| `logs/s2_rf_top20_time_user` | `logs/secom/v1/s2_rf_top20_time_user` |
| `logs/s3_rf_top20_cv` | `logs/secom/v1/s3_rf_top20_cv` |
| `logs/s3_rf_top20_time` | `logs/secom/v1/s3_rf_top20_time` |
| `logs/sensor20_demo_predictions.csv` | `logs/secom/demos/v1/sensor20_demo_predictions.csv` |
| `logs/split_summary.csv` | `logs/secom/data/split_summary.csv` |
| `logs/step3_integrated` | `logs/secom/data/step3_integrated` |
| `logs/temporal_drift_label_summary.csv` | `logs/secom/exploratory/legacy/temporal_drift_label_summary.csv` |
| `logs/temporal_drift_report.csv` | `logs/secom/exploratory/legacy/temporal_drift_report.csv` |
| `logs/temporal_quality_filtered_20261006` | `logs/secom/exploratory/temporal_quality_filtered_20261006` |
| `logs/temporal_quality_filtered_20261006_final` | `logs/secom/exploratory/temporal_quality_filtered_20261006_final` |
| `logs/threshold_compare.csv` | `logs/secom/exploratory/legacy/threshold_compare.csv` |
| `logs/time_validation_compare.csv` | `logs/secom/exploratory/legacy/time_validation_compare.csv` |
| `logs/time_validation_error_cases.csv` | `logs/secom/exploratory/legacy/time_validation_error_cases.csv` |
| `logs/time_validation_error_features.csv` | `logs/secom/exploratory/legacy/time_validation_error_features.csv` |
| `logs/time_validation_error_summary.csv` | `logs/secom/exploratory/legacy/time_validation_error_summary.csv` |
| `logs/v2_fixed_sensor20_compare` | `logs/secom/v2/v2_fixed_sensor20_compare` |
| `logs/v2_missing_p0_p1` | `logs/secom/v2/v2_missing_p0_p1` |
| `logs/v2_missing_p0_p2` | `logs/secom/v2/v2_missing_p0_p2` |
| `logs/v2_p0_all_vs_gain_top20` | `logs/secom/v2/v2_p0_all_vs_gain_top20` |
| `logs/v2_p0_all_vs_rf_top20` | `logs/secom/v2/v2_p0_all_vs_rf_top20` |
| `logs/v2_p0_all_vs_weighted_gain_top20` | `logs/secom/v2/v2_p0_all_vs_weighted_gain_top20` |
| `logs/v2_p0_stable_gain_top20` | `logs/secom/v2/v2_p0_stable_gain_top20` |
| `logs/v2_p0_stable_gain_top20_depth2` | `logs/secom/v2/v2_p0_stable_gain_top20_depth2` |
| `logs/v2_p0_stable_gain_top20_depth2_lambda5` | `logs/secom/v2/v2_p0_stable_gain_top20_depth2_lambda5` |
| `logs/v2_p0_weighted_gain_sensor_counts` | `logs/secom/v2/v2_p0_weighted_gain_sensor_counts` |
| `logs/v2_recall80_demo_predictions.csv` | `logs/secom/demos/v2/v2_recall80_demo_predictions.csv` |
| `logs/v2_recall90_demo_predictions.csv` | `logs/secom/demos/v2/v2_recall90_demo_predictions.csv` |
| `logs/v2_rf_leaf_compare` | `logs/secom/v2/v2_rf_leaf_compare` |
| `logs/v2_rf_lr_time_compare` | `logs/secom/v2/v2_rf_lr_time_compare` |
| `logs/v2_rf_weight_compare` | `logs/secom/v2/v2_rf_weight_compare` |
| `logs/v2_sensor_class_distribution` | `logs/secom/v2/v2_sensor_class_distribution` |
| `logs/v2_sensor_importance` | `logs/secom/v2/v2_sensor_importance` |
| `logs/v2_sensor_importance_run2` | `logs/secom/v2/v2_sensor_importance_run2` |
| `logs/v2_sensor_stability` | `logs/secom/v2/v2_sensor_stability` |
| `logs/v2_time_weight_compare` | `logs/secom/v2/v2_time_weight_compare` |
| `logs/xgboost_20261006` | `logs/secom/exploratory/xgboost_20261006` |
| `logs/xgboost_top20_20261006` | `logs/secom/exploratory/xgboost_top20_20261006` |
| `logs/xgboost_top20_time_20261006` | `logs/secom/exploratory/xgboost_top20_time_20261006` |
