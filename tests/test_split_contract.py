# ==========================================
# split 계약과 누수 방어 테스트
# - 기존 CSV 중복·새 계약의 역할 혼합·시간 역전을 학습 전에 차단
# - 예약 metadata가 all_except_metadata 입력 특징에 섞이지 않음을 확인
# ==========================================

import unittest
from dataclasses import replace

import pandas as pd

from src.modeling_config import MODELING_CONFIG, CrossValidationConfig
from src.dataset_schema import resolve_feature_columns
from src.split_contract import (
    SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID, validate_split_pair,
    validate_train_role, training_folds,
)


class SplitContractTest(unittest.TestCase):
    dataset = replace(MODELING_CONFIG.dataset, timestamp_column=None,
                      feature_selection_mode="all_except_metadata", feature_column_prefix=None)

    @staticmethod
    def _frame(values, *, role="train", protocol="toy"):
        # 원본 ID는 split 파일에서 인덱스가 초기화돼도 유지한다.
        return pd.DataFrame({"value": values, "label": [-1] * len(values),
                             SOURCE_ROW_ID: values, SPLIT_ROLE: role, PROTOCOL_ID: protocol})

    def test_excludes_reserved_metadata_from_features(self):
        self.assertEqual(resolve_feature_columns(self._frame([1, 2]).columns, self.dataset), ("value",))

    def test_rejects_validation_or_test_as_train(self):
        for role in ("validation", "test"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "train 역할"):
                validate_train_role(self._frame([1], role=role))

    def test_rejects_shared_source_ids(self):
        with self.assertRaisesRegex(ValueError, "동일한 원본 행 ID"):
            validate_split_pair(self._frame([1, 2]), self._frame([2, 3], role="validation"), self.dataset)

    def test_rejects_mixed_generation_contracts(self):
        with self.assertRaisesRegex(ValueError, "생성 계약이 다릅니다"):
            validate_split_pair(self._frame([1]), self._frame([2], role="validation", protocol="other"), self.dataset)

    def test_checks_legacy_rows_without_dataframe_index(self):
        train = pd.DataFrame({"value": [1., 2.], "label": [-1, 1]})
        valid = train.iloc[[1]][["label", "value"]].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "동일한 데이터 행"):
            validate_split_pair(train, valid, self.dataset)

    def test_rejects_reversed_time_boundary(self):
        dataset = replace(self.dataset, timestamp_column="time", timestamp_format="%Y-%m-%d")
        train = self._frame([1]).assign(time="2026-02-01")
        valid = self._frame([2], role="validation").assign(time="2026-01-01")
        with self.assertRaisesRegex(ValueError, "시간 범위가 겹칩니다"):
            validate_split_pair(train, valid, dataset, temporal=True)

    def test_repeated_group_folds_keep_groups_separate(self):
        # 각 그룹에 두 label을 넣어 모든 fold의 분류 지표를 계산 가능하게 한다.
        frame = pd.DataFrame({"value": range(12), "label": [-1, 1] * 6,
                              "lot": [n // 2 for n in range(12)]})
        config = replace(MODELING_CONFIG, dataset=replace(self.dataset, group_columns=("lot",)),
                         experiment=replace(MODELING_CONFIG.experiment,
                                            cv=CrossValidationConfig(2, 2, 42)))
        folds = list(training_folds(frame[["value"]], frame.label, frame, config))
        self.assertEqual(len(folds), 4)
        for train_index, valid_index in folds:
            self.assertFalse(set(frame.lot.iloc[train_index]) & set(frame.lot.iloc[valid_index]))


if __name__ == "__main__":
    unittest.main()
