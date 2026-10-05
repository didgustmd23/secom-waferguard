# ==========================================
# 시간 holdout 우선 split 생성 테스트
# - 시간·그룹 경계를 유지하며 원본 행을 정확히 한 구간에 배정
# - 파싱 실패와 불가능한 분할을 중단하고 기존 파일을 덮어쓰지 않음
# ==========================================

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_split_pair
from src.step3_split import (
    prepare_split_frames, write_split_files, make_time_split,
    check_timestamp_overlap, check_duplicate_leakage,
)


class TemporalSplitTest(unittest.TestCase):
    dataset = replace(MODELING_CONFIG.dataset, timestamp_format="%Y-%m-%d")

    @staticmethod
    def _frame():
        # 입력 순서를 뒤집어도 실제 시간순으로 분할하는지 확인한다.
        return pd.DataFrame({"sensor_0": range(20), "label": [-1, 1] * 10,
                             "timestamp": pd.date_range("2026-01-01", periods=20).strftime("%Y-%m-%d")}).iloc[::-1].reset_index(drop=True)

    def test_creates_disjoint_reproducible_temporal_splits(self):
        frame = self._frame()
        splits = prepare_split_frames(frame, self.dataset)
        again = prepare_split_frames(frame, self.dataset)
        self.assertEqual([len(splits[name]) for name in ("time_train", "time_valid", "time_test")], [14, 3, 3])
        self.assertEqual(sorted(pd.concat([splits[name] for name in ("time_train", "time_valid", "time_test")])[SOURCE_ROW_ID]), list(range(20)))
        for name, split in splits.items():
            pd.testing.assert_frame_equal(split, again[name])
        validate_split_pair(splits["time_train"], splits["time_valid"], self.dataset, temporal=True)

    def test_preserves_equal_timestamp_at_boundaries(self):
        frame = self._frame()
        frame["timestamp"] = [f"2026-01-{n // 2 + 1:02d}" for n in range(20)]
        splits = prepare_split_frames(frame, self.dataset)
        timestamp_sets = [set(splits[name].timestamp) for name in ("time_train", "time_valid", "time_test")]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertFalse(timestamp_sets[left] & timestamp_sets[right])

    def test_keeps_declared_groups_in_one_partition(self):
        dataset = replace(self.dataset, group_columns=("lot",))
        frame = self._frame().assign(lot=[n // 4 for n in range(20)])
        splits = prepare_split_frames(frame, dataset)
        lots = [set(splits[name].lot) for name in ("time_train", "time_valid", "time_test")]
        self.assertFalse(lots[0] & lots[1] | lots[0] & lots[2] | lots[1] & lots[2])

    def test_rejects_groups_spanning_every_boundary(self):
        dataset = replace(self.dataset, group_columns=("lot",))
        with self.assertRaisesRegex(ValueError, "세 구간으로 나눌 수 없습니다"):
            prepare_split_frames(self._frame().assign(lot="one_lot"), dataset)

    def test_rejects_invalid_timestamp(self):
        frame = self._frame()
        frame.loc[0, "timestamp"] = "invalid"
        with self.assertRaisesRegex(ValueError, "timestamp 파싱 실패"):
            prepare_split_frames(frame, self.dataset)

    def test_rejects_invalid_split_ratios(self):
        for train_ratio, validation_ratio in ((0, .15), (.7, .4), (float("nan"), .15)):
            with self.subTest(train=train_ratio), self.assertRaises(ValueError):
                make_time_split(self._frame(), "timestamp", "%Y-%m-%d",
                                train_ratio=train_ratio, validation_ratio=validation_ratio)

    def test_writes_manifest_and_prevents_overwrite(self):
        splits = prepare_split_frames(self._frame(), self.dataset)
        with TemporaryDirectory() as directory:
            write_split_files(splits, self.dataset, directory)
            self.assertTrue((Path(directory) / "manifest.json").exists())
            loaded = pd.read_csv(Path(directory) / "time_train.csv")
            self.assertEqual(loaded[PROTOCOL_ID].iloc[0], splits["time_train"][PROTOCOL_ID].iloc[0])
            with self.assertRaises(FileExistsError):
                write_split_files(splits, self.dataset, directory)

    def test_overlap_check_uses_profile_timestamp_format(self):
        frames = [pd.DataFrame({"timestamp": ["2026-01-01"]}) for _ in range(3)]
        result = check_timestamp_overlap(*frames, "timestamp", "time", "%Y-%m-%d")
        self.assertEqual(result[0]["status"], "FAIL")

    def test_timestamp_parse_failure_is_not_pass(self):
        frames = [pd.DataFrame({"timestamp": ["invalid"]}) for _ in range(3)]
        self.assertEqual(check_timestamp_overlap(*frames, "timestamp", "time", "%Y-%m-%d")[0]["status"], "FAIL")

    def test_finds_alternate_boundary_if_forward_move_fails(self):
        frame = pd.DataFrame({"timestamp": [f"2026-01-{day:02d}" for day in [1, 2, 3, 4, 5, 6, 7, 7, 7, 7]],
                              "label": [-1, 1] * 5})
        train, valid, test = make_time_split(frame, "timestamp", "%Y-%m-%d")
        self.assertEqual([len(train), len(valid), len(test)], [5, 1, 4])

    def test_duplicate_content_check_ignores_split_metadata(self):
        splits = prepare_split_frames(self._frame(), self.dataset)
        valid = splits["time_valid"].copy()
        # 내용만 복사하고 원본 ID·역할은 그대로 두어 metadata가 중복을 숨기지 않게 한다.
        columns = ["sensor_0", "label", "timestamp"]
        valid.loc[0, columns] = splits["time_train"].loc[0, columns].to_numpy()
        result = check_duplicate_leakage(splits["time_train"], valid, splits["time_test"])
        self.assertEqual(result[0]["status"], "FAIL")

    def test_no_csv_written_when_checks_fail(self):
        splits = prepare_split_frames(self._frame(), self.dataset)
        splits["time_valid"].loc[0, "timestamp"] = "invalid"
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "검사가 실패"):
                write_split_files(splits, self.dataset, directory)
            self.assertFalse(list(Path(directory).glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
