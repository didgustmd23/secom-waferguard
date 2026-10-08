# ==========================================
# 센서 CSV 예측 CLI 입력·저장 계약 검증
# - 실제 SECOM 대신 합성 CSV와 예측 객체를 사용
# - 원본 중복 헤더·행 길이 오류·추가 센서·잘못된 타입을 거부
# - 정상 결과 행 순서를 보존하고 기존 출력은 덮어쓰지 않음
# ==========================================

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.predict_cli import predict_sensor_file, read_sensor_csv, write_new_csv


class PredictCliTest(unittest.TestCase):
    def test_rejects_csv_format_errors(self):
        """자동 헤더 변경이나 잘못된 행 구조가 검증을 우회하지 못한다."""
        for content in ("a,a\n1,2\n", "a,b\n1,2,3\n", "a,b\n1\n", "", ",b\n1,2\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "input.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_sensor_csv(path)

    def test_prediction_order_and_no_overwrite(self):
        """예측 결과는 입력 행 순서를 유지하고 기존 출력 파일을 보존한다."""
        class SyntheticInference:
            def predict(self, frame):
                # 예측 모델 자체의 검증은 sensor_inference 테스트에서 수행한다.
                return pd.DataFrame({"positive_score": frame.a / 10,
                                     "predicted_positive": frame.a >= 5,
                                     "predicted_label": [1, -1]})

        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "input.csv", Path(temporary) / "predictions.csv"
            write_new_csv(pd.DataFrame({"a": [8, 2], "b": [1, 3]}), source)
            with patch("src.predict_cli.load_sensor_bundle",
                       return_value={"inference": SyntheticInference()}) as load:
                result = predict_sensor_file("bundle", source, target, trusted=True)
            load.assert_called_once_with("bundle", trusted=True)
            self.assertEqual(result.input_row_index.tolist(), [0, 1])
            self.assertEqual(result.positive_score.tolist(), [0.8, 0.2])
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                predict_sensor_file("bundle", source, target, trusted=True)
            self.assertEqual(target.read_bytes(), before)

    def test_invalid_inference_does_not_create_result(self):
        """계약 검증 실패나 신뢰 확인 실패 시 예측 파일을 생성하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "input.csv", Path(temporary) / "predictions.csv"
            write_new_csv(pd.DataFrame({"wrong_sensor": [1]}), source)
            with self.assertRaisesRegex(ValueError, "코드 실행 위험"):
                predict_sensor_file("not-read", source, target, trusted=False)
            self.assertFalse(target.exists())
            with patch("src.predict_cli.load_sensor_bundle") as load:
                load.return_value["inference"].predict.side_effect = ValueError("센서 계약 오류")
                with self.assertRaisesRegex(ValueError, "센서 계약 오류"):
                    predict_sensor_file("bundle", source, target, trusted=True)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
