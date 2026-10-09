# ==========================================
# 프로젝트 영역 분리와 기존 경로 호환 검증
# - 학습·실데이터·저장된 모델을 사용하지 않는다.
# - SECOM 실행기의 실제 위치와 과거 import 조회 경로를 비교한다.
# - 폴더 이동 후 프로젝트 루트·기본 설정 경로가 유지되는지 확인한다.
# - 저장 모델에서 참조하는 공통 클래스 모듈 경로를 보존한다.
# ==========================================
from importlib.util import find_spec
from importlib import import_module
from pathlib import Path
import unittest

from src.modeling_config import DEFAULT_CONFIG_PATH
from src.modeling_models import XGBoostClassifierAdapter
from src.sensor_inference import SensorInference
from src.sensor_ml.data_pipeline.step3_split import PROJECT_ROOT
from src.sensor_ml.experiments.time_weight_compare import DEFAULT_PRESET


class SourceLayoutTest(unittest.TestCase):
    def test_legacy_imports_do_not_replace_new_package(self):
        """옛 경로 조회가 새 패키지 속성·mock 경로를 덮어쓰지 않는다."""
        # 호환 패키지는 파일 위치만 공유하고 패키지 객체는 분리해야 한다.
        canonical = import_module('src.sensor_ml.experiments')
        intermediate = import_module('src.secom.experiments')
        original = import_module('src.experiments')
        parent = import_module('src.sensor_ml')
        self.assertIs(parent.experiments, canonical)
        self.assertIsNot(canonical, intermediate)
        self.assertIsNot(canonical, original)

    def test_sensor_ml_and_legacy_resolve_same_files(self):
        """과거 경로를 위한 소스 복제 없이 같은 실제 파일을 찾는다."""
        modules = ('data_pipeline.step3_split', 'experiments.time_weight_compare',
                   'inference.predict_cli', 'evaluation.run_evaluation',
                   'diagnostics.sensor_importance', 'verification.check_sensor_inference')
        for module in modules:
            with self.subTest(module=module):
                # 모듈 실행이나 fit 대신 import 검색 결과의 파일 위치만 비교한다.
                canonical = find_spec(f'src.sensor_ml.{module}')
                legacy = find_spec(f'src.{module}')
                self.assertEqual(Path(canonical.origin), Path(legacy.origin))
                intermediate = find_spec(f'src.secom.{module}')
                self.assertEqual(Path(canonical.origin), Path(intermediate.origin))
                self.assertIn('sensor_ml', Path(canonical.origin).parts)

    def test_project_paths_and_saved_class_paths(self):
        """데이터·설정 위치 및 직렬화된 클래스의 경로를 변경하지 않는다."""
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(PROJECT_ROOT, root)
        self.assertEqual(DEFAULT_CONFIG_PATH, root / 'config.json')
        self.assertEqual(DEFAULT_PRESET, root / 'configs/experiments/time_weight_v2.json')
        self.assertEqual(SensorInference.__module__, 'src.sensor_inference')
        self.assertEqual(XGBoostClassifierAdapter.__module__, 'src.modeling_models')
