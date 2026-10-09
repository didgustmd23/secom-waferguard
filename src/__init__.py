"""WaferGuard 공통 코어와 도메인별 패키지의 진입점."""

from pathlib import Path
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

# ==========================================
# 기존 SECOM 모듈 경로 호환
# - 실제 소스는 src/sensor_ml 아래에만 유지한다.
# - 과거 src.experiments 등의 import·모듈 실행은 같은 소스를 찾게 한다.
# - 새 코드와 문서는 src.sensor_ml 경로를 사용한다.
# - 저장 모델의 src.sensor_inference 등 공통 클래스 경로는 이동하지 않는다.
# ==========================================
__path__.append(str(Path(__file__).resolve().parent / "sensor_ml"))

# 직전 src.secom 경로도 같은 실제 패키지 위치를 조회하게 한다.
# 클래스가 있는 공통 코어는 그대로 두고 새 실행기는 src.sensor_ml을 사용한다.
from . import sensor_ml

# 별도 패키지 객체로 옛 경로를 제공한다. 같은 객체를 두 이름에 등록하면
# 옛 하위 패키지 import가 새 패키지의 속성을 덮어쓸 수 있다.
secom = ModuleType(f"{__name__}.secom")
secom.__path__ = list(sensor_ml.__path__)
secom.__package__ = secom.__name__
secom.__spec__ = ModuleSpec(secom.__name__, loader=None, is_package=True)
secom.__spec__.submodule_search_locations = secom.__path__
sys.modules.setdefault(secom.__name__, secom)
