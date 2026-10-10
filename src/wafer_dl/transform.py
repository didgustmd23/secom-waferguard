"""모델·에이전트 없이 실행하는 범주형 맵의 결정적 입력 변환."""

from pathlib import Path

import numpy as np

from .validate_canonical import read_json, require


# ==========================================
# 변환 설정 계약 검증
# - 현재 지원하는 좌표 규칙·채널 순서·padding만 명시적으로 허용
# - 지원하지 않는 보간으로 조용히 대체하거나 자동 증강하지 않음
# ==========================================
def load_transform_config(path):
    config = read_json(Path(path))
    keys = {'transform_version', 'target_size', 'resize_backend',
            'preserve_aspect_ratio', 'padding_value', 'channel_order', 'augmentation'}
    require(set(config) == keys and config['transform_version'] == 'wm_resize_v1', 'INVALID_TRANSFORM_CONFIG')
    size = config['target_size']
    require(type(size) is list and len(size) == 2 and
            all(type(value) is int and 1 <= value <= 1024 for value in size), 'INVALID_TARGET_SIZE')
    require(config['resize_backend'] == 'numpy_half_pixel_nearest_v1' and
            config['preserve_aspect_ratio'] is True and
            type(config['padding_value']) is int and config['padding_value'] == 0 and
            config['channel_order'] == ['defect', 'valid'] and
            config['augmentation'] == 'none', 'UNSUPPORTED_TRANSFORM_POLICY')
    return config


def validate_map(array):
    # 0=외부·1=정상·2=불량이라는 이산 상태와 메모리 상한을 검사한다.
    require(type(array) is np.ndarray and array.dtype == np.uint8 and
            array.ndim == 2 and array.size > 0 and array.size <= 1_000_000,
            'INVALID_TRANSFORM_MAP')
    require(np.isin(array, (0, 1, 2)).all() and np.any(array > 0), 'INVALID_TRANSFORM_MAP_VALUES')


# ==========================================
# 비율 유지 최근접 resize와 중앙 padding
# - 원본 crop·중심 이동·다이 상태 보정 없이 같은 좌표계 사용
# - 크기는 floor(길이*배율+0.5), 홀수 여백은 아래·오른쪽에 추가
# - 좌표는 floor((출력 인덱스+0.5)*원본 길이/새 길이)
# - 정수 연산으로 좌표 경계에서 부동소수점 반올림 차이를 방지
# ==========================================
def resize_map(array, target_size=(128, 128)):
    validate_map(array)
    require(type(target_size) in (list, tuple) and len(target_size) == 2 and
            all(type(value) is int and 1 <= value <= 1024 for value in target_size), 'INVALID_TARGET_SIZE')
    height, width = array.shape
    target_h, target_w = target_size
    # min(목표 높이/높이, 목표 너비/너비)를 분수로 유지한다.
    if target_h * width <= target_w * height:
        numerator, denominator = target_h, height
    else:
        numerator, denominator = target_w, width
    new_h = min(target_h, max(1, (2 * height * numerator + denominator) // (2 * denominator)))
    new_w = min(target_w, max(1, (2 * width * numerator + denominator) // (2 * denominator)))
    ys = np.minimum(((2 * np.arange(new_h, dtype=np.int64) + 1) * height) // (2 * new_h), height - 1)
    xs = np.minimum(((2 * np.arange(new_w, dtype=np.int64) + 1) * width) // (2 * new_w), width - 1)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    output = np.zeros((target_h, target_w), dtype=np.uint8)
    output[top:top + new_h, left:left + new_w] = array[ys[:, None], xs[None, :]]
    geometry = {'original_height': height, 'original_width': width,
                'resized_height': new_h, 'resized_width': new_w,
                'scale': numerator / denominator, 'top': top, 'left': left,
                'bottom': target_h - new_h - top, 'right': target_w - new_w - left}
    return output, geometry


def make_channels(array):
    # 그림의 RGB 색상이 아닌 다이 상태값으로 2채널을 생성한다.
    require(type(array) is np.ndarray and array.ndim == 2 and array.size > 0 and
            array.dtype == np.uint8 and np.isin(array, (0, 1, 2)).all(), 'INVALID_CHANNEL_MAP')
    # 유효 영역이 사라진 변환 결과도 진단할 수 있도록 이 함수에서는 빈 마스크를 허용한다.
    return np.ascontiguousarray(np.stack((array == 2, array > 0)).astype(np.float32))


def transform_map(array, config):
    """로드·검증된 설정으로 맵·기하 정보를 반환한다. 마스크는 필요할 때 별도 생성한다."""
    return resize_map(array, config['target_size'])


# ==========================================
# resize 전후의 기계적 패턴 보존 지표
# - 픽셀 수 자체의 동일성을 요구하지 않고 유효 영역·불량 비율 비교
# - 전체 불량 소실과 유효 영역 소실은 후속 학습 전 검토 대상으로 기록
# ==========================================
def preservation_metrics(original, transformed):
    original_valid = int(np.count_nonzero(original))
    original_failed = int(np.count_nonzero(original == 2))
    valid = int(np.count_nonzero(transformed))
    failed = int(np.count_nonzero(transformed == 2))
    before = original_failed / original_valid
    after = failed / valid if valid else None
    return {'original_valid_dies': original_valid, 'original_failed_dies': original_failed,
            'transformed_valid_pixels': valid, 'transformed_failed_pixels': failed,
            'original_failed_ratio': before, 'transformed_failed_ratio': after,
            'absolute_ratio_delta': abs(after - before) if after is not None else None,
            'failed_region_lost': original_failed > 0 and failed == 0,
            'valid_region_lost': valid == 0}
