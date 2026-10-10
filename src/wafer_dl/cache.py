"""확정 분할·검사 승인과 연결된 범주형 uint8 입력 캐시 생성."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from .check_transform import index_split, iter_split_maps
from .transform import load_transform_config, preservation_metrics, transform_map
from .validate_canonical import CLASS_NAMES, ValidationError, read_json, require, validation_stage


def file_digest(path):
    # 파일 내용의 손상·다른 묶음 혼용을 확인한다. 사용자에게 수동 해시 관리를 요구하지 않는다.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache_config(path):
    config = read_json(Path(path))
    require(set(config) == {'cache_version', 'shard_max_rows', 'buffer_max_bytes'} and
            config['cache_version'] == 'wm_input_cache_v1', 'INVALID_CACHE_CONFIG')
    require(type(config['shard_max_rows']) is int and 1 <= config['shard_max_rows'] <= 4096 and
            type(config['buffer_max_bytes']) is int and 1_048_576 <= config['buffer_max_bytes'] <= 268_435_456,
            'INVALID_CACHE_BUFFER')
    return config


# ==========================================
# 개발 구간 검사와 사용자 시각 검토 승인 확인
# - 다른 분할·해상도·변환 규칙의 과거 통과 결과를 재사용하지 않음
# - Test 캐시 생성 승인은 평가·배포 승인과 별도로 취급
# ==========================================
def validate_check(check_dir, split_report, transform_config, approved):
    require(approved is True, 'VISUAL_REVIEW_APPROVAL_REQUIRED')
    completion = read_json(Path(check_dir) / 'SUCCESS.json')
    report = read_json(Path(check_dir) / 'transform_check.json')
    require(completion.get('status') == 'transform_check_complete' and
            completion.get('review_required') is False and
            report.get('status') == 'transform_check_passed', 'TRANSFORM_CHECK_NOT_PASSED')
    require(report.get('split_protocol_id') == split_report['split_protocol_id'] and
            report.get('config') == transform_config and report.get('test_maps_read') is False and
            report.get('rows_checked') == sum(split_report['split_counts'][name] for name in ('train', 'validation')),
            'TRANSFORM_CHECK_CONTRACT_MISMATCH')
    expected = {(name, label): split_report['class_counts'][name][index]
                for name in ('train', 'validation') for index, label in enumerate(CLASS_NAMES)}
    actual = {}
    for item in report.get('classes', []):
        key = (item.get('split'), item.get('class'))
        require(key not in actual and item.get('failed_region_lost') == 0 and
                item.get('valid_region_lost') == 0, 'INVALID_TRANSFORM_CHECK_SUMMARY')
        actual[key] = item.get('rows')
    require(actual == expected, 'INVALID_TRANSFORM_CHECK_SUMMARY')


# ==========================================
# 구간별 bounded buffer와 숫자 전용 shard 저장
# - 표본당 uint8 한 채널만 저장하며 float 채널은 캐시에 넣지 않음
# - 최대 행 수와 버퍼 바이트 상한 중 작은 값을 사용
# - 저장 직후 mmap으로 shape·dtype·범주·유효 영역을 검증
# ==========================================
class CacheWriter:
    def __init__(self, root, split, target_size, config):
        self.root, self.split = root, split
        self.directory = root / split
        self.directory.mkdir(exist_ok=False)
        self.capacity = min(config['shard_max_rows'], config['buffer_max_bytes'] // int(np.prod(target_size)))
        require(self.capacity > 0, 'CACHE_BUFFER_TOO_SMALL')
        self.buffer = np.empty((self.capacity, *target_size), dtype=np.uint8)
        self.used, self.rows = 0, 0
        self.shards, self.counts = [], [0] * 9
        self.index = (self.directory / 'index.jsonl').open('x', encoding='utf-8')

    def append(self, row, transformed):
        path = f'{self.split}/shard_{len(self.shards):06d}.npy'
        self.buffer[self.used] = transformed
        record = {'sample_id': row['sample_id'], 'class_index': row['class_index'],
                  'pattern_label': row['pattern_label'], 'split': self.split,
                  'shard': path, 'offset': self.used}
        self.index.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n')
        self.used += 1
        self.rows += 1
        self.counts[row['class_index']] += 1
        if self.used == self.capacity:
            self.flush()

    def flush(self):
        if not self.used:
            return
        relative = f'{self.split}/shard_{len(self.shards):06d}.npy'
        path = self.root / relative
        # np.save에 object 배열을 전달하지 않으며 파일 이름도 내부 순번으로 고정한다.
        with path.open('xb') as stream:
            np.save(stream, self.buffer[:self.used], allow_pickle=False)
        array = np.load(path, mmap_mode='r', allow_pickle=False, max_header_size=10_000)
        try:
            require(array.dtype == np.uint8 and array.shape == (self.used, *self.buffer.shape[1:]) and
                    np.isin(array, (0, 1, 2)).all() and np.all(np.any(array > 0, axis=(1, 2))), 'INVALID_SAVED_CACHE')
        finally:
            array._mmap.close()
        self.shards.append({'path': relative, 'rows': self.used, 'bytes': path.stat().st_size,
                            'sha256': file_digest(path)})
        self.used = 0

    def finish(self):
        self.flush()
        self.index.close()
        path = self.directory / 'index.jsonl'
        return {'rows': self.rows, 'class_counts': self.counts, 'shards': self.shards,
                'index': f'{self.split}/index.jsonl', 'index_sha256': file_digest(path)}

    def close(self):
        # 실패한 경우 남은 버퍼를 완료 결과로 저장하지 않고 열린 파일만 닫는다.
        self.index.close()


def write_cache_report(report, output):
    (output / 'cache.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# WM-811K 입력 캐시', '', '범주형 uint8 맵을 저장했습니다. 모델 학습·성능 평가 결과가 아닙니다.', '',
             '| 구간 | 표본 수 | shard 수 |', '| --- | ---: | ---: |']
    for name, item in report['splits'].items():
        lines.append(f'| {name} | {item["rows"]} | {len(item["shards"])} |')
    lines += ['', f'Test 배열 준비 여부: {report["test_maps_read"]}',
              'Dataset 첫 배치 계약 검사가 남아 있으므로 training_ready=false입니다.',
              'Dataset은 선택한 구간만 읽습니다. 미라벨·격리 행은 캐시에 포함하지 않았습니다.',
              '원본·분할·검사 산출물은 변경하지 않았습니다. 상세 정보는 에이전트의 외부 LLM에 보내지 마세요.']
    (output / 'cache.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ==========================================
# 캐시 실행 경계
# - 기본은 Train/Validation만 허용, Test 포함 시 명시적 승인 필수
# - 원본·분할 계약 재검증 후 기존 변환 함수를 동일하게 재사용
# - 전체 소실 발생 시 완료 표식 없이 중단하고 자동 보정하지 않음
# ==========================================
def build_cache(split_dir, check_dir, output_dir, transform_path, cache_path,
                splits=('train', 'validation'), approved=False, allow_test=False,
                canonical_dir=None, progress=None):
    require(type(splits) in (tuple, list) and bool(splits) and len(splits) == len(set(splits)) and
            all(name in ('train', 'validation', 'test') for name in splits), 'INVALID_REQUESTED_SPLITS')
    require('test' not in splits or allow_test is True, 'TEST_PREPARATION_APPROVAL_REQUIRED')
    split_dir, check_dir = Path(split_dir).resolve(strict=True), Path(check_dir).resolve(strict=True)
    report = read_json(split_dir / 'split.json')
    root = Path(canonical_dir or report.get('canonical_root', '')).resolve(strict=True)
    output = Path(output_dir).resolve()
    require(not output.exists() and all(not output.is_relative_to(path) and not path.is_relative_to(output)
                                       for path in (root, split_dir, check_dir)), 'INVALID_CACHE_OUTPUT_DIRECTORY')
    transform_config, cache_config = load_transform_config(transform_path), load_cache_config(cache_path)
    with tempfile.TemporaryDirectory(prefix='wm_cache_') as temporary:
        database = sqlite3.connect(str(Path(temporary) / 'manifest.sqlite'))
        writers = {}
        try:
            split_report = index_split(root, split_dir, database)
            validate_check(check_dir, split_report, transform_config, approved)
            expected = sum(split_report['split_counts'][name] for name in splits)
            output.mkdir(parents=True, exist_ok=False)
            for name in splits:
                writers[name] = CacheWriter(output, name, transform_config['target_size'], cache_config)
            checked = 0
            for row, name, original in iter_split_maps(root, database, tuple(splits)):
                transformed, _ = transform_map(original, transform_config)
                metrics = preservation_metrics(original, transformed)
                require(not metrics['valid_region_lost'] and not metrics['failed_region_lost'], 'CACHE_PATTERN_LOSS_REVIEW_REQUIRED')
                writers[name].append(row, transformed)
                checked += 1
                if progress and (checked % 1000 == 0 or checked == expected):
                    progress(checked, expected)
            require(checked == expected, 'CACHE_ROW_COUNT_MISMATCH')
            saved = {name: writer.finish() for name, writer in writers.items()}
            for name, item in saved.items():
                require(item['class_counts'] == split_report['class_counts'][name] and
                        item['rows'] == split_report['split_counts'][name], 'CACHE_CLASS_COUNT_MISMATCH')
            train_counts = split_report['class_counts']['train']
            result = {'status': 'cache_complete', 'cache_version': cache_config['cache_version'],
                      'dataset_version': split_report['dataset_version'],
                      'split_protocol_id': split_report['split_protocol_id'],
                      'source_split_manifest_sha256': file_digest(split_dir / 'split_manifest.jsonl'),
                      'transform_config': transform_config, 'cache_config': cache_config,
                      'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
                      'splits': saved, 'test_maps_read': 'test' in splits,
                      'visual_review_confirmed': True, 'training_ready': False,
                      'train_class_counts': train_counts,
                      'suggested_train_class_weights': [sum(train_counts) / (9 * count) for count in train_counts]}
            write_cache_report(result, output)
            # 캐시 내용을 모두 저장·검증한 후에만 완료 표식을 노출한다.
            (output / 'SUCCESS.json').write_text(json.dumps({'status': 'cache_complete',
                'split_protocol_id': result['split_protocol_id'], 'training_ready': False}), encoding='utf-8')
        finally:
            for writer in writers.values():
                writer.close()
            database.close()
    return {'status': 'cache_complete', 'training_ready': False}


def main():
    parser = argparse.ArgumentParser(description='WM-811K 분할별 uint8 입력 캐시 생성')
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--check-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--transform-config', default='configs/wm811k/transform.json')
    parser.add_argument('--cache-config', default='configs/wm811k/cache.json')
    parser.add_argument('--splits', nargs='+', choices=('train', 'validation', 'test'), default=['train', 'validation'])
    parser.add_argument('--confirm-visual-review', action='store_true')
    parser.add_argument('--confirm-test-preparation', action='store_true')
    parser.add_argument('--canonical-dir')
    args = parser.parse_args()
    try:
        with validation_stage('cache'):
            result = build_cache(args.split_dir, args.check_dir, args.output_dir,
                args.transform_config, args.cache_config, args.splits,
                args.confirm_visual_review, args.confirm_test_preparation, args.canonical_dir,
                progress=lambda done, total: print(f'캐시 생성: {done}/{total}', flush=True))
    except ValidationError as error:
        print('캐시 생성 실패 코드: ' + str(error))
        print('완료 표식 없는 결과는 사용하지 마세요. 기존 폴더를 덮어쓰거나 자동 재시도하지 않습니다.')
        return 1
    print(result['status'])
    print('다음으로 Dataset 첫 배치 계약을 검증하세요. 모델은 학습하지 않았습니다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
