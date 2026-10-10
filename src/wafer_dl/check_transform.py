"""Train/Validation의 입력 변환·패턴 보존을 로컬에서 검사한다."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import zipfile

import numpy as np

from .transform import load_transform_config, make_channels, preservation_metrics, transform_map
from .validate_canonical import (
    CLASS_NAMES, ValidationError, decode_json, read_json, read_numeric_member,
    require, validate_manifest, validate_metadata, validation_stage,
)


# ==========================================
# 확정 분할과 canonical의 metadata 연결 검증
# - 원본 전체 manifest는 검증하지만 Test 배열은 열지 않음
# - 원본 라벨·배열 참조가 파생 manifest에서 바뀌지 않았는지 확인
# - 충돌 제외·클래스 집계·Lot/맵 그룹 보호를 다시 검사
# ==========================================
def index_split(root, split_dir, database):
    success = read_json(split_dir / 'SUCCESS.json')
    report = read_json(split_dir / 'split.json')
    require(success.get('status') == report.get('status') == 'split_complete' and
            report.get('split_complete') is True and report.get('overlap_check_passed') is True,
            'INCOMPLETE_SPLIT')
    protocol = report.get('split_protocol_id')
    require(type(protocol) is str and re.fullmatch('wm_split_[0-9a-f]{64}', protocol) and
            success.get('split_protocol_id') == protocol, 'INVALID_SPLIT_PROTOCOL')
    summary, version = validate_metadata(root)
    require(summary.get('label_normalization_version') == 'wm_label_v2' and
            report.get('dataset_version') == version, 'SPLIT_DATASET_MISMATCH')
    validate_manifest(root, summary, version, database)
    digest = hashlib.sha256()
    with (root / 'manifest.jsonl').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    require(digest.hexdigest() == report.get('source_manifest_sha256'), 'SOURCE_MANIFEST_CHANGED')
    digest.update(json.dumps(report.get('config'), sort_keys=True).encode('utf-8'))
    require('wm_split_' + digest.hexdigest() == protocol, 'SPLIT_PROTOCOL_MISMATCH')
    conflicts = {row[0] for row in database.execute('SELECT hash FROM rows WHERE hash IS NOT NULL '
                 'GROUP BY hash HAVING COUNT(DISTINCT label)>1')}
    database.execute('CREATE TABLE assignments(position INTEGER PRIMARY KEY, split TEXT, gid TEXT, group_split TEXT)')
    path = split_dir / 'split_manifest.jsonl'
    require(path.is_file() and path.resolve().is_relative_to(split_dir), 'UNSAFE_SPLIT_MANIFEST')
    counts = {name: [0] * 9 for name in ('train', 'validation', 'test')}
    excluded = 0
    with path.open('rb') as stream:
        for position, original_text in database.execute('SELECT position,record FROM rows ORDER BY position'):
            payload = stream.readline(65_537)
            require(0 < len(payload) <= 65_536, 'INVALID_SPLIT_MANIFEST_LENGTH')
            row, original = decode_json(payload), decode_json(original_text)
            require(set(row) == set(original) | {'split_protocol_id', 'split_group_id', 'group_split', 'exclusion_reasons'}
                    and all(row.get(key) == value for key, value in original.items() if key != 'split')
                    and row.get('split_protocol_id') == protocol, 'SPLIT_SOURCE_ROW_CHANGED')
            conflict = row['map_hash'] in conflicts
            require(row.get('exclusion_reasons') == (['MAP_LABEL_CONFLICT'] if conflict else []), 'INVALID_EXCLUSION_POLICY')
            gid, group_split, split = row.get('split_group_id'), row.get('group_split'), row.get('split')
            require(gid is None or (type(gid) is str and re.fullmatch('group_[0-9]{8}', gid)), 'INVALID_SPLIT_GROUP')
            require(group_split is None or group_split in counts, 'INVALID_GROUP_SPLIT')
            if conflict:
                excluded += 1
                require(split == 'quarantined' and gid is None and group_split is None, 'CONFLICT_NOT_QUARANTINED')
            elif row['quality_status'] != 'accepted':
                require(split == 'quarantined', 'INVALID_QUARANTINE_SPLIT')
            elif row['label_status'] != 'labeled':
                require(split == 'unlabeled', 'INVALID_UNLABELED_SPLIT')
            else:
                require(split in counts and group_split == split and gid is not None, 'INVALID_LABELED_SPLIT')
                counts[split][row['class_index']] += 1
            database.execute('INSERT INTO assignments VALUES(?,?,?,?)', (position, split, gid, group_split))
        require(not stream.read(1), 'EXTRA_SPLIT_ROWS')
    require(counts == report.get('class_counts') and all(all(value > 0 for value in values) for values in counts.values())
            and excluded == report.get('excluded_rows') and report.get('row_count') == summary['row_count']
            and {name: sum(values) for name, values in counts.items()} == report.get('split_counts'), 'SPLIT_SUMMARY_MISMATCH')
    # 한 그룹·Lot·맵이 두 구간으로 나뉘거나 일부만 미배정되는 것을 차단한다.
    for field in ('a.gid', 'r.lot', 'r.hash'):
        query = (f'SELECT COUNT(*) FROM (SELECT {field} FROM rows r JOIN assignments a USING(position) '
                 f'WHERE {field} IS NOT NULL AND a.split != "quarantined" GROUP BY {field} '
                 'HAVING COUNT(DISTINCT a.gid)>1 OR COUNT(DISTINCT a.group_split)>1 '
                 'OR (COUNT(a.group_split)>0 AND COUNT(a.group_split)<COUNT(*)))')
        require(database.execute(query).fetchone()[0] == 0, 'SPLIT_GROUP_OVERLAP_OR_MISSING')
    database.execute('CREATE INDEX rows_shard ON rows(shard)')
    return report


def read_checked_map(archive, row):
    # 수치 NPY만 읽고 참조된 원본 맵의 내용 식별값·통계를 다시 대조한다.
    member = archive.getinfo(row['map_ref']['key'] + '.npy')
    require(not member.flag_bits & 1, 'ENCRYPTED_ARRAY')
    array = read_numeric_member(archive, member)
    digest = hashlib.sha256(np.asarray(array.shape, dtype='<u8').tobytes())
    digest.update(array.tobytes(order='C'))
    require(digest.hexdigest() == row['map_hash'] and
            array.shape == (row['map_height'], row['map_width']) and
            int(np.count_nonzero(array)) == row['valid_die_count'] and
            int(np.count_nonzero(array == 2)) == row['failed_die_count'], 'SOURCE_MAP_CHANGED')
    return array


# ==========================================
# Train/Validation만 shard별 순회
# - zip을 한 번씩 열고 선택된 맵만 복원하여 전체 배열 적재 방지
# - 같은 shard에 Test 맵이 있어도 해당 NPY 구성원은 읽지 않음
# ==========================================
def iter_development_maps(root, database):
    yield from iter_split_maps(root, database, ('train', 'validation'))


def iter_split_maps(root, database, splits):
    # 호출자가 승인한 구간만 전달한다. 사용자 문자열을 SQL에 직접 삽입하지 않는다.
    require(bool(splits) and len(splits) == len(set(splits)) and
            all(name in ('train', 'validation', 'test') for name in splits), 'INVALID_REQUESTED_SPLITS')
    placeholders = ','.join('?' for _ in splits)
    shards = database.execute('SELECT DISTINCT r.shard FROM rows r JOIN assignments a USING(position) '
                              f'WHERE a.split IN ({placeholders}) ORDER BY r.shard', splits).fetchall()
    for (shard,) in shards:
        path = root / shard
        require(path.is_file() and path.resolve().is_relative_to(root) and
                path.stat().st_size <= 1_100_000_000, 'UNSAFE_TRANSFORM_SHARD')
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            require(1 <= len(members) <= 1000 and len({m.filename for m in members}) == len(members)
                    and sum(m.file_size for m in members) <= 1_010_000_000, 'INVALID_TRANSFORM_SHARD')
            for record, split in database.execute('SELECT r.record,a.split FROM rows r JOIN assignments a USING(position) '
                                                  f'WHERE r.shard=? AND a.split IN ({placeholders}) '
                                                  'ORDER BY r.position', (shard, *splits)):
                row = decode_json(record)
                yield row, split, read_checked_map(archive, row)


def save_examples(examples, output):
    # 클래스별 첫 사례와 크기·종횡비 극단 사례만 비교하며 대표성을 주장하지 않는다.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(['#eeeeee', '#348a50', '#d94a45'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    figure, axes = plt.subplots(len(examples), 2, figsize=(9, 3 * len(examples)), squeeze=False)
    try:
        for (title, (_, original, transformed)), pair in zip(examples.items(), axes):
            for axis, array, role in zip(pair, (original, transformed), ('Original', 'Transformed')):
                axis.imshow(array, cmap=cmap, norm=norm, interpolation='nearest')
                axis.set_title(f'{title} / {role} {array.shape}')
                axis.axis('off')
        figure.tight_layout()
        figure.savefig(output / 'examples.png', dpi=120)
    finally:
        plt.close(figure)


def write_check_report(report, output):
    (output / 'transform_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                          allow_nan=False), encoding='utf-8')
    lines = ['# WM-811K 변환·패턴 보존 검사', '', f'상태: `{report["status"]}`', '',
             'Train/Validation 전체 대상의 기계적 검사입니다. Test 맵은 읽지 않았습니다.', '',
             '| 구간 | 클래스 | 검사 수 | 불량 전체 소실 | 유효 영역 소실 | 평균 불량 비율 절대 변화 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in report['classes']:
        mean = row['mean_absolute_ratio_delta']
        lines.append(f'| {row["split"]} | {row["class"]} | {row["rows"]} | {row["failed_region_lost"]} | '
                     f'{row["valid_region_lost"]} | {mean:.6f} |' if mean is not None else
                     f'| {row["split"]} | {row["class"]} | {row["rows"]} | {row["failed_region_lost"]} | {row["valid_region_lost"]} | N/A |')
    lines += ['', '비율 차이 0.01은 1%p 차이입니다. 픽셀 수가 달라지므로 다이 개수의 동일성을 요구하지 않습니다.',
              '불량·유효 영역 전체 소실이 한 건이라도 있으면 검토 필요입니다. 자동 해상도 변경은 하지 않습니다.',
              '통과해도 가는 선·고리·국소 결함의 형태 보존을 보장하지 않습니다. 아래 원본/변환 그림을 직접 검토하세요.', '',
              '![원본·변환 비교](examples.png)', '',
              '클래스별 첫 사례와 작은 맵·큰 맵·극단 종횡비 사례입니다. 대표 표본이 아니며 Test 예시는 포함하지 않습니다.',
              '변환 캐시·Dataset·모델은 생성하지 않았습니다. 학습 준비 완료나 모델 성능 평가 결과가 아닙니다.',
              '집계·그림·행별 결과는 로컬 전용입니다. 에이전트의 외부 LLM·추적 서비스에 보내지 마세요.']
    (output / 'transform_check.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ==========================================
# 검사 실행·로컬 결과 저장
# - 행별 지표는 CSV에 즉시 기록하고 예시만 소수 보관
# - 전체 불량 소실이 없어도 사용자 시각 검토는 별도로 필요
# - 실패한 부분 출력은 완료 표식을 만들지 않으며 기존 결과를 덮어쓰지 않음
# ==========================================
def run_transform_check(split_dir, output_dir, config_path, canonical_dir=None, progress=None):
    split_dir, output = Path(split_dir).resolve(strict=True), Path(output_dir).resolve()
    source = read_json(split_dir / 'split.json')
    root = Path(canonical_dir or source.get('canonical_root', '')).resolve(strict=True)
    require(not output.exists() and all(not output.is_relative_to(path) and not path.is_relative_to(output)
                                       for path in (root, split_dir)), 'INVALID_TRANSFORM_OUTPUT_DIRECTORY')
    config = load_transform_config(config_path)
    with tempfile.TemporaryDirectory(prefix='wm_transform_') as temporary:
        database = sqlite3.connect(str(Path(temporary) / 'manifest.sqlite'))
        try:
            split_report = index_split(root, split_dir, database)
            expected = split_report['split_counts']['train'] + split_report['split_counts']['validation']
            output.mkdir(parents=True, exist_ok=False)
            aggregates, examples = {}, {}
            fields = ['sample_id', 'split', 'class', 'original_height', 'original_width', 'resized_height',
                      'resized_width', 'scale', 'top', 'left', 'bottom', 'right',
                      'original_valid_dies', 'original_failed_dies', 'transformed_valid_pixels',
                      'transformed_failed_pixels', 'original_failed_ratio', 'transformed_failed_ratio',
                      'absolute_ratio_delta', 'failed_region_lost', 'valid_region_lost']
            checked = 0
            with (output / 'preservation.csv').open('x', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row, split, original in iter_development_maps(root, database):
                    transformed, geometry = transform_map(original, config)
                    channels = make_channels(transformed)
                    require(channels.shape == (2, *config['target_size']) and channels.flags.c_contiguous and
                            np.all(channels[0] <= channels[1]), 'INVALID_CHANNEL_CONTRACT')
                    metrics = preservation_metrics(original, transformed)
                    writer.writerow({'sample_id': row['sample_id'], 'split': split,
                                     'class': row['pattern_label'], **geometry, **metrics})
                    key = (split, row['pattern_label'])
                    item = aggregates.setdefault(key, {'split': split, 'class': row['pattern_label'], 'rows': 0,
                        'failed_region_lost': 0, 'valid_region_lost': 0, 'delta_sum': 0., 'delta_count': 0,
                        'max_absolute_ratio_delta': None})
                    item['rows'] += 1
                    for name in ('failed_region_lost', 'valid_region_lost'):
                        item[name] += int(metrics[name])
                    delta = metrics['absolute_ratio_delta']
                    if delta is not None:
                        item['delta_sum'] += delta
                        item['delta_count'] += 1
                        item['max_absolute_ratio_delta'] = max(item['max_absolute_ratio_delta'] or 0., delta)
                    # 클래스별 첫 사례와 전체 개발 구간의 크기·종횡비 극단값을 보관한다.
                    label = row['pattern_label']
                    if label not in examples:
                        examples[label] = (0, original.copy(), transformed.copy())
                    for name, score in (('Smallest map', -original.size), ('Largest map', original.size),
                                        ('Extreme aspect', max(original.shape) / min(original.shape))):
                        if name not in examples or score > examples[name][0]:
                            examples[name] = (score, original.copy(), transformed.copy())
                    if metrics['failed_region_lost'] and 'First lost defect' not in examples:
                        examples['First lost defect'] = (0, original.copy(), transformed.copy())
                    checked += 1
                    if progress and (checked % 1000 == 0 or checked == expected):
                        progress(checked, expected)
            require(checked == expected, 'TRANSFORM_ROW_COUNT_MISMATCH')
            classes = []
            for key in sorted(aggregates):
                item = aggregates[key]
                item['mean_absolute_ratio_delta'] = item.pop('delta_sum') / item['delta_count'] if item['delta_count'] else None
                item['ratio_comparison_rows'] = item.pop('delta_count')
                classes.append(item)
            review = any(item['failed_region_lost'] or item['valid_region_lost'] for item in classes)
            report = {'status': 'transform_review_required' if review else 'transform_check_passed',
                      'config': config, 'split_protocol_id': split_report['split_protocol_id'],
                      'rows_checked': checked, 'test_maps_read': False, 'training_ready': False,
                      'visual_review_required': True, 'classes': classes}
            save_examples(examples, output)
            write_check_report(report, output)
            (output / 'SUCCESS.json').write_text(json.dumps({'status': 'transform_check_complete',
                'review_required': review, 'training_ready': False}), encoding='utf-8')
        finally:
            database.close()
    return {'status': report['status'], 'training_ready': False}


def main():
    parser = argparse.ArgumentParser(description='Train/Validation 입력 변환·패턴 보존 검사')
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default='configs/wm811k/transform.json')
    parser.add_argument('--canonical-dir', help='이동한 canonical 위치를 명시적으로 연결')
    args = parser.parse_args()
    try:
        with validation_stage('transform_check'):
            result = run_transform_check(args.split_dir, args.output_dir, args.config, args.canonical_dir,
                                         progress=lambda done, total: print(f'변환 검사: {done}/{total}', flush=True))
    except ValidationError as error:
        print('변환 검사 실패 코드: ' + str(error))
        print('완료 표식 없는 결과는 사용하지 마세요. 자동 재실행·덮어쓰기는 하지 않습니다.')
        return 1
    print(result['status'])
    print('transform_check.md의 소실 지표·원본/변환 그림을 로컬에서 확인하세요. Test 맵은 읽지 않았습니다.')
    return 2 if result['status'] == 'transform_review_required' else 0


if __name__ == '__main__':
    raise SystemExit(main())
