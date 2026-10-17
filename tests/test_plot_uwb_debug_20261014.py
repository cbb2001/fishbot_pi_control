import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'plot_uwb_debug_20261014.py'


def load_module():
    assert SCRIPT.is_file(), '绘图脚本尚未实现'
    spec = importlib.util.spec_from_file_location('plot_uwb_debug_20261014', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(path):
    rows = [
        dict(status='mc', elapsed_s=.1, frame_id='01:x:t0:0',
             ranges_m=[1., 2., 3., 4.], range_errors=[None] * 4),
        dict(status='range_error', elapsed_s=.15, raw_text='$RANGE_ERROR,ID=0'),
        dict(status='display_line', elapsed_s=.16, raw_text='$KT0,1,2,3,4'),
        dict(status='mc', elapsed_s=.2, frame_id='02:x:t0:0',
             ranges_m=[1.2, None, 3.2, 4.2],
             range_errors=[None, 'invalid_value', None, None]),
        dict(status='mc', elapsed_s=.3, frame_id='03:x:t0:0',
             ranges_m=[.8, 2.2, None, 3.8],
             range_errors=[None, None, 'invalid_value', None]),
    ]
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')


def test_loads_directory_and_preserves_each_anchor_gap(tmp_path):
    api = load_module()
    source = tmp_path / 'run'
    source.mkdir()
    write_fixture(source / 'uwb_diagnostics.jsonl')
    points, error_times, resolved = api.load_diagnostics(source)
    assert resolved == source / 'uwb_diagnostics.jsonl'
    assert [p['elapsed_s'] for p in points] == [.1, .2, .3]
    assert points[1]['ranges_m'] == [1.2, None, 3.2, 4.2]
    assert points[2]['ranges_m'] == [.8, 2.2, None, 3.8]
    assert error_times == [.15]


def test_summary_is_per_anchor_and_uses_population_std(tmp_path):
    api = load_module()
    source = tmp_path / 'uwb_diagnostics.jsonl'
    write_fixture(source)
    points, error_times, _ = api.load_diagnostics(source)
    summary = api.summarize(points, error_times, source)
    assert summary['measurement_frames'] == 3
    assert summary['device_range_errors'] == 1
    assert summary['duration_s'] == pytest.approx(.2)
    assert summary['ranges'][0] == pytest.approx(dict(
        valid_frames=3, invalid_frames=0, valid_percent=100., min_m=.8,
        max_m=1.2, mean_m=1., median_m=1., std_m=math.sqrt(0.08 / 3)))
    assert summary['ranges'][1]['valid_frames'] == 2
    assert summary['ranges'][1]['invalid_frames'] == 1
    assert summary['ranges'][2]['invalid_frames'] == 1


def test_exports_csv_with_blank_invalid_cells(tmp_path):
    api = load_module()
    source = tmp_path / 'uwb_diagnostics.jsonl'
    write_fixture(source)
    points, _, _ = api.load_diagnostics(source)
    output = tmp_path / 'output'
    api.export_csv(points, output)
    with (output / 'uwb_ranges.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ['elapsed_s', 'frame_id', 'A0_m', 'A1_m', 'A2_m', 'A3_m']
    assert rows[2] == ['0.2', '02:x:t0:0', '1.2', '', '3.2', '4.2']


def test_rejects_missing_or_empty_diagnostics(tmp_path):
    api = load_module()
    with pytest.raises(FileNotFoundError):
        api.load_diagnostics(tmp_path)
    empty = tmp_path / 'uwb_diagnostics.jsonl'
    empty.write_text(json.dumps(dict(status='display_line', elapsed_s=0)) + '\n')
    with pytest.raises(ValueError, match='没有mc测距帧'):
        api.load_diagnostics(empty)
