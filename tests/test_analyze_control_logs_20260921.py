"""验证统一时间基准、旧实验标定读取及不完整日志处理。"""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('offline_analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze_control_logs_20260921.py')
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)


def write_rows(path, rows):
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n', encoding='utf-8')


@pytest.mark.parametrize('coefficient,expected_depth', [(None,.1),(750,1000/750)])
def test_shared_origin_calibrated_depth_and_missing_sensors(tmp_path, coefficient, expected_depth):
    write_rows(tmp_path/'commands.jsonl', [{'t_ns': 2_000_000_000}])
    write_rows(tmp_path/'events.jsonl', [{'t_ns': 4_000_000_000, 'event':'action_completed'}])
    write_rows(tmp_path/'raw_depth.jsonl', [
        {'t_ns': 1_000_000_000, 'ok':True, 'data':{'pressure_pa':100000}},
        {'t_ns': 3_000_000_000, 'ok':True, 'data':{'pressure_pa':101000}},
        {'t_ns': 4_000_000_000, 'ok':False, 'data':{'pressure_pa':999999}},
    ])
    (tmp_path/'calibration.json').write_text(json.dumps({'p_surface_pa':100000,
        'water_density_kg_m3':1000, 'gravity_mps2':10,
        **({'depth_conversion':'linear_pressure','pressure_per_meter':coefficient} if coefficient else {})}))
    with (tmp_path/'raw_depth.jsonl').open('a') as f:
        f.write('{incomplete')
    report = api.analyze(tmp_path, tmp_path/'output', start=0)
    result = json.loads((report.parent/'summary.json').read_text(encoding='utf-8'))
    assert result['origin_t_ns'] == 2_000_000_000
    assert result['control_end_s'] == 2
    assert result['metrics']['depth.pressure_kpa']['mean'] == 101
    assert result['metrics']['depth.depth_m']['mean'] == pytest.approx(expected_depth)
    assert result['sampling']['depth']['count'] == 1
    assert len(result['warnings']) >= 4
    assert (report.parent/'pressure_depth.png').is_file()
    assert not (report.parent/'power.png').exists()


def test_latest_uses_session_name_and_skips_simulation(tmp_path):
    for name, dry in [('20260920_100000_run',False), ('20260921_100000_run',False), ('20260921_110000_run',True)]:
        p=tmp_path/name
        p.mkdir()
        (p/'raw_depth.jsonl').touch()
        (p/'metadata.yaml').write_text('dry_run: '+str(dry).lower())
    assert api.find_latest(tmp_path).name == '20260921_100000_run'
    assert api.find_latest(tmp_path, True).name == '20260921_110000_run'


def test_no_usable_data_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match='没有有效'):
        api.analyze(tmp_path, tmp_path/'output')


def test_nonfinite_and_missing_vector_fields_are_gaps():
    rows=[{'data':{'acc_mps2':[1,2,3]}}, {'data':{}}, {'data':{'acc_mps2':[float('inf')]}}]
    result=api.stats(api.channel(rows,'acc_mps2',0))
    assert result['valid_count'] == 1
    assert result['missing_count'] == 2
    assert result['mean'] == 1
