import json
import math
from pathlib import Path

import pytest

from control.runtime.uwb_capture_20261013 import capture, export_ranges


PACKET = (b'mc 00 000006ff 00001387 00001cee 000016a9 f6b0 b1 006bed37 t0:0 0000\r\n'
          b'$KT0,1.79,5.00,7.41,5.80,null,null,null,null,LO=[no solution]\r\n')


class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t


class Port:
    def __init__(self, clock, chunks):
        self.clock = clock
        self.chunks = iter(chunks)
        self.timeout = 0.2

    def read_until(self, expected, size):
        assert expected == b'\r\n'
        assert size == 4096
        self.clock.t += self.timeout
        return next(self.chunks, b'')


def test_capture_deadline_and_raw_split_replay(tmp_path):
    clock = Clock()
    port = Port(clock, [PACKET[:30], PACKET[30:], PACKET])
    result = capture(port, .65, tmp_path, clock=clock, clock_ns=lambda: round(clock.t*1e9))
    assert result['reason'] == 'time_limit'
    assert clock.t == pytest.approx(.65)
    rows = [json.loads(s) for s in (tmp_path/'raw_uwb.jsonl').read_text().splitlines()]
    assert bytes.fromhex(rows[0]['data']['raw_hex']) == PACKET[:30]
    points, summary = export_ranges(tmp_path/'raw_uwb.jsonl', tmp_path)
    assert sum(math.isfinite(p[1]) for p in points) == 1  # 拆包拼接，重复mc和KT0不能重复计数。
    assert points[0][1:] == pytest.approx([1.791,4.999,7.406,5.801])
    assert summary['valid_frames'] == 1
    assert (tmp_path/'uwb_ranges.csv').is_file()


def test_no_signal_still_stops_and_exports(tmp_path):
    clock = Clock()
    result = capture(Port(clock, []), .3, tmp_path, clock=clock, clock_ns=lambda: round(clock.t*1e9))
    assert result['reason'] == 'time_limit'
    points, summary = export_ranges(tmp_path/'raw_uwb.jsonl', tmp_path)
    assert summary['valid_frames'] == 0
    assert summary['empty_or_failed_reads'] == 2


def test_serial_failure_keeps_partial_record(tmp_path):
    clock = Clock()
    class BrokenPort(Port):
        def read_until(self, expected, size):
            if self.clock.t:
                raise OSError('disconnected')
            return super().read_until(expected, size)
    result = capture(BrokenPort(clock, [PACKET]), 1., tmp_path, clock=clock, clock_ns=lambda: round(clock.t*1e9))
    assert result['reason'] == 'error' and 'disconnected' in result['error']
    points, _ = export_ranges(tmp_path/'raw_uwb.jsonl', tmp_path)
    assert len(points) == 2 and points[0][1] == pytest.approx(1.791)


def test_keyboard_interrupt_preserves_received_data(tmp_path):
    clock = Clock()
    class InterruptedPort(Port):
        def read_until(self, expected, size):
            if self.clock.t:
                raise KeyboardInterrupt
            return super().read_until(expected, size)
    result = capture(InterruptedPort(clock, [PACKET]), 1., tmp_path, clock=clock, clock_ns=lambda: round(clock.t*1e9))
    assert result['reason'] == 'interrupted' and result['error'] is None
    _, summary = export_ranges(tmp_path/'raw_uwb.jsonl', tmp_path)
    assert summary['valid_frames'] == 1


def test_serial_empty_read_does_not_destroy_split_frame(tmp_path):
    clock = Clock()
    capture(Port(clock, [PACKET[:30], b'', PACKET[30:]]), .6, tmp_path,
            clock=clock, clock_ns=lambda: round(clock.t*1e9))
    _, summary = export_ranges(tmp_path/'raw_uwb.jsonl', tmp_path)
    assert summary['valid_frames'] == 1


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf')])
def test_duration_validation(tmp_path, duration):
    with pytest.raises(ValueError):
        capture(None, duration, tmp_path)


def test_cli_capture_writes_metadata_and_plots(tmp_path, monkeypatch):
    """覆盖实际入口到元数据/CSV/图片的完整链路，只有串口硬件被替换。"""
    import importlib.util
    import sys
    import time
    import types
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('uwb_capture_cli', scripts/'record_uwb_20261013.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    class SerialPort:
        def __init__(self, *args, **kwargs):
            self.timeout = kwargs['timeout']
            self.closed = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.closed = True
        def read_until(self, expected, size):
            assert expected == b'\r\n' and size == 4096
            time.sleep(min(.01, self.timeout))
            return PACKET

    device = SerialPort(timeout=.2)
    monkeypatch.setitem(sys.modules, 'serial', types.SimpleNamespace(Serial=lambda *a, **kw: device))
    config = tmp_path/'robot.yaml'
    config.write_text('{}', encoding='utf-8')
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['record_uwb_20261013.py', '--t', '.04',
                                    '--config', str(config), '--output-dir', str(output)])
    assert cli.main() == 0
    assert device.closed
    session = next(output.iterdir())
    meta = json.loads((session/'metadata.json').read_text(encoding='utf-8'))
    assert meta['requested_duration_s'] == .04 and meta['reason'] == 'time_limit'
    assert json.loads((session/'summary.json').read_text())['valid_frames'] == 1
    for name in ('raw_uwb.jsonl','uwb_ranges.csv','uwb_ranges.png','uwb_ranges.pdf'):
        assert (session/name).stat().st_size > 0
