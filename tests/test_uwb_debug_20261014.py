import json
import sys
import time
from types import SimpleNamespace

import pytest

from control.runtime import uwb_debug_20261014 as debug


PACKET = b'mc 00 000006ff 00000000 ffffffff 7fffffff 4705 01 00000001 t0:0 0000\r\n'


def test_single_anchor_split_and_duplicate_remain_visible():
    decoder = debug.DiagnosticDecoder('t0:0')
    assert decoder.feed(PACKET[:25]) == []
    row = decoder.feed(PACKET[25:])[0]
    assert row['ranges_m'] == [1.791, None, None, None]
    assert row['range_errors'] == [None, 'invalid_value', 'invalid_value', 'invalid_value']
    assert row['duplicate'] is False
    assert decoder.feed(PACKET)[0]['duplicate'] is True


def test_bad_other_anchor_does_not_hide_a0_and_other_tags_are_visible():
    decoder = debug.DiagnosticDecoder('t0:0')
    row = decoder.feed(PACKET.replace(b'ffffffff', b'xxxxxxxx'))[0]
    assert row['ranges_m'][0] == 1.791
    assert row['range_errors'][2] == 'not_hex8'
    assert decoder.feed(PACKET.replace(b't0:0', b't1:0'))[0]['status'] == 'other_tag'
    assert decoder.feed(b'mc bad\r\n')[0]['status'] == 'malformed_mc'
    assert decoder.feed(b'$KT0,1.79,null\r\n')[0]['status'] == 'display_line'


def test_eight_ranges_and_partial_reset():
    decoder = debug.DiagnosticDecoder('t0:0')
    eight = PACKET.replace(b' 4705', b' 00000000 00000000 00000000 00000000 4705')
    assert decoder.feed(eight)[0]['ranges_m'][0] == 1.791
    decoder.feed(PACKET[:20])
    decoder.reset_partial()
    assert decoder.feed(PACKET[20:])[0]['status'] == 'other_line'
    assert decoder.feed(b'x' * 4097)[0]['status'] == 'line_too_long'


def test_real_worker_path_only_opens_uwb_and_preserves_bytes(tmp_path, monkeypatch):
    opened = []

    class Serial:
        def __init__(self, port, baudrate, timeout):
            opened.append((port, baudrate, timeout))
            self.chunks = iter([PACKET[:25], b'', PACKET[25:-1], PACKET[-1:], PACKET])
            self.closed = False
            instances.append(self)

        def read_until(self, expected, size):
            assert expected == b'\r\n'
            assert size == 4096
            time.sleep(.01)
            return next(self.chunks, b'')

        def close(self):
            self.closed = True

    instances = []
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=Serial))
    robot = {'sensors': {'uwb': {'port': 'test-uwb', 'baudrate': 57600,
                                'serial_timeout_s': .03},
                         'imu': {'enabled': True}, 'depth': {'enabled': True}}}
    result = debug.run_capture(robot, tmp_path, duration_s=.15, emit=lambda _: None)
    assert opened == [('test-uwb', 57600, .03)]
    assert all(s.closed for s in instances)
    assert robot['sensors']['imu']['enabled'] is True
    rows = [json.loads(s) for s in (tmp_path / 'raw_uwb.jsonl').read_text().splitlines()]
    assert b''.join(bytes.fromhex(r['data'].get('raw_hex', '')) for r in rows) == PACKET * 2
    assert result['a0_valid_frames'] == 2
    assert result['duplicates'] == 1
    assert result['empty_reads'] >= 1
    assert result['reason'] == 'time_limit'
    assert (tmp_path / 'uwb_diagnostics.jsonl').is_file()


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf')])
def test_invalid_duration_rejected_before_hardware(tmp_path, duration):
    with pytest.raises(ValueError):
        debug.run_capture({}, tmp_path, duration_s=duration)


def test_no_signal_still_finishes_and_logs_timeout(tmp_path, monkeypatch):
    class SilentSerial:
        def __init__(self, *args, **kwargs):
            pass

        def read_until(self, expected, size):
            time.sleep(.01)
            return b''

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=SilentSerial))
    result = debug.run_capture({}, tmp_path, duration_s=.06, emit=lambda _: None)
    assert result['reason'] == 'time_limit'
    assert result['a0_valid_frames'] == result['received_bytes'] == 0
    assert result['empty_reads'] >= 1
    assert result['error'] is None


def test_serial_open_error_saved_and_shutdown_interrupts_retry(tmp_path, monkeypatch):
    def broken_serial(*args, **kwargs):
        raise OSError('port unavailable')

    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=broken_serial))
    result = debug.run_capture({}, tmp_path, duration_s=.06, emit=lambda _: None)
    assert result['read_errors'] == 1
    rows = [json.loads(s) for s in (tmp_path / 'raw_uwb.jsonl').read_text().splitlines()]
    assert rows[0]['ok'] is False
    assert 'port unavailable' in rows[0]['error']
    assert json.loads((tmp_path / 'summary.json').read_text())['read_errors'] == 1


def test_crlf_split_across_reads_and_device_error_classification():
    decoder = debug.DiagnosticDecoder()
    assert decoder.feed(PACKET[:-1]) == []
    rows = decoder.feed(b'\n$KT0,1.79,null\r\n$RANGE_ERROR,ID=0,rb=57\r\n')
    assert [r['status'] for r in rows] == ['mc', 'display_line', 'range_error']
    assert rows[0]['ranges_m'][0] == 1.791
    assert rows[2]['raw_text'] == '$RANGE_ERROR,ID=0,rb=57'


@pytest.mark.parametrize('source', ['debug', 'sensor_manager'])
def test_complete_mc_returns_without_waiting_for_another_frame(source):
    class SerialStream:
        def __init__(self):
            self.data = bytearray(PACKET + b'$KT0,1.79,null\r\n')

        def read_until(self, expected, size):
            assert expected == b'\r\n' and size == 4096
            end = self.data.index(expected) + len(expected)
            chunk = bytes(self.data[:end])
            del self.data[:end]
            return chunk

    from control.runtime.ring_buffer import RingBuffer
    worker = debug.UWBLineWorker(RingBuffer(10))
    if source == 'sensor_manager':
        from control.runtime.sensor_manager import SENSOR_NAMES, SensorManager
        manager = SensorManager({'sensors': {name: {'enabled': name == 'uwb'}
                                             for name in SENSOR_NAMES}}, mock=False)
        manager._create_workers()
        worker, = manager.workers
    worker._serial = SerialStream()
    first = worker.read_once()
    second = worker.read_once()
    assert bytes.fromhex(first.data['raw_hex']) == PACKET
    assert bytes.fromhex(second.data['raw_hex']) == b'$KT0,1.79,null\r\n'
    assert second.seq == first.seq + 1
    assert second.t_ns >= first.t_ns  # 系统时钟分辨率不保证相邻调用严格递增。


def test_device_error_is_reported_even_when_raw_output_hidden(tmp_path, monkeypatch):
    class Serial:
        def __init__(self, *args, **kwargs):
            self.chunks = iter([b'$KT0,1.79,null\r\n', b'$RANGE_ERROR,ID=0,rb=57\r\n', PACKET])

        def read_until(self, expected, size):
            time.sleep(.01)
            return next(self.chunks, b'')

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=Serial))
    output = []
    result = debug.run_capture({}, tmp_path, duration_s=.1, emit=output.append, show_raw=False)
    assert result['device_range_errors'] == result['display_lines'] == 1
    assert result['a0_valid_frames'] == 1
    assert any('RANGE_ERROR,ID=0,rb=57' in line for line in output)
    assert not any(' RX ' in line for line in output)
    rows = [json.loads(s) for s in (tmp_path / 'uwb_diagnostics.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows[:3]] == ['display_line', 'range_error', 'mc']
