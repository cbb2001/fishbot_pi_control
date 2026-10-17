"""沿用导航串口配置的CRLF逐行UWB诊断；不操作任何舵机设备。"""
from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
from collections import deque
from pathlib import Path

from .sensor_manager import SENSOR_NAMES, SensorManager, sensor_config
from .uart_workers import UWBWorker, UWB_LINE_DELIMITER, UWB_MAX_READ_BYTES


# 保留旧worker导入名称；不再维护测试专用读取分支。
UWBLineWorker = UWBWorker


class DiagnosticDecoder:
    """按导航mc字段布局拆帧，但独立检查各路距离，并保留重复/异常报文。"""

    def __init__(self, tag='t0:0'):
        if not re.fullmatch(r't\d+:\d+', tag):
            raise ValueError('tag格式须如t0:0')
        self.tag = tag
        self.pending = b''
        self.seen = deque(maxlen=64)

    def reset_partial(self):
        self.pending = b''

    def feed(self, chunk):
        lines = (self.pending + chunk).split(UWB_LINE_DELIMITER)
        self.pending = lines.pop()
        rows = [self.parse_line(line) for line in lines if line.strip()]
        if len(self.pending) > UWB_MAX_READ_BYTES:
            rows.append(dict(status='line_too_long', raw_hex=self.pending.hex()))
            self.reset_partial()
        return rows

    def parse_line(self, line):
        row = dict(status='other_line', raw_text=line.decode('utf-8', errors='replace'),
                   raw_hex=line.hex())
        if line.startswith(b'$KT'):
            row['status'] = 'display_line'
            return row
        if line.startswith(b'$RANGE_ERROR,'):
            row['status'] = 'range_error'
            return row
        parts = line.strip().split()
        if not parts or parts[0] != b'mc':
            return row
        tag_index = next((i for i, p in enumerate(parts) if re.fullmatch(rb't\d+:\d+', p)), None)
        if tag_index not in (9, 13) or len(parts) != tag_index + 2:
            row.update(status='malformed_mc', error='mc字段数量或标签位置无效')
            return row
        tag = parts[tag_index].decode('ascii')
        row.update(tag=tag, header_field=parts[1].decode('ascii', errors='replace'))
        if tag != self.tag:
            row['status'] = 'other_tag'
            return row
        ranges, errors = [], []
        for token in parts[2:6]:
            if not re.fullmatch(rb'[0-9a-fA-F]{8}', token):
                ranges.append(None)
                errors.append('not_hex8')
                continue
            value = int(token, 16)
            valid = 0 < value < 0x7fffffff
            ranges.append(value / 1000. if valid else None)
            errors.append(None if valid else 'invalid_value')
        identity = b':'.join(parts[tag_index-2:tag_index+1]).decode('ascii', errors='replace')
        duplicate = identity in self.seen
        if not duplicate:
            self.seen.append(identity)
        row.update(status='mc', frame_id=identity, duplicate=duplicate,
                   ranges_m=ranges, range_errors=errors,
                   raw_ranges=[v.decode('ascii', errors='replace') for v in parts[2:6]])
        return row


def run_capture(robot, directory, *, duration_s=None, tag='t0:0', stop_event=None,
                emit=print, show_raw=True):
    """None表示直到停止；CRLF逐行读取，普通超时保留半帧。"""
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError('采集时长必须为正有限秒数')
    decoder = DiagnosticDecoder(tag)
    config = copy.deepcopy(robot)
    config.setdefault('runtime', {})['environment'] = 'surface'
    for name in SENSOR_NAMES:
        config.setdefault('sensors', {}).setdefault(name, {})['enabled'] = name == 'uwb'
    cfg = sensor_config(config, 'uwb')
    timeout = float(cfg.get('serial_timeout_s', .2))
    if not math.isfinite(timeout) or timeout <= 0 or int(cfg['baudrate']) <= 0:
        raise ValueError('UWB serial_timeout_s和baudrate必须为正有限数')
    stop = stop_event or threading.Event()
    sensors = SensorManager(config, mock=False, stop_event=stop)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'metadata.json').write_text(json.dumps(dict(
        uwb=cfg, tag=tag, duration_s=duration_s, acquisition='SensorManager/UWBWorker',
        read_method='read_until', delimiter=UWB_LINE_DELIMITER.decode('ascii'), max_read_bytes=UWB_MAX_READ_BYTES,
        active_servos=[], distance_unit='m', duplicate_frames_retained=True),
        ensure_ascii=False, indent=2), encoding='utf-8')
    result = dict(reason='stopped', raw_chunks=0, received_bytes=0, empty_reads=0,
                  read_errors=0, sequence_gaps=0, a0_valid_frames=0, duplicates=0,
                  malformed_lines=0, device_range_errors=0, display_lines=0, error=None)
    start = time.monotonic()
    last_seq = 0
    latest_a0_time = None
    print_due = start
    emit(f"UWB {cfg['port']} @ {cfg['baudrate']}, timeout={timeout:g}s, tag={tag}; CRLF逐行采集，不操作舵机")

    with (directory / 'raw_uwb.jsonl').open('x', encoding='utf-8', buffering=1) as raw, \
            (directory / 'uwb_diagnostics.jsonl').open('x', encoding='utf-8', buffering=1) as diagnostics:

        def record(row):
            diagnostics.write(json.dumps(row, ensure_ascii=False) + '\n')

        def drain():
            nonlocal last_seq, latest_a0_time
            for sample in sensors.buffers['uwb'].snapshot():
                if sample.seq <= last_seq:
                    continue
                if sample.seq != last_seq + 1:
                    result['sequence_gaps'] += 1
                    decoder.reset_partial()
                    record(dict(status='buffer_gap', after_seq=last_seq, next_seq=sample.seq))
                    emit(f'BUFFER_GAP: {last_seq} -> {sample.seq}')
                last_seq = sample.seq
                raw.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
                result['raw_chunks'] += 1
                context = dict(t_ns=sample.t_ns, seq=sample.seq,
                               elapsed_s=(sample.t_ns / 1e9) - start)
                if not sample.ok:
                    empty = sample.error == 'no_signal_or_timeout'
                    result['empty_reads' if empty else 'read_errors'] += 1
                    # 空读可能落在半帧中间；串口错误/重连则禁止跨连接拼接。
                    if not empty:
                        decoder.reset_partial()
                    record(dict(**context, status='empty_read' if empty else 'read_error', error=sample.error))
                    emit(f"[{context['elapsed_s']:.3f}s] {sample.error}")
                    continue
                chunk = bytes.fromhex(sample.data['raw_hex'])
                result['received_bytes'] += len(chunk)
                if show_raw:
                    emit(f"[{context['elapsed_s']:.3f}s] RX {chunk!r}")
                for row in decoder.feed(chunk):
                    record(dict(**context, **row))
                    if row['status'] == 'mc':
                        result['duplicates'] += int(row['duplicate'])
                        if row['ranges_m'][0] is not None:
                            result['a0_valid_frames'] += 1
                            if not row['duplicate']:
                                latest_a0_time = sample.t_ns / 1e9
                        values = ' '.join(f'A{i}={v:.3f}m' if v is not None else
                                          f"A{i}=INVALID({row['raw_ranges'][i]})"
                                          for i, v in enumerate(row['ranges_m']))
                        emit(f"[{context['elapsed_s']:.3f}s] {values} id={row['frame_id']} duplicate={row['duplicate']}")
                    elif row['status'] == 'display_line':
                        result['display_lines'] += 1
                    elif row['status'] == 'range_error':
                        result['device_range_errors'] += 1
                        emit(f"[{context['elapsed_s']:.3f}s] RANGE_ERROR: {row['raw_text']}")
                    elif row['status'] != 'other_line':
                        result['malformed_lines'] += int(row['status'] in ('malformed_mc', 'line_too_long'))
                        emit(f"[{context['elapsed_s']:.3f}s] {row['status']}: {row.get('raw_text', '')!r}")

        try:
            sensors.start_all()
            while not stop.is_set():
                drain()
                if sensors.fatal_errors():
                    raise RuntimeError(str(sensors.fatal_errors()))
                now = time.monotonic()
                if now >= print_due:
                    age = '尚无有效新帧' if latest_a0_time is None else f'{now-latest_a0_time:.2f}s'
                    emit(f"STATUS {now-start:.1f}s bytes={result['received_bytes']} "
                         f"A0有效帧(含重复)={result['a0_valid_frames']} A0新帧间隔={age} "
                         f"测距错误={result['device_range_errors']}")
                    print_due = now + 1.
                if duration_s is not None and now - start >= duration_s:
                    result['reason'] = 'time_limit'
                    break
                stop.wait(.005)
        except KeyboardInterrupt:
            result['reason'] = 'interrupted'
        except Exception as exc:
            result.update(reason='error', error=f'{type(exc).__name__}: {exc}')
        finally:
            sensors.stop_all(timeout=max(5., timeout + 1.))
            drain()
            if decoder.pending:
                record(dict(status='partial_at_exit', raw_hex=decoder.pending.hex()))
            if sensors.alive_worker_names() or sensors.fatal_errors():
                result.update(reason='error', error=str(sensors.fatal_errors() or sensors.alive_worker_names()))
            result['elapsed_s'] = time.monotonic() - start
            (directory / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
