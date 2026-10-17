"""10 Hz 压力漂移查表；保留原始缓存，提供供标定和控制读取的修正视图。"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import replace
from pathlib import Path


class PressureDrift:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = json.loads(self.path.read_text(encoding='utf-8-sig'))
        table = self.config['lookup']
        self.start = float(table['start_s'])
        self.end = float(table['end_s'])
        self.step = float(table['step_s'])
        self.values = tuple(float(x) for x in table['pressure_drift_pa'])
        if (self.config['pressure_unit'] != 'Pa' or self.config['time_unit'] != 's'
                or self.config['sample_rate_hz'] != 10
                or self.start != 0 or not math.isfinite(self.end) or self.end <= 0
                or not math.isfinite(self.step) or self.step <= 0
                or len(self.values) < 2 or table['count'] != len(self.values)
                or not math.isclose(self.end, (len(self.values)-1)*self.step)
                or not all(math.isfinite(x) for x in self.values) or self.values[0] != 0):
            raise ValueError('无效的 10 Hz / Pa / s 压力漂移查表配置')
        self.origin_t_ns = None
        self._lock = threading.Lock()

    def observe(self, samples):
        # 第一条有效压力记录只设定一次零点；标定完成、保持姿态均不重置。
        with self._lock:
            if self.origin_t_ns is None:
                for sample in samples:
                    p = sample.data.get('pressure_pa')
                    if sample.ok and isinstance(p, (int, float)) and math.isfinite(p) and p > 0:
                        self.origin_t_ns = sample.t_ns
                        break

    def drift_pa(self, elapsed_s):
        if not math.isfinite(elapsed_s) or not 0 <= elapsed_s <= self.end:
            raise ValueError(f'压力漂移时间 {elapsed_s} s 超出查表范围 [0, {self.end}]')
        index = elapsed_s/self.step
        left = min(int(index), len(self.values)-1)
        right = min(left+1, len(self.values)-1)
        return self.values[left] + (index-left)*(self.values[right]-self.values[left])

    def elapsed_s(self, t_ns):
        if self.origin_t_ns is None:
            raise ValueError('尚无有效压力样本，无法确定漂移计时起点')
        return (t_ns-self.origin_t_ns)/1e9

    def correct(self, sample):
        if not sample.ok:
            return sample
        try:
            pressure = float(sample.data['pressure_pa'])
            if not math.isfinite(pressure) or pressure <= 0:
                raise ValueError('pressure must be positive and finite')
        except (KeyError, ValueError, TypeError):
            return replace(sample, ok=False, error='invalid raw pressure')
        elapsed = self.elapsed_s(sample.t_ns)
        drift = self.drift_pa(elapsed)
        return replace(sample, data={**sample.data,
            'raw_pressure_pa': pressure, 'pressure_pa': pressure-drift,
            'pressure_drift_pa': drift, 'drift_elapsed_s': elapsed})


class CorrectedDepthBuffer:
    def __init__(self, buffer, drift):
        self.buffer, self.drift = buffer, drift
        self.calibration = None

    def _correct(self, sample):
        if sample is None:
            return None
        result = self.drift.correct(sample)
        if self.calibration is not None and result.ok:
            result = replace(result, data={**result.data,
                'depth_m': self.calibration.depth_m(result.data['pressure_pa'])})
        return result

    def snapshot(self):
        rows = self.buffer.snapshot()
        self.drift.observe(rows)
        return [self._correct(s) for s in rows]

    def latest(self):
        # 首次读取从缓存最早有效样本建立起点，不能用最后一条初始化。
        if self.drift.origin_t_ns is None:
            self.drift.observe(self.buffer.snapshot())
        row = self.buffer.latest()
        return self._correct(row)

    def get_latest_before(self, t_ns):
        # 同步状态只能使用截止时刻之前的压力，并按该压力的采样时刻修正。
        if self.drift.origin_t_ns is None:
            self.drift.observe(self.buffer.snapshot())
        return self._correct(self.buffer.get_latest_before(t_ns))


class ObservedPressureBuffer:
    """在写入原始缓存时锁定首样本，避免等待较久后首样本被环形缓存淘汰。"""
    def __init__(self, buffer, drift):
        self.buffer, self.drift = buffer, drift
        self.failure = None

    def append(self, sample):
        if not sample.ok and self.drift.origin_t_ns is not None:
            # 读取失败可能导致底层重置；禁止在重置后静默沿用旧计时与P0。
            self.failure = f'压力连续采样中断，请重新启动并标定: {sample.error}'
        self.drift.observe([sample])
        self.buffer.append(sample)

    def latest(self):
        sample = self.buffer.latest()
        if sample is not None and self.failure:
            return replace(sample, ok=False, error=self.failure)
        return sample

    def __getattr__(self, name):
        return getattr(self.buffer, name)


class CorrectedSensors:
    def __init__(self, sensors, drift):
        self.sensors = sensors
        self.buffers = {**sensors.buffers,
                        'depth': CorrectedDepthBuffer(sensors.buffers['depth'], drift)}

    def __getattr__(self, name):
        return getattr(self.sensors, name)
