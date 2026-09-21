"""在线训练健康监测；本线程只请求停车，PWM 仍由执行器线程独占。"""
from __future__ import annotations

import math
import queue
import threading
import time


class RLSafetyFault(RuntimeError):
    def __init__(self, code, message):
        self.code = str(code)
        super().__init__(f'{code}: {message}')


class RLSafetyMonitor:
    def __init__(self, sensors, executor, failures, stop_event, config, robot_config,
                 *, clock_ns=time.monotonic_ns):
        self.sensors, self.executor, self.failures = sensors, executor, failures
        self.stop_event, self.clock_ns = stop_event, clock_ns
        self.config = config
        self.enforce_data_freshness = config.get('state', {}).get('enforce_data_freshness', True)
        self.latest_data_age_ms = {}
        self.data_age_warnings = []
        self.limits = {**robot_config.get('safety', {}), **config.get('safety', {})}
        for name in ('min_voltage_v', 'max_current_a', 'max_depth_m'):
            value = float(self.limits.get(name, float('nan')))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'safety.{name} 必须是正有限数')
            self.limits[name] = value
        self.period = float(config.get('runtime', {}).get('safety_poll_interval_s', .01))
        self.grace_s = float(config.get('runtime', {}).get('sensor_ready_timeout_s', 5))
        self.max_state_age_ms = float(config.get('state', {}).get('max_state_age_ms', 150))
        if not all(math.isfinite(x) and x > 0 for x in (self.period, self.grace_s, self.max_state_age_ms)):
            raise ValueError('安全检查周期、就绪超时和状态有效期必须为正有限数')
        self.age_limits = {}
        for name, default in (('imu', 100), ('depth', 300), ('power', 2000)):
            base = robot_config.get('sensors', {}).get(name, {}).get('timeout_ms', default)
            age = float(config.get('state', {}).get('sensor_age', {}).get(name, {}).get('fault_ms', base))
            if not math.isfinite(age) or age <= 0:
                raise ValueError(f'{name} fault_ms 非法')
            self.age_limits[name] = age
        self.started_ns = clock_ns()
        self.calibration = self.statebuf = self.sync = None
        self.executor_started = False
        self.failure = None
        self._halt = threading.Event()
        self._thread = None

    @staticmethod
    def _number(data, key, code):
        try:
            value = float(data[key])
            if not math.isfinite(value):
                raise ValueError('nonfinite')
            return value
        except (KeyError, TypeError, ValueError) as exc:
            raise RLSafetyFault(code, f'{key} 缺失或非有限数') from exc

    def check_once(self):
        # 先保留线程上报的具体原因，再处理公共停止信号。
        try:
            fault = self.failures.get_nowait()
        except queue.Empty:
            pass
        else:
            raise RLSafetyFault(fault.get('fault_code', 'WORKER_FAILED'), fault.get('message', str(fault)))
        if self.executor.failure is not None:
            raise RLSafetyFault('SERVO_FAILED', str(self.executor.failure))
        if self.sync is not None:
            self.sync.raise_if_failed()
            if not self.sync.is_alive():
                raise RLSafetyFault('STATE_WORKER_STOPPED', 'RL 状态采样线程已停止')
        fatal = self.sensors.fatal_errors()
        if fatal:
            raise RLSafetyFault('SENSOR_WORKER_FAILED', str(fatal))
        if self.executor_started and not self.executor.is_alive():
            raise RLSafetyFault('SERVO_WORKER_STOPPED', '舵机执行器意外退出')
        if self.stop_event.is_set():
            raise RLSafetyFault('STOP_REQUESTED', '收到停止请求')
        now = self.clock_ns()
        data = {}
        ages, age_warnings = {}, []
        for name in ('imu', 'depth', 'power'):
            sample = self.sensors.buffers[name].latest()
            if sample is None:
                # 启动时只宽限尚未到达的首个样本，已报告的错误立即处理。
                if now - self.started_ns < self.grace_s * 1e9:
                    continue
                raise RLSafetyFault(name.upper() + '_MISSING', '未收到首个样本')
            age_ms = (self.clock_ns() - sample.t_ns) / 1e6
            ages[name] = age_ms
            if age_ms > self.age_limits[name]:
                age_warnings.append(name)
            # 关闭数据年龄拦截后沿用最近有效样本，未来时间戳仍属非法数据。
            if age_ms < 0 or (self.enforce_data_freshness and age_ms > self.age_limits[name]):
                raise RLSafetyFault(name.upper() + '_STALE', f'样本年龄 {age_ms:.1f} ms')
            if not sample.ok:
                raise RLSafetyFault(name.upper() + '_INVALID', str(sample.error))
            data[name] = sample.data
        if 'power' in data:
            voltage = self._number(data['power'], 'voltage_v', 'POWER_INVALID')
            current = self._number(data['power'], 'current_a', 'POWER_INVALID')
            self._number(data['power'], 'power_w', 'POWER_INVALID')
            if voltage < self.limits['min_voltage_v']:
                raise RLSafetyFault('LOW_VOLTAGE', f'{voltage:.3f} V')
            if abs(current) > self.limits['max_current_a']:
                raise RLSafetyFault('OVERCURRENT', f'{current:.3f} A')
        if 'depth' in data:
            pressure = self._number(data['depth'], 'pressure_pa', 'DEPTH_INVALID')
            if pressure <= 0:
                raise RLSafetyFault('DEPTH_INVALID', '绝对压力必须为正')
            if self.calibration is not None:
                # 必须使用本次标定零点，不能读取旧驱动固定基准的 depth_m。
                depth = self.calibration.depth_m(pressure)
                if not math.isfinite(depth):
                    raise RLSafetyFault('DEPTH_INVALID', '标定后深度非有限数')
                if depth > self.limits['max_depth_m']:
                    raise RLSafetyFault('OVERDEPTH', f'{depth:.3f} m')
        if self.statebuf is not None:
            latest = self.statebuf.latest()
            age = None if latest is None else (self.clock_ns() - latest.t_ns) / 1e6
            ages['rl_state'] = age
            if age is not None and age > self.max_state_age_ms:
                age_warnings.append('rl_state')
            if age is None or age < 0 or (self.enforce_data_freshness and age > self.max_state_age_ms):
                raise RLSafetyFault('RL_STATE_STALE', f'状态年龄 {age} ms，限制 {self.max_state_age_ms} ms')
        self.latest_data_age_ms = ages
        self.data_age_warnings = age_warnings

    def _run(self):
        while not self._halt.is_set():
            try:
                self.check_once()
            except BaseException as exc:
                self.failure = exc
                self.executor.request_emergency_stop()
                self.stop_event.set()
                return
            self._halt.wait(self.period)

    def start(self):
        self.started_ns = self.clock_ns()
        self._thread = threading.Thread(target=self._run, name='rl-safety-monitor', daemon=True)
        self._thread.start()

    def raise_if_failed(self):
        if self.failure is not None:
            raise self.failure

    def stop(self, timeout=5.):
        self._halt.set()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError('安全监测线程未按时退出')
