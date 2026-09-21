"""Causal, validated 15-component state snapshots for online RL."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import copy
import time
import numpy as np

from .sensor_synchronizer import SensorSynchronizer
from .rl_sensor_calibration_20260914 import RLSensorCalibration, finite_float

CRITICAL_SENSORS = ('imu', 'depth', 'power')
DEFAULT_FAULT_AGE_MS = {'imu': 100., 'depth': 300., 'power': 2000.}
OBSERVATION_NAMES = ('pitch_deg', 'yaw_deg', 'roll_deg', 'a1', 'a2', 'a3',
                     'depth_m', 'power_w', *(f'servo{i}_estimated_deg' for i in range(1, 8)))


class RLStateError(RuntimeError):
    def __init__(self, message, *, sensor=None, fault_code='INVALID_RL_STATE', diagnostics=None):
        super().__init__(message)
        self.sensor, self.fault_code = sensor, fault_code
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class RLState:
    t_ns: int
    observation: np.ndarray
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    linear_acc_mps2: tuple[float, float, float]
    raw_acc_mps2: tuple[float, float, float]
    raw_angles_deg: dict[str, float]
    raw_quaternion_wxyz: tuple[float, float, float, float]
    depth_m: float
    pressure_pa: float
    power_w: float
    servo_estimated_angles_deg: dict[int, float]
    servo_pose: dict[str, Any]
    sensor_age_ms: dict[str, float]
    sensor_sample_t_ns: dict[str, int]
    warnings: tuple[str, ...] = ()
    reward: Any = None

    @property
    def obs(self):
        return self.observation

    @property
    def h(self):
        return self.depth_m

    @property
    def reward_total(self):
        return None if self.reward is None else self.reward.reward

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.observation, dtype=dtype)

    def to_dict(self):
        result = {name: getattr(self, name) for name in self.__dataclass_fields__
                  if name not in ('observation', 'reward')}
        result['observation'] = self.observation.tolist()
        if self.reward is not None:
            result['reward'] = self.reward.as_dict()
            result.update(self.reward.as_dict())
        return result


class RLStateBuilder:
    """Use one SensorSynchronizer snapshot per query; reject unknown values.

    Thresholds default to the sensor's robot.yaml timeout. ``state.sensor_age``
    can provide per-sensor ``warn_ms`` and ``fault_ms``. Warnings keep a fresh
    finite sample usable. With enforce_data_freshness=False, old valid samples
    remain usable with their original timestamps and age warnings; missing or
    invalid data never produces a substitute value.
    """
    OBSERVATION_DIM = 15

    def __init__(self, sensors, calibration, servo_state_tracker, config=None, *,
                 reward=None, max_sensor_age_ms=None):
        self.sensors, self.calibration, self.servo_state_tracker = sensors, calibration, servo_state_tracker
        self.config = dict(config or getattr(sensors, 'config', {}))
        self.reward_calculator = reward
        self.last_state = None
        self.last_sensor_age = {}
        self.last_diagnostics = {}
        self.last_synchronized_sample = None
        self.age_limits = {}
        state_cfg = self.config.get('state', {})
        self.enforce_data_freshness = state_cfg.get('enforce_data_freshness', True)
        sync_config = copy.deepcopy(self.config)
        sync_config.setdefault('sensors', {})
        for name in CRITICAL_SENSORS:
            base = self.config.get('sensors', {}).get(name, {}).get('timeout_ms', DEFAULT_FAULT_AGE_MS[name])
            override = state_cfg.get('sensor_age', {}).get(name, {})
            fault = finite_float((max_sensor_age_ms or {}).get(name, override.get('fault_ms', base)), f'{name}.fault_ms')
            warn = finite_float(override.get('warn_ms', fault * .75), f'{name}.warn_ms')
            if fault <= 0 or not 0 <= warn <= fault:
                raise ValueError(f'{name}: sensor age thresholds must satisfy 0 <= warn <= fault')
            self.age_limits[name] = warn, fault
            sync_config['sensors'].setdefault(name, {})['timeout_ms'] = fault
        if hasattr(sensors, 'buffers'):
            self.synchronizer = SensorSynchronizer(sensors.buffers, sync_config,
                enforce_data_freshness=self.enforce_data_freshness)
        elif isinstance(sensors, Mapping) and all(hasattr(v, 'get_latest_before') for v in sensors.values()):
            self.synchronizer = SensorSynchronizer(dict(sensors), sync_config,
                enforce_data_freshness=self.enforce_data_freshness)
        elif hasattr(sensors, 'build'):
            self.synchronizer = sensors
        else:
            self.synchronizer = None

    def _snapshot(self, t_ns):
        if self.synchronizer is not None:
            return self.synchronizer.build(t_ns)
        if isinstance(self.sensors, Mapping):
            return self.sensors
        raise RLStateError('SensorManager/SensorSynchronizer unavailable')

    def _sensor(self, snapshot, name, t_ns):
        field = snapshot.get(name)
        if field is None:
            raise RLStateError(f'{name}: missing sensor', sensor=name, fault_code=f'{name.upper()}_MISSING')
        if not isinstance(field, Mapping):
            field = {'data': field.data, 'sample_t_ns': field.t_ns,
                     'valid': field.ok, 'error': field.error}
        st = field.get('sample_t_ns', field.get('t_ns'))
        if st is None:
            raise RLStateError(f'{name}: sample timestamp missing', sensor=name, fault_code=f'{name.upper()}_MISSING')
        st = int(st)
        age = (t_ns-st)/1e6
        self.last_sensor_age[name] = age
        if st > t_ns:
            raise RLStateError(f'{name}: future sample {st} > {t_ns}', sensor=name, fault_code=f'{name.upper()}_FUTURE')
        warn, fault = self.age_limits[name]
        if self.enforce_data_freshness and age > fault:
            raise RLStateError(f'{name}: stale sample ({age:.2f} ms > {fault:.2f} ms)', sensor=name,
                               fault_code=f'{name.upper()}_STALE', diagnostics={'sensor_age_ms': dict(self.last_sensor_age)})
        if field.get('valid', field.get('ok', True)) is False:
            raise RLStateError(f'{name}: invalid sample ({field.get("error")})', sensor=name,
                               fault_code=f'{name.upper()}_INVALID')
        data = field.get('data', field)
        if not isinstance(data, Mapping):
            raise RLStateError(f'{name}: data missing', sensor=name)
        return data, st, age, f'{name}: age {age:.2f} ms' if age > warn else None

    @staticmethod
    def _required(data, key, sensor):
        if key not in data:
            raise RLStateError(f'{sensor}: missing {key}', sensor=sensor, fault_code=f'{sensor.upper()}_INVALID')
        try:
            return finite_float(data[key], f'{sensor}.{key}')
        except ValueError as exc:
            raise RLStateError(str(exc), sensor=sensor, fault_code=f'{sensor.upper()}_NONFINITE') from exc

    def build(self, t_ns=None, *, reward=None):
        t_ns = int(time.monotonic_ns() if t_ns is None else t_ns)
        if not self.calibration.ready:
            raise RLStateError('IMU/gravity/current-session surface calibration is not ready', fault_code='CALIBRATION_REQUIRED')
        self.last_sensor_age = {}
        snapshot = self._snapshot(t_ns)
        self.last_synchronized_sample = snapshot
        data, timestamps, ages, warnings = {}, {}, {}, []
        for name in CRITICAL_SENSORS:
            data[name], timestamps[name], ages[name], warning = self._sensor(snapshot, name, t_ns)
            if warning:
                warnings.append(warning)
        imu = data['imu']
        raw_angles = {key: self._required(imu, key, 'imu') for key in ('pitch_deg','yaw_deg','roll_deg')}
        pitch,yaw,roll = self.calibration.angles(*(raw_angles[k] for k in ('pitch_deg','yaw_deg','roll_deg')))
        try:
            # Always compute from measured acceleration. A decoder-supplied
            # linear acceleration cannot bypass the verified convention.
            linear_acc = self.calibration.linear_acceleration(imu)
            self.calibration.validate_orientation(imu)
            raw_acc = tuple(finite_float(v, 'imu.acc_mps2') for v in imu['acc_mps2'])
            quat = tuple(finite_float(v, 'imu.quat') for v in imu['quat'])
        except (ValueError, KeyError, TypeError) as exc:
            raise RLStateError(f'imu: {exc}', sensor='imu', fault_code='IMU_INVALID') from exc
        pressure = self._required(data['depth'], 'pressure_pa', 'depth')
        if pressure <= 0:
            raise RLStateError('depth: absolute pressure must be positive', sensor='depth', fault_code='DEPTH_INVALID')
        depth = self.calibration.depth_m(pressure)
        power = self._required(data['power'], 'power_w', 'power')
        if self.servo_state_tracker is None:
            raise RLStateError('servo state tracker missing', fault_code='SERVO_STATE_MISSING')
        pose = self.servo_state_tracker.get_servo_pose_at(t_ns)
        pose = pose.to_dict() if hasattr(pose, 'to_dict') else pose
        if not isinstance(pose, Mapping):
            raise RLStateError('servo pose must expose a mapping', fault_code='SERVO_STATE_INVALID')
        angles = pose.get('estimated_angles_deg', pose.get('estimated_angle_deg'))
        if not isinstance(angles, Mapping):
            raise RLStateError('servo estimated angles missing', fault_code='SERVO_STATE_MISSING')
        servos = {}
        for sid in range(1,8):
            value = angles.get(sid, angles.get(str(sid)))
            try:
                servos[sid] = finite_float(value, f'servo{sid}_estimated')
            except ValueError as exc:
                raise RLStateError(str(exc), fault_code='SERVO_STATE_INVALID') from exc
        observation = np.asarray([pitch,yaw,roll,*linear_acc,depth,power,*servos.values()], dtype=np.float32)
        if observation.shape != (15,) or not np.isfinite(observation).all():
            raise RLStateError('observation contains non-finite values', fault_code='OBSERVATION_NONFINITE')
        observation.setflags(write=False)
        calculator = reward if reward is not None else self.reward_calculator
        reward_result = None if calculator is None else calculator.compute(
            {'pitch_deg':pitch, 'yaw_deg':yaw, 'roll_deg':roll, 'depth_m':depth})
        result = RLState(t_ns, observation, pitch, yaw, roll, linear_acc, raw_acc, raw_angles, quat,
                         depth, pressure, power, servos, dict(pose), ages, timestamps,
                         tuple(warnings), reward_result)
        self.last_state = result
        self.last_diagnostics = result.to_dict()
        return result

    def build_observation(self, t_ns=None, **kwargs):
        return self.build(t_ns, **kwargs).observation.copy()

    build_state = build

# Public convenience alias; implementation lives beside the state RingBuffer.
from .rl_state_buffer_20260914 import RLStateSyncWorker
