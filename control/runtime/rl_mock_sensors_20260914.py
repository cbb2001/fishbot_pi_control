"""Fin-only smoke-test sensors with real surface calibration and valid IMU.

This is a deterministic sensor trace, not an action-driven fish dynamics
simulation. Actions cannot change its trajectory, so rewards obtained from
this manager must not be interpreted as evidence of controller learning.
The existing manual-control SensorManager and its mock behavior are unchanged.
"""
from __future__ import annotations

import math
import threading
import time

from .rl_sensor_calibration_20260914 import RLSensorCalibration, quaternion_gravity_body
from .sensor_manager import MockSensorWorker, SENSOR_NAMES, SensorManager, sensor_config


class _RLSensorWorker(MockSensorWorker):
    def __init__(self, manager, name, rate_hz, save_fps):
        super().__init__(name, manager.buffers[name], rate_hz=rate_hz,
                         environment=manager.environment, save_fps=save_fps,
                         stop_event=manager.stop_event)
        self.manager = manager

    def read_once(self):
        if self.name not in ('imu', 'depth'):
            return super().read_once()
        surface, elapsed = self.manager._motion_state()
        calibration = self.manager._convention
        if self.name == 'depth':
            # Starts at the calibrated surface and settles around 0.5 metres.
            depth = 0. if surface else .5 * (1. - math.exp(-3. * elapsed)) + .02 * math.sin(.5 * elapsed)
            return self.make_sample({
                'depth_m': depth,
                'pressure_pa': float(self.manager.config.get('depth_sensor', {}).get('surface_pressure_mbar', 1013.25)) * 100.
                    + depth * calibration.water_density_kg_m3 * calibration.gravity_mps2,
                'temperature_c': 22.,
            })

        # A nonlevel zero also exercises calibration in an arbitrary mounting pose.
        pitch, yaw, roll = -7., 25., 5.
        dp = dy = dr = 0.
        if not surface:
            pitch += .8 * math.sin(.5 * elapsed)
            yaw += 2. * math.sin(.4 * elapsed)
            roll += 1.2 * math.sin(.7 * elapsed)
            dp, dy, dr = (.4 * math.cos(.5 * elapsed), .8 * math.cos(.4 * elapsed),
                          .84 * math.cos(.7 * elapsed))
        p, y, r = (math.radians(v) / 2. for v in (pitch, yaw, roll))
        cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
        quat = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
        if calibration.quaternion_rotation == 'world_to_body':
            quat = (quat[0], -quat[1], -quat[2], -quat[3])
        acceleration = quaternion_gravity_body(quat, calibration.gravity_mps2, **calibration.convention)
        dp, dy, dr = (math.radians(v) for v in (dp, dy, dr))
        gyro = (dr - dy * math.sin(2. * p),
                dp * math.cos(2. * r) + dy * math.sin(2. * r) * math.cos(2. * p),
                -dp * math.sin(2. * r) + dy * math.cos(2. * r) * math.cos(2. * p))
        return self.make_sample({'pitch_deg': pitch, 'yaw_deg': yaw, 'roll_deg': roll,
                                 'quat': list(quat), 'acc_mps2': list(acceleration),
                                 'gyro_radps': list(gyro), 'temperature_c': 25.})


class MockRLSensorManager(SensorManager):
    """Hardware-free manager; leave surface mode only after calibration passes."""
    def __init__(self, config, *, mock=True, log_dir=None, stop_event=None):
        super().__init__(config, mock=True, log_dir=log_dir, stop_event=stop_event)
        self._mode_lock = threading.Lock()
        self._surface_mode = True
        self._motion_start_s = time.monotonic()
        self._convention = RLSensorCalibration(config)

    def set_surface_mode(self, enabled):
        with self._mode_lock:
            enabled = bool(enabled)
            if enabled != self._surface_mode:
                self._motion_start_s = time.monotonic()
            self._surface_mode = enabled

    def _motion_state(self):
        with self._mode_lock:
            return self._surface_mode, max(0., time.monotonic() - self._motion_start_s)

    def _create_mock_workers(self):
        defaults = {'imu': 100., 'depth': 20., 'power': 2., 'uwb': 2., 'vision': 15.}
        for name in SENSOR_NAMES:
            cfg = sensor_config(self.config, name)
            if not bool(cfg.get('enabled', True)):
                continue
            rate_key = 'sample_rate_hz' if name == 'imu' else 'rate_hz'
            self.workers.append(_RLSensorWorker(
                self, name, float(cfg.get(rate_key, defaults[name])), float(cfg.get('save_fps', 5.))))
