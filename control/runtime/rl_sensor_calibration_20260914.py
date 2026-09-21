"""Session-local IMU/surface calibration with checked gravity compensation.

Yesense's existing decoder exposes scalar-first [q0,q1,q2,q3]=[w,x,y,z]
and measured acceleration in m/s² including gravity. The configured rotation
is checked against raw Euler tilt and static acceleration before control.
Calibrated Euler angles must never be used to remove gravity.
"""
from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Mapping


class CalibrationError(ValueError):
    """A moving or incomplete calibration window cannot define the zero."""


def finite_float(value, name='value'):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def wrap180(angle):
    return ((finite_float(angle, 'angle')+180.) % 360.)-180.


def circular_mean_deg(values):
    vals = [finite_float(v, 'angle') for v in values]
    if not vals:
        raise CalibrationError('angles empty')
    sine = statistics.fmean(math.sin(math.radians(v)) for v in vals)
    cosine = statistics.fmean(math.cos(math.radians(v)) for v in vals)
    if math.hypot(sine,cosine) < 1e-8:
        raise CalibrationError('angles have no unique circular mean')
    return wrap180(math.degrees(math.atan2(sine,cosine)))


def _vec(values, length, name):
    try:
        result = tuple(finite_float(v,name) for v in values)
    except TypeError as exc:
        raise ValueError(f'{name} requires {length} values') from exc
    if len(result) != length:
        raise ValueError(f'{name} requires {length} values')
    return result


def quaternion_gravity_body(quat, gravity=9.80665, *,
                            quaternion_rotation='body_to_world', gravity_world_z_sign=1):
    """R(q)^T [0,0,+g] for body->world, or R(q) for world->body.

    The sign is the *measured static accelerometer* direction in world axes.
    It is not the sign of the physical gravitational force.
    """
    w,x,y,z = _vec(quat,4,'quat [w,x,y,z]')
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if not .8 <= norm <= 1.2:
        raise ValueError(f'quaternion norm invalid: {norm}')
    if quaternion_rotation not in ('body_to_world','world_to_body'):
        raise ValueError('unsupported quaternion convention')
    if gravity_world_z_sign not in (-1,1):
        raise ValueError('gravity_world_z_sign must be +/-1')
    w,x,y,z = (v/norm for v in (w,x,y,z))
    if quaternion_rotation == 'world_to_body':
        x,y,z = -x,-y,-z
    g = finite_float(gravity, 'gravity')
    if g <= 0:
        raise ValueError('gravity must be positive')
    g *= gravity_world_z_sign
    return 2*(x*z-w*y)*g, 2*(y*z+w*x)*g, (1-2*(x*x+y*y))*g


def remove_gravity(acc_mps2, quat=None, gravity=9.80665, **convention):
    if quat is None:
        raise ValueError('gravity removal requires the actual IMU quaternion')
    acc = _vec(acc_mps2,3,'acc_mps2')
    g = quaternion_gravity_body(quat,gravity,**convention)
    return tuple(a-b for a,b in zip(acc,g))


def _data(sample):
    if isinstance(sample,Mapping):
        valid = sample.get('ok',sample.get('valid',True))
        data = sample.get('data',sample)
    else:
        valid = getattr(sample,'ok',True)
        data = getattr(sample,'data',None)
    if not valid or not isinstance(data,Mapping):
        raise CalibrationError('invalid calibration sample')
    return data


def pressure_to_depth(pressure_pa, p_surface_pa, rho_water=1000., gravity=9.80665):
    rho,g = finite_float(rho_water,'rho_water'), finite_float(gravity,'gravity')
    if rho <= 0 or g <= 0:
        raise ValueError('rho/gravity must be positive')
    return (finite_float(pressure_pa,'pressure_pa')-finite_float(p_surface_pa,'p_surface_pa'))/(rho*g)


class RLSensorCalibration:
    def __init__(self, config=None, **overrides):
        root = dict(config or {})
        self.config = dict(root.get('calibration',root))
        self.config.update(overrides)
        c = self.config
        self.minimum_imu_samples = int(c.get('minimum_imu_samples',200))
        self.minimum_depth_samples = int(c.get('minimum_depth_samples',50))
        if self.minimum_imu_samples <= 0 or self.minimum_depth_samples <= 0:
            raise ValueError('minimum calibration sample counts must be positive')
        for key,default in (('max_angle_std_deg',1.), ('max_pressure_std_pa',100.),
                            ('max_linear_acc_rms_mps2',.5), ('max_acc_std_mps2',.25),
                            ('gravity_mps2',9.80665), ('water_density_kg_m3',1000.),
                            ('max_gyro_radps',.15), ('max_quat_euler_tilt_error_deg',5.)):
            value = finite_float(c.get(key,default),key)
            if value <= 0:
                raise ValueError(f'{key} must be positive')
            setattr(self,key,value)
        self.quaternion_rotation = c.get('quaternion_rotation','body_to_world')
        self.gravity_world_z_sign = c.get('gravity_world_z_sign',1)
        quaternion_gravity_body((1,0,0,0),self.gravity_mps2,**self.convention)
        self.pitch_zero_deg = self.yaw_zero_deg = self.roll_zero_deg = 0.
        self.p_surface_pa = None
        self.imu_calibrated = self.depth_calibrated = self.gravity_validated = False
        self.diagnostics = {}
        self.fixed_surface_pressure_mbar = c.get('surface_pressure_mbar')
        if 'surface_pressure_mbar' in c:
            value = finite_float(self.fixed_surface_pressure_mbar, 'surface_pressure_mbar')
            if isinstance(self.fixed_surface_pressure_mbar, bool) or value <= 0:
                raise ValueError('surface_pressure_mbar must be positive')
            self.fixed_surface_pressure_mbar = value
            # 固定参考模式仅转换单位，不用启动时的压力样本覆盖配置零点。
            self.p_surface_pa = finite_float(value * 100., 'surface pressure in Pa')
            self.depth_calibrated = True  # 表示深度换算就绪；来源另记为 fixed_config。
            self.diagnostics.update(depth_sample_count=0, pressure_std_pa=None)

    @property
    def convention(self):
        return {'quaternion_rotation':self.quaternion_rotation,
                'gravity_world_z_sign':self.gravity_world_z_sign}

    @property
    def ready(self):
        return self.imu_calibrated and self.depth_calibrated and self.gravity_validated

    def linear_acceleration(self, data):
        if 'acc_mps2' not in data or 'quat' not in data:
            raise CalibrationError('IMU requires acc_mps2 and quat [w,x,y,z]')
        return remove_gravity(data['acc_mps2'],data['quat'],self.gravity_mps2,**self.convention)

    def validate_orientation(self, data):
        """Check quaternion/Euler tilt in each frame, without imposing level pitch/roll."""
        qg = quaternion_gravity_body(data['quat'],1.,quaternion_rotation=self.quaternion_rotation)
        pitch = math.radians(finite_float(data['pitch_deg'],'pitch_deg'))
        roll = math.radians(finite_float(data['roll_deg'],'roll_deg'))
        eg = (-math.sin(pitch),math.sin(roll)*math.cos(pitch),math.cos(roll)*math.cos(pitch))
        dot = max(-1.,min(1.,sum(a*b for a,b in zip(qg,eg))))
        error = math.degrees(math.acos(dot))
        if error > self.max_quat_euler_tilt_error_deg:
            raise CalibrationError(f'raw Euler/quaternion tilt mismatch {error:.2f} deg; verify IMU axes')
        return error

    def calibrate_imu(self, samples):
        rows = [_data(s) for s in samples]
        if len(rows) < self.minimum_imu_samples:
            raise CalibrationError(f'insufficient IMU samples: {len(rows)}/{self.minimum_imu_samples}')
        means,stds = {},{}
        for key in ('pitch_deg','yaw_deg','roll_deg'):
            if any(key not in row for row in rows):
                raise CalibrationError(f'IMU missing {key}')
            vals = [finite_float(row[key],key) for row in rows]
            means[key] = circular_mean_deg(vals)
            stds[key] = math.sqrt(statistics.fmean(wrap180(v-means[key])**2 for v in vals))
            if stds[key] > self.max_angle_std_deg:
                raise CalibrationError(f'{key} unstable: std={stds[key]:.3f} deg')
        residual = [self.linear_acceleration(row) for row in rows]
        rms = math.sqrt(statistics.fmean(sum(v*v for v in r) for r in residual))
        if rms > self.max_linear_acc_rms_mps2:
            raise CalibrationError(f'static gravity residual {rms:.3f} m/s2: moving fish or invalid coordinate convention')
        acc = [_vec(row['acc_mps2'],3,'acc_mps2') for row in rows]
        acc_std = [statistics.pstdev(a[k] for a in acc) for k in range(3)]
        if max(acc_std) > self.max_acc_std_mps2:
            raise CalibrationError('acceleration unstable: fish still moving')
        max_tilt_error = 0.
        for row in rows:
            if 'gyro_radps' in row and max(abs(v) for v in _vec(row['gyro_radps'],3,'gyro_radps')) > self.max_gyro_radps:
                raise CalibrationError('angular velocity too high during calibration')
            max_tilt_error = max(max_tilt_error,self.validate_orientation(row))
        # Do not publish the new zero until every stability/coordinate check passes.
        self.pitch_zero_deg,self.yaw_zero_deg,self.roll_zero_deg = (means[k] for k in ('pitch_deg','yaw_deg','roll_deg'))
        self.imu_calibrated = self.gravity_validated = True
        self.diagnostics.update(imu_sample_count=len(rows),angle_std_deg=stds,
                                linear_acc_rms_mps2=rms,acceleration_std_mps2=acc_std,
                                max_quat_euler_tilt_error_deg=max_tilt_error)
        return self.state_dict()

    def calibrate_depth(self, samples):
        if self.fixed_surface_pressure_mbar is not None:
            return self.p_surface_pa
        values = []
        for sample in samples:
            p = sample if isinstance(sample,(int,float)) else _data(sample).get('pressure_pa')
            p = finite_float(p,'pressure_pa')
            if p <= 0:
                raise CalibrationError('absolute pressure must be positive')
            values.append(p)
        if len(values) < self.minimum_depth_samples:
            raise CalibrationError(f'insufficient depth samples: {len(values)}/{self.minimum_depth_samples}')
        std = statistics.pstdev(values)
        if std > self.max_pressure_std_pa:
            raise CalibrationError(f'water surface pressure unstable: std={std:.2f} Pa')
        self.p_surface_pa = statistics.fmean(values)
        self.depth_calibrated = True
        self.diagnostics.update(depth_sample_count=len(values),pressure_std_pa=std)
        return self.p_surface_pa

    def fit(self, imu_samples, depth_samples):
        candidate = type(self)(self.config)
        candidate.calibrate_imu(imu_samples)
        candidate.calibrate_depth(depth_samples)
        self.__dict__.update(candidate.__dict__)
        return self.state_dict()

    calibrate = fit

    def angles(self, pitch, yaw, roll):
        if not self.imu_calibrated:
            raise CalibrationError('IMU zero has not been calibrated')
        return wrap180(pitch-self.pitch_zero_deg),wrap180(yaw-self.yaw_zero_deg),wrap180(roll-self.roll_zero_deg)

    def depth_m(self, pressure_pa):
        if not self.depth_calibrated:
            raise CalibrationError('surface pressure reference has not been initialized')
        return pressure_to_depth(pressure_pa,self.p_surface_pa,self.water_density_kg_m3,self.gravity_mps2)

    def state_dict(self):
        return {'pitch_zero_deg':self.pitch_zero_deg,'yaw_zero_deg':self.yaw_zero_deg,
                'roll_zero_deg':self.roll_zero_deg,'p_surface_pa':self.p_surface_pa,
                'pressure_reference_mode': 'fixed_config' if self.fixed_surface_pressure_mbar is not None else 'sample_mean',
                'surface_pressure_mbar': None if self.p_surface_pa is None else self.p_surface_pa / 100.,
                'imu_calibrated':self.imu_calibrated,'depth_calibrated':self.depth_calibrated,
                'gravity_validated':self.gravity_validated,'gravity_mps2':self.gravity_mps2,
                'water_density_kg_m3':self.water_density_kg_m3,'quaternion_order':'wxyz',
                'gravity_validation_scope':'static residual + raw Euler tilt; mounting axes require physical verification',
                **self.convention,**self.diagnostics}


def wait_for_calibration(sensors, config, *, stop_event=None, clock_ns=time.monotonic_ns, on_retry=None, on_check=None):
    stop_event = stop_event or threading.Event()
    startup = config.get('startup',{})
    window = finite_float(startup.get('calibration_window_s',5.),'calibration_window_s')
    timeout = finite_float(startup.get('calibration_timeout_s',30.),'calibration_timeout_s')
    if window <= 0 or timeout < window:
        raise ValueError('calibration timeout must be >= positive window duration')
    buffers = sensors.buffers if hasattr(sensors,'buffers') else sensors
    reference = RLSensorCalibration(config)
    # PPO 固定压力参考时只等待 IMU 稳定窗口；压力有效性仍由健康监测负责。
    required = (('imu',100.),) if reference.fixed_surface_pressure_mbar is not None else (('imu',100.),('depth',300.))
    start = clock_ns()
    last_error = CalibrationError('waiting for stable calibration window')
    while not stop_event.is_set():
        if on_check is not None:
            on_check()
        now = clock_ns()
        if now-start >= int(timeout*1e9):
            raise CalibrationError(f'calibration timeout: {last_error}') from last_error
        if now-start >= int(window*1e9):
            try:
                samples = {}
                for name,default_timeout in required:
                    samples[name] = [s for s in buffers[name].snapshot() if now-int(window*1e9) <= s.t_ns <= now]
                    rows = samples[name]
                    if not rows or rows[-1].t_ns-rows[0].t_ns < int(window*.8*1e9):
                        raise CalibrationError(f'{name}: calibration window incomplete')
                    limit = config.get('sensors',{}).get(name,{}).get('timeout_ms',default_timeout)
                    if config.get('state', {}).get('enforce_data_freshness', True) and (now-rows[-1].t_ns)/1e6 > float(limit):
                        raise CalibrationError(f'{name}: calibration sample stale')
                calibration = RLSensorCalibration(config)
                calibration.fit(samples['imu'],samples.get('depth', []))
                return calibration
            except (ValueError,KeyError) as exc:
                last_error = exc
                if on_retry is not None:
                    on_retry(str(exc))
        stop_event.wait(.05)
    if on_check is not None:
        on_check()  # 保留导致标定停止的具体健康故障。
    raise InterruptedError('calibration stopped')


SensorCalibration = RLSensorCalibration
RLSensorCalibrator = RLSensorCalibration
circular_mean = circular_mean_deg
gravity_compensate = remove_gravity
