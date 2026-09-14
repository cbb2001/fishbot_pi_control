from __future__ import annotations
import time, numpy as np
from collections.abc import Mapping
from .rl_sensor_calibration_20260914 import RLSensorCalibration, remove_gravity

def _field(src, name, t):
    if isinstance(src, Mapping) and name in src: return src[name]
    if hasattr(src, 'build'):
        try: return src.build(t).get(name)
        except TypeError: return src.build().get(name)
    if hasattr(src, 'buffers') and name in src.buffers: return src.buffers[name].get_latest_before(t)
    return None
def _data(f): return (f.get('data', f) if isinstance(f, Mapping) else getattr(f, 'data', {})) or {}

class RLStateBuilder:
    OBSERVATION_DIM = 15
    def __init__(self, sensors=None, calibration=None, servo_state_tracker=None, config=None):
        self.sensors=sensors; self.calibration=calibration or RLSensorCalibration(); self.servo_state_tracker=servo_state_tracker; self.last_sensor_age={}
    def build(self, t_ns=None):
        t=int(time.monotonic_ns() if t_ns is None else t_ns); fs={n:_field(self.sensors,n,t) for n in ('imu','depth','power')}; d={n:_data(f) for n,f in fs.items()}
        i=d['imu']; pitch,yaw,roll=self.calibration.angles(i.get('pitch_deg',0),i.get('yaw_deg',0),i.get('roll_deg',0)); lin=i.get('linear_acc_mps2') or remove_gravity(i.get('acc_mps2',[0,0,0]),i.get('quat',i.get('quaternion')),self.calibration.gravity_mps2); dep=d['depth']; h=dep.get('depth_m',0)
        if dep.get('pressure_pa') is not None and self.calibration.p_surface_pa is not None: h=self.calibration.depth_m(dep['pressure_pa'])
        vals=[pitch,yaw,roll,*list(lin)[:3],h,d['power'].get('power_w',0)]; sv=[0.]*7; tr=self.servo_state_tracker
        if tr is not None and hasattr(tr,'get_servo_pose_at'):
            p=tr.get_servo_pose_at(t); p=p.to_dict() if hasattr(p,'to_dict') else p; a=p.get('estimated_angles_deg',p.get('estimated_angle_deg',{})) if isinstance(p,Mapping) else {}
            if isinstance(a,Mapping): sv=[float(a.get(k,a.get(str(k),0))) for k in range(1,8)]
        return np.asarray(vals+sv,dtype=np.float32)[:15]
    build_state=build
