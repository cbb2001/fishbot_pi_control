import math
from collections.abc import Mapping
def wrap180(x): return ((float(x)+180)%360)-180
def circular_mean_deg(v):
 v=list(v)
 if not v: raise ValueError('empty angles')
 return wrap180(math.degrees(math.atan2(sum(math.sin(math.radians(x)) for x in v),sum(math.cos(math.radians(x)) for x in v))))
def _d(s):
 if isinstance(s,Mapping): return s.get('data',s)
 return getattr(s,'data',{}) or {k:getattr(s,k) for k in ('pitch_deg','yaw_deg','roll_deg','pressure_pa','acc_mps2','quat') if hasattr(s,k)}
def remove_gravity(a,q=None,g=9.80665):
 a=list(a); a=(a+[0,0,0])[:3]; q=list(q or (1,0,0,0)); q=(q+[1,0,0,0])[:4]; n=math.sqrt(sum(x*x for x in q)); w,x,y,z=[i/n for i in q] if n else (1,0,0,0); gb=(2*(x*z-w*y)*g,2*(y*z+w*x)*g,(1-2*(x*x+y*y))*g); return tuple(a[i]-gb[i] for i in range(3))
class RLSensorCalibration:
 def __init__(self,config=None,**kw):
  c=dict(config or {}); self.minimum_imu_samples=int(kw.get('minimum_imu_samples',c.get('minimum_imu_samples',1))); self.minimum_depth_samples=int(kw.get('minimum_depth_samples',c.get('minimum_depth_samples',1))); self.water_density_kg_m3=float(c.get('water_density_kg_m3',1000)); self.gravity_mps2=float(c.get('gravity_mps2',9.80665)); self.pitch_zero_deg=self.yaw_zero_deg=self.roll_zero_deg=0.; self.p_surface_pa=None
 def calibrate_imu(self,s):
  r=[_d(x) for x in s];
  if len(r)<self.minimum_imu_samples: raise ValueError('insufficient IMU samples')
  self.pitch_zero_deg=circular_mean_deg(x.get('pitch_deg',0) for x in r); self.yaw_zero_deg=circular_mean_deg(x.get('yaw_deg',0) for x in r); self.roll_zero_deg=circular_mean_deg(x.get('roll_deg',0) for x in r); return self.state_dict()
 def calibrate_depth(self,s):
  v=[float(_d(x).get('pressure_pa',x if isinstance(x,(int,float)) else 0)) for x in s]
  if len(v)<self.minimum_depth_samples: raise ValueError('insufficient depth samples')
  self.p_surface_pa=sum(v)/len(v); return self.p_surface_pa
 def angles(self,p,y,r): return wrap180(p-self.pitch_zero_deg),wrap180(y-self.yaw_zero_deg),wrap180(r-self.roll_zero_deg)
 def depth_m(self,p): return (float(p)-self.p_surface_pa)/(self.water_density_kg_m3*self.gravity_mps2)
 def state_dict(self): return {'pitch_zero_deg':self.pitch_zero_deg,'yaw_zero_deg':self.yaw_zero_deg,'roll_zero_deg':self.roll_zero_deg,'p_surface_pa':self.p_surface_pa}
SensorCalibration=RLSensorCalibration; RLSensorCalibrator=RLSensorCalibration; circular_mean=circular_mean_deg; gravity_compensate=remove_gravity
