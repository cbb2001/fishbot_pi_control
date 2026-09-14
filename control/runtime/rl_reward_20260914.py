from dataclasses import dataclass,asdict
import math
from .rl_sensor_calibration_20260914 import wrap180
@dataclass
class RewardComponents:
    r_depth:float; r_heading:float; r_pitch:float; r_roll:float; reward:float; depth_error_m:float; heading_error_deg:float; pitch_error_deg:float; roll_error_deg:float
    def as_dict(self): return asdict(self)
    def __getitem__(self,k): return getattr(self,k)
class RLReward:
    def __init__(self,config=None,**kw):
        c=dict(config or {}); self.depth_weight=c.get('depth_weight',1); self.heading_weight=c.get('heading_weight',1); self.pitch_weight=c.get('pitch_weight',1); self.roll_weight=c.get('roll_weight',1); self.ds=c.get('depth_scale_m',.1); self.hs=c.get('heading_scale_deg',15); self.ps=c.get('pitch_scale_deg',10); self.rs=c.get('roll_scale_deg',10); self.target_depth_m=kw.get('target_depth_m',c.get('target_depth_m',0)); self.target_heading_deg=kw.get('target_heading_deg',c.get('target_heading_deg',0))
    def compute(self,s,**kw):
        g=lambda k,d=0:s.get(k,d) if isinstance(s,dict) else getattr(s,k,d); de=g('depth_m',g('h',0))-kw.get('target_depth_m',self.target_depth_m); he=wrap180(g('yaw_deg',g('yaw',0))-kw.get('target_heading_deg',self.target_heading_deg)); pe=wrap180(g('pitch_deg',g('pitch',0))); re=wrap180(g('roll_deg',g('roll',0))); f=lambda e,z:math.exp(-abs(e)/max(z,1e-9)); rd,rh,rp,rr=f(de,self.ds),f(he,self.hs),f(pe,self.ps),f(re,self.rs); return RewardComponents(rd,rh,rp,rr,self.depth_weight*rd+self.heading_weight*rh+self.pitch_weight*rp+self.roll_weight*rr,de,he,pe,re)
    calculate=compute
class RewardCalculator(RLReward): pass
