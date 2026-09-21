"""Only depth, heading, pitch and roll enter the four Gaussian rewards."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, asdict
import math
from .rl_sensor_calibration_20260914 import finite_float, wrap180


@dataclass(frozen=True)
class RewardComponents:
    r_depth: float
    r_heading: float
    r_pitch: float
    r_roll: float
    reward: float
    depth_error_m: float
    heading_error_deg: float
    pitch_error_deg: float
    roll_error_deg: float

    @property
    def reward_total(self):
        return self.reward

    def as_dict(self):
        return {**asdict(self),'reward_total':self.reward}

    to_dict = as_dict

    def __getitem__(self,key):
        return getattr(self,key)


class RLReward:
    def __init__(self, config=None, *, target_depth_m=None, target_heading_deg=None):
        root = dict(config or {})
        c = root.get('reward',root)
        targets = root.get('targets',{})
        self.target_depth_m = finite_float(target_depth_m if target_depth_m is not None else targets.get('depth_m',c.get('target_depth_m',0.)), 'target_depth_m')
        self.target_heading_deg = finite_float(target_heading_deg if target_heading_deg is not None else targets.get('heading_deg',c.get('target_heading_deg',0.)), 'target_heading_deg')
        for name,default in (('depth',.10),('heading',15.),('pitch',10.),('roll',10.)):
            weight = finite_float(c.get(name+'_weight',1.),name+'_weight')
            suffix = '_scale_m' if name == 'depth' else '_scale_deg'
            scale = finite_float(c.get(name+suffix,default),name+suffix)
            if weight < 0 or scale <= 0:
                raise ValueError('reward weights must be nonnegative and scales positive')
            setattr(self,name+'_weight',weight)
            setattr(self,name+suffix,scale)
        if not math.isfinite(self.theoretical_max) or self.theoretical_max <= 0:
            raise ValueError('reward weight sum must be positive and finite')
        # Compatibility aliases for existing callers.
        self.ds,self.hs,self.ps,self.rs = self.depth_scale_m,self.heading_scale_deg,self.pitch_scale_deg,self.roll_scale_deg

    @property
    def theoretical_max(self):
        return self.depth_weight+self.heading_weight+self.pitch_weight+self.roll_weight

    @staticmethod
    def _get(state, name, alias):
        if isinstance(state,Mapping):
            value = state.get(name,state.get(alias))
        else:
            value = getattr(state,name,getattr(state,alias,None))
        return finite_float(value,name)

    @staticmethod
    def _gaussian(error,scale):
        ratio = abs(error/scale)
        # Large valid errors give zero without overflowing the square.
        return 0. if ratio > 40. else math.exp(-.5*ratio*ratio)

    def compute(self,state,*,target_depth_m=None,target_heading_deg=None):
        target_depth = self.target_depth_m if target_depth_m is None else finite_float(target_depth_m,'target_depth_m')
        target_heading = self.target_heading_deg if target_heading_deg is None else finite_float(target_heading_deg,'target_heading_deg')
        depth_error = finite_float(self._get(state,'depth_m','h')-target_depth,'depth_error_m')
        heading_error = wrap180(self._get(state,'yaw_deg','yaw')-target_heading)
        pitch_error = wrap180(self._get(state,'pitch_deg','pitch'))
        roll_error = wrap180(self._get(state,'roll_deg','roll'))
        rd,rh,rp,rr = [self._gaussian(e,s) for e,s in zip(
            (depth_error,heading_error,pitch_error,roll_error),
            (self.depth_scale_m,self.heading_scale_deg,self.pitch_scale_deg,self.roll_scale_deg))]
        total = self.depth_weight*rd+self.heading_weight*rh+self.pitch_weight*rp+self.roll_weight*rr
        return RewardComponents(rd,rh,rp,rr,total,depth_error,heading_error,pitch_error,roll_error)

    calculate = compute
    __call__ = compute


RewardCalculator = RLReward
