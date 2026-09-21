"""PPO action spaces and trajectories."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Mapping,Any

TAIL_THETAS=(-30.,-20.,-10.,0.,10.,20.,30.); FIN_THETAS=(-53.,0.,53.); ACTION_DURATIONS=(.2,.3,.4,.5,.6)
FIN_DURATIONS=(.3,.6)
FIN_MODES=((0,0),(1,-1),(1,1))
TAIL_SERVO_IDS=(1,2,3); LEFT_FIN_SERVO_IDS=(4,5); RIGHT_FIN_SERVO_IDS=(6,7); ALL_SERVO_IDS=TAIL_SERVO_IDS+LEFT_FIN_SERVO_IDS+RIGHT_FIN_SERVO_IDS
FIN_SERVO_IDS=LEFT_FIN_SERVO_IDS+RIGHT_FIN_SERVO_IDS
class ActionValidationError(ValueError): pass
@dataclass(frozen=True)
class TailAction:
 theta:float; t:float
 def __post_init__(self):
  if float(self.theta) not in TAIL_THETAS or float(self.t) not in ACTION_DURATIONS: raise ActionValidationError('action1 theta/t 不在离散动作空间')
 def to_dict(self): return {'theta':float(self.theta),'t':float(self.t)}
@dataclass(frozen=True)
class FinAction:
 theta:float; t:float; b1:int; b2:int
 def __post_init__(self):
  if float(self.theta) not in FIN_THETAS or float(self.t) not in FIN_DURATIONS or type(self.b1) is not int or self.b1 not in (0,1) or type(self.b2) is not int or self.b2 not in (-1,0,1) or (self.b1 == 1 and self.b2 == 0): raise ActionValidationError('action2 参数不在离散动作空间')
  if self.b1 == 0: object.__setattr__(self,'b2',0)
 def to_dict(self): return {'theta':float(self.theta),'t':float(self.t),'b1':self.b1,'b2':self.b2}
action1=lambda theta,t:TailAction(theta,t); action2=lambda theta,t,b1,b2:FinAction(theta,t,b1,b2)
class TailActionSpace:
 nvec=(7,5)
 def __len__(self): return 35
 def decode(self,i,j): return TailAction(TAIL_THETAS[int(i)],ACTION_DURATIONS[int(j)])
 def all(self): return tuple(self.decode(i,j) for i in range(7) for j in range(5))
class FinActionSpace:
 nvec=(3,2,3)
 def __len__(self): return 18
 def decode(self,theta_index,time_index,mode_index):
  indices=(theta_index,time_index,mode_index)
  if any(isinstance(i,bool) or int(i)!=i or not 0<=int(i)<n for i,n in zip(indices,self.nvec)): raise ActionValidationError('fin action index out of range')
  return FinAction(FIN_THETAS[int(theta_index)],FIN_DURATIONS[int(time_index)],*FIN_MODES[int(mode_index)])
 def all(self): return tuple(self.decode(i,j,k) for i in range(3) for j in range(2) for k in range(3))
 def decode_flat(self,index):
  if isinstance(index,bool) or int(index)!=index or not 0<=index<len(self): raise ActionValidationError('fin action index out of range')
  theta,rest=divmod(int(index),6)
  duration,mode=divmod(rest,3)
  return self.decode(theta,duration,mode)


@dataclass(frozen=True)
class FinSequenceState:
 """仅记录已完成动作；独立于历史窗口，跨 PPO 更新和 episode 保留。"""
 previous_theta: float = 0.
 next_b1: int = 0
 last_effective_direction: int = 0  # 0=尚无记录，+1=前进，-1=后退

 def __post_init__(self):
  if self.previous_theta not in FIN_THETAS or self.next_b1 not in (0,1) or self.last_effective_direction not in (-1,0,1):
   raise ActionValidationError('非法侧鳍序列状态')

 def direction(self,action):
  """中间位置固定，比较绝对目标角等价于比较 theta；静止动作不记方向。"""
  delta=action.theta-self.previous_theta
  return (1 if delta>0 else -1)*action.b2 if action.b1==1 and delta!=0 else 0

 def allows(self,action):
  direction=self.direction(action)
  return action.b1==self.next_b1 and (direction==0 or self.last_effective_direction==0 or direction!=self.last_effective_direction)

 def validate(self,action):
  if not self.allows(action):
   raise ActionValidationError('侧鳍动作违反 b1 交替或有效前进/后退交替规则')

 def action_mask(self):
  """联合 18 动作掩码，采样和 PPO 概率重算必须使用同一份。"""
  return tuple(self.allows(action) for action in FinActionSpace().all())

 def after(self,action):
  """仅在端点 PWM 写入成功后调用；无效行程保留上次有效方向。"""
  self.validate(action)
  return FinSequenceState(float(action.theta),1-action.b1,self.direction(action) or self.last_effective_direction)

 def vector(self):
  return (self.previous_theta/53.,float(self.next_b1),float(self.last_effective_direction))

 def to_dict(self):
  return {'previous_theta':self.previous_theta,'next_b1':self.next_b1,
          'last_effective_direction':self.last_effective_direction}
def quintic_smoothstep(u):
 u=float(u)
 if u<=0:return 0.
 if u>=1:return 1.
 return 10*u**3-15*u**4+6*u**5
def smooth_motion(a,b,e,d):
 if d<=0:raise ValueError('duration_s must be positive')
 if e<=0:return float(a)
 if e>=d:return float(b)
 return float(a)+(float(b)-float(a))*quintic_smoothstep(e/d)
def evaluate_fin_tip_motion(c,p,e,d):
 if e<=0:return float(c)
 if e>=d:return float(c)
 h=d/2
 return smooth_motion(c,p,e,h) if e<=h else smooth_motion(p,c,e-h,h)
def mirror_fin_angles(x:Mapping[int,float],left_center=111.,right_center=143.,tip_center=90.,right_tip_center=None):
 return {6:right_center-(x[4]-left_center),7:(tip_center if right_tip_center is None else right_tip_center)-(x[5]-tip_center)}
@dataclass(frozen=True)
class RLActionTrajectory:
 sequence_name:str; action:Any; previous_theta:float; centers:Mapping[int,float]; tip_peak_angle_deg:float|None=None
 start_override:Mapping[int,float]=field(default_factory=dict)
 @property
 def duration_s(self):return float(self.action.t)
 @property
 def theta_delta(self):return float(self.action.theta)-self.previous_theta
 @property
 def absolute_theta_delta(self):return abs(self.theta_delta)
 @property
 def start_angles_deg(self):
  angles=({i:self.centers[i]+self.previous_theta for i in TAIL_SERVO_IDS} if isinstance(self.action,TailAction) else {4:self.centers[4]+self.previous_theta,5:self.centers[5]})
  return {i:self.start_override.get(i,a) for i,a in angles.items()}
 @property
 def target_angles_deg(self): return ({i:self.centers[i]+self.action.theta for i in TAIL_SERVO_IDS} if isinstance(self.action,TailAction) else {4:self.centers[4]+self.action.theta,5:self.centers[5]})
 def evaluate(self,e):
  if isinstance(self.action,TailAction): return {i:smooth_motion(self.start_angles_deg[i],self.target_angles_deg[i],e,self.duration_s) for i in TAIL_SERVO_IDS}
  h=self.duration_s/2
  tip=(smooth_motion(self.start_angles_deg[5],self.tip_peak_angle_deg,e,h) if e<=h else smooth_motion(self.tip_peak_angle_deg,self.centers[5],e-h,h)) if self.action.b1 else smooth_motion(self.start_angles_deg[5],self.centers[5],e,self.duration_s)
  return {4:smooth_motion(self.start_angles_deg[4],self.target_angles_deg[4],e,self.duration_s),5:tip}
 def evaluate_all(self,e):
  x=self.evaluate(e)
  return {**x,**(mirror_fin_angles(x,self.centers[4],self.centers[6],self.centers[5],self.centers[7]) if isinstance(self.action,FinAction) else {})}
def build_rl_trajectory(action,previous_theta,calibration,*,start_angles_deg=None):
 c=calibration.centers if hasattr(calibration,'centers') else calibration
 coupling=getattr(calibration,'coupling',{})
 root_span=float(coupling.get('root_reference_span_deg',106.)); tip_span=float(coupling.get('tip_reference_span_deg',90.))
 p=float(previous_theta); peak=(c[5]+action.b2*abs(action.theta-p)/root_span*tip_span) if isinstance(action,FinAction) and action.b1 else None
 return RLActionTrajectory('tail' if isinstance(action,TailAction) else 'fin',action,p,dict(c),peak,dict(start_angles_deg or {}))
@dataclass(frozen=True)
class RLRobotCalibration:
 servos:Mapping[int,Any]; centers:Mapping[int,float]; limits:Mapping[int,tuple[float,float]]
 coupling:Mapping[str,float]=field(default_factory=dict)
 @property
 def initial_angles_deg(self): return dict(self.centers)
 @property
 def initial_previous_thetas(self): return {'tail':0.,'fin':0.}
def build_calibration(config):
 channels=config.get('servo',{}).get('channels',[]); servos={int(x['servo_id']):x for x in channels if isinstance(x,dict) and 'servo_id' in x}
 missing=set(ALL_SERVO_IDS)-set(servos)
 if missing: raise ActionValidationError(f'缺少舵机: {sorted(missing)}')
 centers={i:float(servos[i].get('center_angle',90)) for i in ALL_SERVO_IDS}; limits={i:(float(servos[i].get('min_angle',0)),float(servos[i].get('max_angle',180))) for i in ALL_SERVO_IDS}
 coupling=dict(config.get('discrete_action_coupling',{}).get('left',{}))
 if not coupling: coupling=dict(config.get('discrete_absolute_actions_20260725',{}).get('left_fin',{}))
 for name,default in (('root_reference_span_deg',106.),('tip_reference_span_deg',90.)):
  coupling[name]=float(coupling.get(name,default))
  if not math.isfinite(coupling[name]) or coupling[name]<=0: raise ActionValidationError(f'invalid coupling {name}')
 return RLRobotCalibration(servos,centers,limits,coupling)
def validate_trajectory(trajectory,limits):
 # 五次插值在各半段单调；两端及半程峰值覆盖全部机械极值，禁止裁剪。
 for elapsed in (0.,trajectory.duration_s/2,trajectory.duration_s):
  for sid,angle in trajectory.evaluate_all(elapsed).items():
   lo,hi=limits[sid]
   if not math.isfinite(angle) or not lo<=angle<=hi: raise ActionValidationError(f'servo {sid} angle {angle:g} exceeds limits [{lo:g}, {hi:g}]')

def validate_action_space(calibration,agents=('tail','fin')):
 spaces={'tail':(TailActionSpace(),TAIL_THETAS),'fin':(FinActionSpace(),FIN_THETAS)}
 for agent in agents:
  if agent not in spaces: raise ActionValidationError(f'unknown agent: {agent}')
  space,previous_values=spaces[agent]
  for action in space.all():
   for previous in previous_values: validate_trajectory(build_rl_trajectory(action,previous,calibration),calibration.limits)
 return True
