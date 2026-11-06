"""水下导航纯计算：参数、分段惯性/EKF、批量平均测距与双组动作调度。

输出和三维EKF的 z 向下；去重力加速度暂保持 Z 向上，进入EKF前转换。
滤波只抑制摆鳍振动，不保证水平位置精度。
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field, fields, replace

import numpy as np

from .fixed_dive_20260920_100122 import FixedDiveScheduler, FixedFinSequence, ContinuousFinAction
from .rl_actions_20260914 import build_rl_trajectory, validate_trajectory
from .surface_navigation_20261010 import NavigationConfig, NavigationScheduler, UWBFrame
from .underwater_ekf_20261101 import EKFConfig, NavigationEKF


@dataclass(frozen=True)
class UnderwaterWaypoint:
    """一个目标段；t 为 None 时持续守点，正数表示到点或超时即切段。"""
    x: float
    y: float
    h: float
    t: float | None = None

    def __post_init__(self):
        for name in ('x','y','h','t'):
            value=getattr(self,name)
            if name=='t' and (value is None or isinstance(value,str) and value.lower()=='null'):
                object.__setattr__(self,name,None)
                continue
            if isinstance(value,bool): raise ValueError(f'目标点 {name} 不能为布尔值')
            try: value=float(value)
            except (TypeError,ValueError) as exc:
                raise ValueError(f'目标点 {name} 必须为有限数值，只有 t 可为 null') from exc
            if not math.isfinite(value) or value<0 or (name=='t' and value==0):
                raise ValueError(f'目标点 {name} 必须非负有限，t 必须为正数或 null')
            if name=='t' and (not math.isfinite(value*1e9) or round(value*1e9)<1):
                raise ValueError('目标点 t 无法表示为有效纳秒时间')
            object.__setattr__(self,name,value)


def parse_waypoints(rows):
    """YAML 支持 {x,y,h,t} 或四元素列表；CLI 的四个字符串也在此校验。"""
    if not isinstance(rows,(list,tuple)) or not rows:
        raise ValueError('waypoints 必须至少包含一个 (x,y,h,t) 目标点')
    result=[]
    for index,row in enumerate(rows,1):
        if isinstance(row,UnderwaterWaypoint): point=row
        elif isinstance(row,dict) and set(row)=={'x','y','h','t'}:
            point=UnderwaterWaypoint(**row)
        elif isinstance(row,(list,tuple)) and len(row)==4:
            point=UnderwaterWaypoint(*row)
        else: raise ValueError(f'第 {index} 个目标点必须包含 x,y,h,t 四项')
        result.append(point)
    if result[-1].t is not None:
        raise ValueError('最后一个目标点（含单目标）的 t 必须为 null')
    return tuple(result)


def validate_safety_depth(cfg,max_depth_m):
    maximum=float(max_depth_m)
    if not math.isfinite(maximum) or maximum<=0: raise ValueError('安全深度必须为正有限数')
    for index,point in enumerate(cfg.route,1):
        if point.h>=maximum:
            raise ValueError(f'第 {index} 个目标点 h 必须低于安全深度 {maximum:g} 米')


@dataclass(frozen=True)
class UnderwaterConfig:
    navigation: NavigationConfig
    h0: float = .3
    depth_kp: float = 1.
    depth_ki: float = 0.
    fin_duration_s: float = .3
    initial_b2: int = 1
    t1: float = 5.
    t2: float = 5.
    t3: float = 120.
    t4: float = 5.
    accel_tau_s: float = .20
    yaw_tau_s: float = .15
    depth_tau_s: float = .15
    history_s: float = 5.
    imu_gap_s: float = .05
    imu_timeout_s: float = .1
    pressure_timeout_s: float = .3
    power_timeout_s: float = 2.
    fix_timeout_s: float = 30.
    surface_timeout_s: float = 30.
    settle_s: float = 1.
    window_s: float = 2.
    min_span_s: float = 1.5
    min_frames: int = 12
    uwb_age_s: float = .3
    uwb_max_range_delta_m: float = 7.
    uwb_average_frames: int = 5
    ekf: EKFConfig = field(default_factory=EKFConfig)
    uwb_gap_s: float = .3
    min_retained: float = .8
    range_floor_m: float = .15
    range_mad_m: float = .08
    residual_m: float = .5
    position_p90_m: float = .25
    velocity_change_mps: float = .10
    max_surface_speed_mps: float = .5
    stable_s: float = .5
    stable_frames: int = 3
    boundary_m: float = .25
    resume_depth_m: float = .02
    resume_depth_samples: int = 3
    auto_bias: bool = False
    bias_window_s: float = 2.
    bias_speed_mps: float = .03
    bias_gyro_radps: float = .05
    bias_acc_std_mps2: float = .05
    bias_depth_std_m: float = .005
    bias_gain: float = .1
    bias_step_mps2: float = .005
    # 下列默认值仅方便纯计算测试构造；生产入口 from_robot 要求 YAML 完整。
    control_poll_s: float = .005
    merge_delay_s: float = .020
    status_hz: float = 1.
    ready_timeout_s: float = 5.
    safety_poll_s: float = .01
    center_ack_timeout_s: float = 2.
    center_settle_s: float = 1.
    shutdown_timeout_s: float = 3.
    calibration_coverage_ratio: float = .8
    calibration_sample_ratio: float = .9
    calibration_max_angle_std_deg: float = 1.
    calibration_max_pressure_std_pa: float = 100.
    calibration_max_linear_acc_rms_mps2: float = .5
    calibration_max_acc_std_mps2: float = .25
    calibration_max_gyro_radps: float = .15
    calibration_max_quat_euler_tilt_error_deg: float = 5.
    gravity_mps2: float = 9.80665
    water_density_kg_m3: float = 1000.
    pressure_drift_file: str = 'config/pressure_drift_10hz.json'
    imu_rate_hz: float = 200.
    pressure_rate_hz: float = 10.
    command_hz: float = 50.
    log_flush_s: float = 1.
    log_queue_size: int = 10000
    waypoints: tuple[UnderwaterWaypoint, ...] | None = None

    def __post_init__(self):
        if self.navigation.H is None or self.navigation.yaw_sign != 1:
            raise ValueError('水下导航要求 H 已知、yaw_sign=+1，先验证安装坐标')
        nonnegative = {'t1','depth_kp','depth_ki','accel_tau_s','yaw_tau_s','depth_tau_s',
                       'settle_s','stable_s','boundary_m'}
        integers = {'min_frames','stable_frames','resume_depth_samples','log_queue_size','uwb_average_frames'}
        if not isinstance(self.ekf,EKFConfig): raise ValueError('ekf 必须为 EKFConfig')
        for f in fields(self):
            if f.name in ('navigation','auto_bias','initial_b2','pressure_drift_file','ekf','waypoints'):
                continue
            if f.name in integers and isinstance(getattr(self,f.name),bool):
                raise ValueError(f'{f.name} 必须为整数，不能为布尔值')
            value = float(getattr(self,f.name))
            if not math.isfinite(value) or value < 0 or (value==0 and f.name not in nonnegative):
                raise ValueError(f'{f.name} 必须为有效正数（允许零的项除外）')
            if f.name in integers:
                if not value.is_integer(): raise ValueError(f'{f.name} 必须为整数')
                value=int(value)
            if (f.name in ('t1','t2','t3','t4') or f.name.endswith('_s')) and (
                    not math.isfinite(value*1e9) or (value>0 and round(value*1e9)<1)):
                raise ValueError(f'{f.name} 无法表示为有效纳秒时间')
            object.__setattr__(self,f.name,value)
        if (type(self.auto_bias) is not bool or type(self.initial_b2) is not int
                or self.initial_b2 not in (-1,1)):
            raise ValueError('auto_bias/initial_b2 类型或数值无效')
        if (self.min_frames < 4 or self.stable_frames < 2 or self.min_span_s > self.window_s
                or self.history_s < self.window_s+self.stable_s or self.min_retained > 1
                or self.bias_gain > 1):
            raise ValueError('窗口/保留比例/历史配置不一致')
        if not isinstance(self.pressure_drift_file,str) or not self.pressure_drift_file.strip():
            raise ValueError('pressure_drift_file 必须为非空路径')
        if self.calibration_coverage_ratio>1 or self.calibration_sample_ratio>1:
            raise ValueError('标定覆盖率和样本比例不能超过1')
        if self.merge_delay_s+self.control_poll_s>=min(self.imu_timeout_s,self.pressure_timeout_s):
            raise ValueError('合并等待与控制周期之和须小于传感器新鲜度期限')
        if self.history_s<self.bias_window_s:
            raise ValueError('history_s 必须覆盖 bias_window_s')
        if any(round(1e9/rate)<1 for rate in (self.imu_rate_hz,self.pressure_rate_hz,self.command_hz,self.status_hz)):
            raise ValueError('配置频率无法表示为有效纳秒周期')
        if 1/self.imu_rate_hz>self.imu_gap_s:
            raise ValueError('imu_gap_s 必须覆盖配置的 IMU 采样周期')
        if self.shutdown_timeout_s <= self.center_settle_s+1/self.command_hz:
            raise ValueError('shutdown_timeout_s 必须大于 center_settle_s 加一个 PWM 写入周期')
        if self.waypoints is not None:
            object.__setattr__(self,'waypoints',parse_waypoints(self.waypoints))
        for index,point in enumerate(self.route,1):
            if not (0<=point.x<=self.navigation.L and 0<=point.y<=self.navigation.W):
                raise ValueError(f'第 {index} 个目标点必须在 [0,L] × [0,W] 内')

    @property
    def route(self):
        # 未配置点表时完全沿用旧的单目标入口。
        return self.waypoints or (UnderwaterWaypoint(self.navigation.x_d,self.navigation.y_d,self.h0),)

    @classmethod
    def from_robot(cls, robot, **overrides):
        values=dict(robot.get('underwater_navigation_20261026',{}) or {})
        values.setdefault('waypoints',None)  # 兼容已有的单目标配置文件。
        # 共享采集/PWM/日志设置只保留一个来源，禁止水下段再复制一份。
        shared_paths={
            'imu_rate_hz':('sensors','imu','sample_rate_hz'),
            'pressure_rate_hz':('sensors','depth','rate_hz'),
            'imu_timeout_s':('sensors','imu','timeout_ms'),
            'pressure_timeout_s':('sensors','depth','timeout_ms'),
            'power_timeout_s':('sensors','power','timeout_ms'),
            'command_hz':('servo','discrete_executor','command_hz'),
            'center_settle_s':('servo','discrete_executor','center_settle_s'),
            'log_flush_s':('logging','flush_interval_s'),
            'log_queue_size':('logging','write_queue_maxsize'),
        }
        for name,path in shared_paths.items():
            if name in values or overrides.get(name) is not None:
                raise ValueError(f'{name} 请只在 robot.yaml 的 {".".join(path)} 配置')
            value=robot
            try:
                for key in path: value=value[key]
            except (KeyError,TypeError) as exc:
                raise ValueError(f'robot.yaml 缺少 {".".join(path)}') from exc
            values[name]=float(value)/1000 if path[-1]=='timeout_ms' else value
        values.update({k:v for k,v in overrides.items() if v is not None})
        if values['waypoints'] is not None and any(overrides.get(k) is not None for k in ('x_d','y_d','h0')):
            raise ValueError('waypoints 与单目标 --x-d/--y-d/--h0 不能同时使用；请直接修改点表')
        nav_names={f.name for f in fields(NavigationConfig)}-{f.name for f in fields(cls)}
        nav_values={k:values.pop(k) for k in list(values) if k in nav_names}
        if any(name in nav_values for name in ('L','W','H')):
            raise ValueError('基站 L/W/H 请只在 surface_navigation_20261010 配置')
        nav_config={**robot.get('surface_navigation_20261010',{}),**nav_values}
        missing_nav={f.name for f in fields(NavigationConfig)}-nav_config.keys()
        if missing_nav:
            raise ValueError('robot.yaml 水面/尾鳍配置缺少: '+', '.join(sorted(missing_nav)))
        nav=NavigationConfig.from_robot(robot,**nav_values)
        missing={f.name for f in fields(cls)}-{'navigation'}-values.keys()
        if missing: raise ValueError('robot.yaml 水下配置缺少: '+', '.join(sorted(missing)))
        values['ekf']=EKFConfig.from_mapping(values['ekf'])
        return cls(nav,**values)


class UnderwaterScheduler(FixedDiveScheduler):
    def schedule(self, agent, action, start_t_ns=None):
        if agent=='tail':
            return NavigationScheduler.schedule(self,agent,action,start_t_ns)
        return super().schedule(agent,action,start_t_ns)

    def hold_groups(self, angles_deg, agents):
        result=super().hold_groups(angles_deg,agents)
        # 只能在写入中位成功后恢复胸鳍顺序；取消的半程不计入 last_b2。
        if 'fin' in agents and all(abs(angles_deg[i]-self.centers[i])<1e-9 for i in (4,5,6,7)):
            self.fin_sequence=FixedFinSequence(last_b2=self.fin_sequence.last_b2)
            self._fin_sequence_interrupted=False
        return result


class SurfaceFinPolicy:
    """固定上浮：4号根部最大/最小交替，b2仅在成功完成b1=1后取反。"""
    def __init__(self,calibration,duration_s,initial_b2=1,last_b2=0):
        self.sequence=FixedFinSequence(last_b2=last_b2)
        self.initial_b2=initial_b2
        self.high=calibration.limits[4][1]-calibration.centers[4]
        self.low=calibration.limits[4][0]-calibration.centers[4]
        self.duration_s=duration_s
        # 完整行程必须满足左右根部/鳍尖机械限位，不静默缩幅。
        for previous in (0.,self.low):
            action=ContinuousFinAction(self.high,duration_s,0,1)
            validate_trajectory(build_rl_trajectory(action,previous,calibration),calibration.limits)
        for b2 in (-1,1):
            action=ContinuousFinAction(self.low,duration_s,1,b2)
            validate_trajectory(build_rl_trajectory(action,self.high,calibration),calibration.limits)

    def next_action(self):
        if self.sequence.next_b1==0:
            # action2旧定义在b1=0时忽略b2，并将日志中的b2规范化为0。
            return ContinuousFinAction(self.high,self.duration_s,0,1)
        b2=-self.sequence.last_b2 if self.sequence.last_b2 else self.initial_b2
        return ContinuousFinAction(self.low,self.duration_s,1,b2)

    def acknowledge(self,action):
        if action!=self.next_action(): raise RuntimeError('上浮完成回执与当前动作不匹配')
        self.sequence=self.sequence.after(action)


def rotation(quat):
    q=np.asarray(quat,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all(): raise ValueError('invalid quaternion')
    norm=float(np.linalg.norm(q))
    if not .8<=norm<=1.2: raise ValueError('invalid quaternion norm')
    w,x,y,z=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def alpha(dt,tau):
    return 1. if tau==0 else -math.expm1(-dt/tau)


def forward_tid(previous,current):
    """16 位 TID 允许回绕，拒绝重复和反向帧；不拿重建时间戳掩盖协议重复。"""
    if current is None or previous is None: return True
    return 0<(int(current)-int(previous))%65536<32768


class InertialEstimator:
    """旧连续积分/延迟回放估计器，保留供历史分析；实机使用 NavigationEstimator。"""
    def __init__(self,cfg,bias=(0,0,0),yaw_zero_deg=0,gravity=None):
        self.cfg,self.bias,self.gravity=cfg,np.asarray(bias,dtype=float),cfg.gravity_mps2 if gravity is None else gravity
        angle=math.radians(-yaw_zero_deg)
        self.pool=np.array([[math.cos(angle),-math.sin(angle),0],
                            [math.sin(angle),math.cos(angle),0],[0,0,1]])
        self.p=np.zeros(3); self.v=np.zeros(3)
        self.a1=np.zeros(3); self.a2=np.zeros(3); self.raw_acc=np.zeros(3)
        self.last_imu=None; self.last_depth=None; self.depth=None
        self.last_tid=None
        self.history=deque(); self.initialized=False

    @property
    def xyz(self): return [float(self.p[0]),float(self.p[1]),float(-self.p[2])]
    @property
    def filtered_acc(self): return self.a2.tolist()

    def _snapshot(self):
        return (self.p.copy(),self.v.copy(),self.a1.copy(),self.a2.copy(),self.raw_acc.copy(),
                self.last_imu,self.last_depth,self.depth,self.last_tid)

    def _restore(self,state):
        (self.p,self.v,self.a1,self.a2,self.raw_acc,self.last_imu,self.last_depth,self.depth,self.last_tid)=copy.deepcopy(state)

    def _record(self,t,kind,data):
        self.history.append((t,kind,copy.deepcopy(data),self._snapshot()))
        cutoff=t-round(self.cfg.history_s*1e9)
        while len(self.history)>1 and self.history[1][0]<cutoff: self.history.popleft()

    def imu(self,t,data):
        if self.last_imu is not None and t<=self.last_imu: return False
        if not forward_tid(self.last_tid,data.get('tid')): return False
        dt=0 if self.last_imu is None else (t-self.last_imu)/1e9
        if dt>self.cfg.imu_gap_s: raise ValueError(f'IMU time gap {dt:.3f}s')
        acc=np.asarray(data['acc_mps2'],dtype=float)
        if acc.shape!=(3,) or not np.isfinite(acc).all(): raise ValueError('invalid acceleration')
        self.raw_acc=self.pool @ (rotation(data['quat']) @ (acc-self.bias)-[0,0,self.gravity])
        previous=self.a2.copy()
        if self.last_imu is None: self.a1=self.raw_acc.copy(); self.a2=self.raw_acc.copy()
        else:
            gain=alpha(dt,self.cfg.accel_tau_s)
            self.a1+=gain*(self.raw_acc-self.a1)
            self.a2+=gain*(self.a1-self.a2)
            v=self.v.copy()
            self.v[:2]+=.5*(previous[:2]+self.a2[:2])*dt
            self.p[:2]+=.5*(v[:2]+self.v[:2])*dt
        # 垂向始终由压力约束，不在两个压力样本之间重新双积分。
        self.last_imu=t
        self.last_tid=data.get('tid')
        self._record(t,'imu',data)
        return True

    def pressure(self,t,depth):
        if self.last_depth is not None and t<=self.last_depth: return False
        if not math.isfinite(depth): raise ValueError('invalid pressure depth')
        if self.last_depth is None: self.depth=float(depth); self.v[2]=0.
        else:
            dt=(t-self.last_depth)/1e9
            gain=alpha(dt,self.cfg.depth_tau_s)
            filtered=self.depth+gain*(depth-self.depth)
            vz=-(filtered-self.depth)/dt
            self.v[2]+=gain*(vz-self.v[2]); self.depth=filtered
        self.p[2]=-self.depth; self.last_depth=t
        self._record(t,'pressure',float(depth))
        return True

    def depth_at(self,t):
        for row in reversed(self.history):
            if row[0]<=t and row[3][7] is not None:
                if t-row[3][6]>self.cfg.pressure_timeout_s*1e9: break
                return row[3][7]
        return None

    def correct(self,t,xy,velocity,*,reset_accel_filter=True):
        history=list(self.history)
        # IMU/压力输入必须按采集时间合并，检查点以同一时间轴保存。
        eligible=[i for i,row in enumerate(history) if row[0]<=t]
        if not eligible: raise ValueError('fix outside estimator history')
        index=eligible[-1]
        checkpoint=history[index][3]
        if checkpoint[5] is not None:
            dt=(t-checkpoint[5])/1e9
            if dt<0 or dt>self.cfg.imu_gap_s: raise ValueError('fix outside IMU history')
        self._restore(checkpoint)
        if self.last_imu is not None:
            self.p[:2]+=self.v[:2]*dt+.5*self.a2[:2]*dt*dt
            self.v[:2]+=self.a2[:2]*dt
            self.last_imu=t
        innovation={'position':(np.asarray(xy)-self.p[:2]).tolist(),
                    'velocity':(np.asarray(velocity)-self.v[:2]).tolist()}
        self.p[:2]=xy; self.v[:2]=velocity
        if reset_accel_filter:
            # 在定位采集时刻重置两级加速度滤波，避免上一航段/上浮动作的
            # 滤波残留继续驱动积分；保留原始观测、标定偏置和压力状态。
            self.a1.fill(0.)
            self.a2.fill(0.)
        # 丢弃校正前历史，避免后续定位回到旧坐标系；保留校正检查点。
        self.history=deque([(t,'fix',None,self._snapshot())])
        for stamp,kind,data,_ in history[index+1:]:
            if stamp<=t: continue
            if kind=='imu': self.imu(stamp,data)
            elif kind=='pressure': self.pressure(stamp,data)
        self.initialized=True
        return innovation


class NavigationEstimator:
    """只在导航阶段传播的估计器；传感器健康时间与积分时间分别维护。"""
    def __init__(self,cfg,bias=(0,0,0),yaw_zero_deg=0,gravity=None):
        self.cfg=cfg; self.bias=np.asarray(bias,dtype=float).copy()
        self.gravity=cfg.gravity_mps2 if gravity is None else gravity
        angle=math.radians(-yaw_zero_deg)
        self.pool=np.array([[math.cos(angle),-math.sin(angle),0],
                            [math.sin(angle),math.cos(angle),0],[0,0,1]])
        self.p=np.zeros(3); self.v=np.zeros(3)
        self.a1=np.zeros(3); self.a2=np.zeros(3); self.raw_acc=np.zeros(3)
        self.last_imu=None; self.last_depth=None; self.last_tid=None; self.depth=None
        self.initialized=False
        nav=cfg.navigation
        self.ekf=NavigationEKF(cfg.ekf,nav.L,nav.W,nav.H)
        self.origin_xy=np.zeros(2)
        self.integrating=False; self.segment_start_ns=None; self.segment_end_ns=None
        self.last_integrated_ns=None; self.last_filter_ns=None

    @property
    def relative_xy(self): return (self.p[:2]-self.origin_xy).tolist()

    @property
    def relative_xyz(self): return (self.p-np.r_[self.origin_xy,0.]).tolist()

    @property
    def xyz(self): return self.p.tolist()

    @property
    def filtered_acc(self): return self.a2.tolist()

    def start_navigation(self,t_ns,end_ns):
        if not self.initialized or end_ns<=t_ns: raise ValueError('导航须先完成有效UWB平均定位')
        self.integrating=True; self.segment_start_ns=int(t_ns); self.segment_end_ns=int(end_ns)
        self.last_integrated_ns=None; self.last_filter_ns=None
        self.a1.fill(0.); self.a2.fill(0.); self.ekf.resume()

    def pause_navigation(self,t_ns):
        if self.integrating:
            self.integrating=False
            self.ekf.pause(t_ns)
        self.last_integrated_ns=None; self.last_filter_ns=None
        self.a1.fill(0.); self.a2.fill(0.)

    def _in_segment(self,t):
        if self.integrating and t>=self.segment_end_ns: self.pause_navigation(self.segment_end_ns)
        return self.integrating and t>self.segment_start_ns

    def _sync_state(self):
        self.p[:]=self.ekf.x[:3]; self.v[:]=self.ekf.x[3:6]

    def _advance(self,t,acc_up):
        previous_t=self.segment_start_ns if self.last_integrated_ns is None else self.last_integrated_ns
        dt=(t-previous_t)/1e9
        if dt<=0: return  # 同一时刻的压力/IMU只传播一次。
        if dt>self.cfg.imu_gap_s: raise ValueError('IMU navigation gap exceeds imu_gap_s')
        acceleration=np.asarray(acc_up,dtype=float).copy(); acceleration[2]*=-1
        self.ekf.predict(dt,acceleration)
        self.last_integrated_ns=t; self._sync_state()

    def imu(self,t,data):
        if self.last_imu is not None and t<=self.last_imu: return False
        if not forward_tid(self.last_tid,data.get('tid')): return False
        if self.last_imu is not None and (t-self.last_imu)/1e9>self.cfg.imu_gap_s:
            raise ValueError('IMU time gap exceeds imu_gap_s')
        acc=np.asarray(data['acc_mps2'],dtype=float)
        if acc.shape!=(3,) or not np.isfinite(acc).all(): raise ValueError('invalid acceleration')
        self.raw_acc=self.pool @ (rotation(data['quat']) @ (acc-self.bias)-[0,0,self.gravity])
        # 即使暂停积分也继续检查采样连续性/姿态，不能把健康时间一起清零。
        self.last_imu=t; self.last_tid=data.get('tid')
        if not self._in_segment(t): return True
        previous_t=self.segment_start_ns if self.last_filter_ns is None else self.last_filter_ns
        dt=(t-previous_t)/1e9
        if dt>self.cfg.imu_gap_s: raise ValueError('IMU navigation start gap exceeds imu_gap_s')
        previous=self.a2.copy(); gain=alpha(dt,self.cfg.accel_tau_s)
        self.a1+=gain*(self.raw_acc-self.a1); self.a2+=gain*(self.a1-self.a2)
        # 无低通时首样本在本段短间隔内常值；有低通时从零建立，不能继承上浮滤波。
        if self.last_filter_ns is None and self.cfg.accel_tau_s==0: previous=self.a2.copy()
        self._advance(t,.5*(previous+self.a2)); self.last_filter_ns=t
        return True

    def pressure(self,t,depth):
        if self.last_depth is not None and t<=self.last_depth: return False
        if not math.isfinite(depth): raise ValueError('invalid pressure depth')
        if self.last_depth is None or self.depth is None: self.depth=float(depth)
        else:
            gain=alpha((t-self.last_depth)/1e9,self.cfg.depth_tau_s)
            self.depth+=gain*(depth-self.depth)
        self.last_depth=t
        if self._in_segment(t):
            # 先用最近已知加速度传播到压力采集时刻，再做观测更新；不重复积分。
            self._advance(t,self.a2)
            self.ekf.update_depth(float(depth))  # 使用校准压力，避免低通延迟被当成深度误差。
            self._sync_state()
        elif not self.integrating:
            # 暂停不做惯性传播/偏差学习，显示压力深度，XY保持不动。
            self.p[2]=self.depth; self.v[2]=0.
        return True

    def depth_at(self,t):
        # UWB定位固定使用水面深度0；导航中的压力另作为EKF观测。
        return 0.

    def apply_surface_fix(self,fix,now):
        if self.integrating: raise ValueError('导航阶段不能重新设置UWB起点')
        innovation={'position':(np.asarray(fix.xy)-self.p[:2]).tolist(),
                    'velocity':(-self.v[:2]).tolist()}
        # 无导航期间的回放。平均测距描述水面位置，校正后从实际恢复时刻重新计时。
        info=self.ekf.surface_fix(fix.t_ns,fix.xy,fix.ranges_m)
        self.p[:2]=fix.xy; self.p[2]=0.; self.v.fill(0.); self.depth=0.
        self.origin_xy=np.asarray(fix.xy,dtype=float).copy()
        self.a1.fill(0.); self.a2.fill(0.); self.raw_acc.fill(0.)
        self.last_integrated_ns=None; self.last_filter_ns=None
        self.ekf.last_pressure_update={}
        self.segment_start_ns=None; self.segment_end_ns=None; self.initialized=True
        innovation['ekf']=info
        return innovation


@dataclass(frozen=True)
class UWBObservation:
    t_ns: int
    frame: UWBFrame
    depth: float | None


@dataclass(frozen=True)
class StableFix:
    t_ns: int
    xy: list
    velocity: list
    quality: dict


@dataclass(frozen=True)
class RawUWBFix:
    t_ns: int
    frame_id: str
    ranges_m: tuple
    xy: list
    velocity: list
    quality: dict


class AcceptedUWB:
    """累计 n 个有效四路帧，逐路平均后定位；归中前后/不同航段不混用。"""
    def __init__(self,cfg):
        self.cfg=cfg
        self.last_accepted=None
        self.committed_ids=deque(maxlen=4096)
        self.reset(-1)

    def reset(self,boundary_ns):
        self.boundary=boundary_ns
        self.seen=deque(maxlen=512)
        self.rows=[]; self.published=False
        self.reason='waiting_frames'
        self.quality={}

    def add(self,obs,now):
        if (self.published or obs.t_ns<=self.boundary or obs.frame.frame_id in self.seen
                or any(row.frame.frame_id==obs.frame.frame_id for row in self.rows)
                or obs.frame.frame_id in self.committed_ids): return None
        if self.last_accepted is not None and (obs.t_ns<=self.last_accepted.t_ns
                or obs.frame.frame_id==self.last_accepted.frame_id): return None
        self.seen.append(obs.frame.frame_id)
        if self.rows and obs.t_ns<self.rows[-1].t_ns: return None
        self.quality={'mode':'mean_valid_ranges','ranges_m':list(obs.frame.ranges_m),
                      'frames':len(self.rows),'required_frames':self.cfg.uwb_average_frames,
                      'reference_frame_id':None if self.last_accepted is None else self.last_accepted.frame_id,
                      'max_range_delta_m':self.cfg.uwb_max_range_delta_m}
        if not 0<=now-obs.t_ns<=self.cfg.uwb_age_s*1e9:
            self.reason='stale'; return None
        ranges=obs.frame.ranges_m
        if len(ranges)!=4 or any(v is None or not math.isfinite(v) or v<=0 for v in ranges):
            self.reason='incomplete'; return None
        reference=(self.last_accepted.ranges_m if self.last_accepted is not None else
                   self.rows[0].frame.ranges_m if self.rows else None)
        if reference is not None:
            delta=[abs(a-b) for a,b in zip(ranges,reference)]
            self.quality['range_delta_m']=delta
            if any(d>self.cfg.uwb_max_range_delta_m+1e-9 for d in delta):
                self.reason='range_jump'; return None
        self.rows.append(obs)
        self.quality['frames']=len(self.rows)
        if len(self.rows)<self.cfg.uwb_average_frames:
            self.reason='collecting'; return None
        # 平均的是四路距离，不是不同报文拼凑的四路，也不是先解算坐标再平均。
        averaged=np.mean([row.frame.ranges_m for row in self.rows],axis=0)
        a,b,d,e=np.square(averaged); c=self.cfg.navigation
        xy=[float((a-b+e-d+2*c.L*c.L)/(4*c.L)),
            float((a-e+b-d+2*c.W*c.W)/(4*c.W))]
        if not np.isfinite(xy).all():
            self.rows.clear(); self.reason='invalid_geometry'; return None
        self.quality.update(frame_ids=[row.frame.frame_id for row in self.rows],
                            sample_times_ns=[row.t_ns for row in self.rows],
                            samples_ranges_m=[list(row.frame.ranges_m) for row in self.rows],
                            ranges_m=averaged.tolist(),surface_depth_m=0.)
        self.reason='accepted'
        self.published=True
        return RawUWBFix(obs.t_ns,obs.frame.frame_id,tuple(averaged.tolist()),xy,[0.,0.],dict(self.quality))

    def commit(self,fix):
        self.last_accepted=fix
        self.committed_ids.extend(fix.quality['frame_ids'])


class StableUWB:
    """旧稳定窗口算法，仅保留供历史回放；当前导航入口使用AcceptedUWB。"""
    def __init__(self,cfg):
        self.cfg=cfg; self.reset(-1)

    def reset(self,boundary_ns):
        self.boundary=boundary_ns; self.rows=deque(); self.seen=deque(maxlen=512)
        self.signal_seen=deque(maxlen=512); self.last_signal=None
        self.candidate_ns=None; self.candidate_count=0; self.reason='waiting_frames'
        self.quality={}; self.last_complete_ns=None

    def signal(self,obs,now):
        if (obs.t_ns<=self.boundary or now-obs.t_ns<0 or now-obs.t_ns>self.cfg.uwb_age_s*1e9
                or obs.frame.frame_id in self.signal_seen): return False
        self.signal_seen.append(obs.frame.frame_id)
        if not any(v is not None and math.isfinite(v) and v>0 for v in obs.frame.ranges_m):
            self.last_signal=None; return False
        passed=(self.last_signal is not None and 0<obs.t_ns-self.last_signal<=self.cfg.uwb_gap_s*1e9)
        self.last_signal=obs.t_ns
        return passed

    def _reject(self,reason):
        self.reason=reason; self.candidate_ns=None; self.candidate_count=0
        return None

    def _coverage(self,rows,original,now):
        c=self.cfg
        return (len(rows)>=c.min_frames and len(rows)>=c.min_retained*original
                and rows[-1].t_ns-rows[0].t_ns>=c.min_span_s*1e9
                and 0<=now-rows[-1].t_ns<=c.uwb_age_s*1e9
                and max(b.t_ns-a.t_ns for a,b in zip(rows,rows[1:]))<=c.uwb_gap_s*1e9)

    def _xy_residual(self,ranges,depth):
        c=self.cfg.navigation; a,b,d,e=np.square(ranges)
        x=(a-b+e-d+2*c.L*c.L)/(4*c.L)
        y=(a-e+b-d+2*c.W*c.W)/(4*c.W)
        predicted=np.sqrt(np.square(x-np.array([0,c.L,c.L,0]))+
                          np.square(y-np.array([0,0,c.W,c.W]))+(c.H+depth)**2)
        rms=float(np.sqrt(np.mean(np.square(predicted-ranges))))
        valid=(-self.cfg.boundary_m<=x<=c.L+self.cfg.boundary_m
               and -self.cfg.boundary_m<=y<=c.W+self.cfg.boundary_m and rms<=self.cfg.residual_m)
        return [x,y],rms,valid

    @staticmethod
    def _fit(rows,xy):
        t=np.array([(r.t_ns-rows[-1].t_ns)/1e9 for r in rows])
        fit=np.linalg.lstsq(np.column_stack((np.ones(len(t)),t)),xy,rcond=None)[0]
        errors=np.linalg.norm(xy-(fit[0]+t[:,None]*fit[1]),axis=1)
        return fit,errors

    def add(self,obs,now):
        c=self.cfg
        if obs.t_ns<=self.boundary or obs.frame.frame_id in self.seen: return None
        self.seen.append(obs.frame.frame_id)
        if now-obs.t_ns<0 or now-obs.t_ns>c.uwb_age_s*1e9: return self._reject('stale')
        ranges=obs.frame.ranges_m
        if any(v is None or not math.isfinite(v) or v<=0 for v in ranges):
            return self._reject('incomplete')
        if obs.depth is None or not math.isfinite(obs.depth): return self._reject('no_pressure')
        if self.rows and obs.t_ns<=self.rows[-1].t_ns: return None
        self.last_complete_ns=obs.t_ns; self.rows.append(obs)
        while self.rows and obs.t_ns-self.rows[0].t_ns>c.window_s*1e9: self.rows.popleft()
        rows=list(self.rows); original=len(rows)
        if not self._coverage(rows,original,now): return self._reject('coverage')
        ranges=np.array([r.frame.ranges_m for r in rows])
        median=np.median(ranges,axis=0); mad=np.median(abs(ranges-median),axis=0)
        mask=np.all(abs(ranges-median)<=np.maximum(c.range_floor_m,3*1.4826*mad),axis=1)
        rows=[r for r,keep in zip(rows,mask) if keep]
        if not self._coverage(rows,original,now): return self._reject('range_outliers')
        retained=np.array([r.frame.ranges_m for r in rows]); median=np.median(retained,axis=0)
        mad=np.median(abs(retained-median),axis=0)
        if max(mad)>c.range_mad_m: return self._reject('range_mad')
        if not self._xy_residual(median,float(np.median([r.depth for r in rows])))[2]:
            return self._reject('median_geometry')
        geometry=[self._xy_residual(r.frame.ranges_m,r.depth) for r in rows]
        xy=np.array([g[0] for g in geometry if g[2]])
        rows=[r for r,g in zip(rows,geometry) if g[2]]
        if not self._coverage(rows,original,now): return self._reject('geometry')
        fit,errors=self._fit(rows,xy)
        # 在速度拟合前剔除二维位置异常，再重新检查帧数、跨度和间隔。
        em=float(np.median(errors)); emad=float(np.median(abs(errors-em)))
        keep=errors<=max(c.position_p90_m,em+3*1.4826*emad)
        rows=[r for r,k in zip(rows,keep) if k]; xy=xy[keep]
        if not self._coverage(rows,original,now): return self._reject('position_outliers')
        fit,errors=self._fit(rows,xy)
        split=len(rows)//2
        before,_=self._fit(rows[:split],xy[:split]); after,_=self._fit(rows[split:],xy[split:])
        p90=float(np.percentile(errors,90)); change=float(np.linalg.norm(before[1]-after[1]))
        self.quality=dict(frames=len(rows),original_frames=original,range_mad_m=mad.tolist(),
                          p90_m=p90,velocity_change_mps=change,age_s=(now-rows[-1].t_ns)/1e9,
                          span_s=(rows[-1].t_ns-rows[0].t_ns)/1e9,
                          max_half_speed_mps=float(max(np.linalg.norm(before[1]),np.linalg.norm(after[1]))))
        if p90>c.position_p90_m or change>c.velocity_change_mps or np.linalg.norm(fit[1])>c.max_surface_speed_mps:
            return self._reject('motion_or_scatter')
        if rows[-1].t_ns!=obs.t_ns:
            # 新异常使连续稳定计时重新开始；后续新证据允许使用筛除后的窗口。
            return self._reject('latest_outlier')
        if self.candidate_ns is None: self.candidate_ns=obs.t_ns; self.candidate_count=0
        self.candidate_count+=1; self.reason='confirming'
        if (obs.t_ns-self.candidate_ns<c.stable_s*1e9 or self.candidate_count<c.stable_frames): return None
        self.reason='stable'
        return StableFix(rows[-1].t_ns,fit[0].tolist(),fit[1].tolist(),dict(self.quality))
