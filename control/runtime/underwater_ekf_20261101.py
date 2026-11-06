"""三维九状态 EKF：导航阶段惯性传播，压力更新垂向，水面测距重设起点。

复用 20261030 离线回放的状态/噪声/迭代测距模型。状态为
[x,y,z,vx,vy,vz,bx,by,bz]，z向下。b 是池坐标系等效加速度误差，
不修改启动 IMU 标定。XY默认参数来自原回放；压力噪声另行配置。
"""
from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class EKFConfig:
    enabled: bool = True
    retain_bias_on_fix: bool = True
    range_sigma_m: float = .3
    accel_noise_density: float = .03
    bias_random_walk: float = .003
    initial_bias_sigma_mps2: float = .03
    initial_position_sigma_m: float = .7
    segment_velocity_sigma_mps: float = .1
    pause_velocity_sigma_mps: float = .3
    pressure_sigma_m: float = .01
    pressure_innovation_gate: float = 9.
    innovation_gate: float = 100.
    iterations: int = 3

    def __post_init__(self):
        for f in fields(self):
            value=getattr(self,f.name)
            if f.name in ('enabled','retain_bias_on_fix'):
                if type(value) is not bool: raise ValueError(f'ekf.{f.name} 必须为布尔值')
            elif f.name=='iterations':
                if type(value) is not int or not 1<=value<=20:
                    raise ValueError('ekf.iterations 必须为 1..20 的整数')
            else:
                if isinstance(value,bool): raise ValueError(f'ekf.{f.name} 必须为数值')
                number=float(value)
                allow_zero=f.name in ('accel_noise_density','bias_random_walk')
                if not math.isfinite(number) or number<0 or (number==0 and not allow_zero):
                    raise ValueError(f'ekf.{f.name} 数值无效')
                object.__setattr__(self,f.name,number)

    @classmethod
    def from_mapping(cls,values):
        if not isinstance(values,dict): raise ValueError('robot.yaml 缺少 ekf 配置字典')
        names={f.name for f in fields(cls)}
        missing=names-values.keys(); unknown=values.keys()-names
        if missing: raise ValueError('robot.yaml ekf 缺少: '+', '.join(sorted(missing)))
        if unknown: raise ValueError('robot.yaml ekf 未知字段: '+', '.join(sorted(unknown)))
        return cls(**values)


def transition(dt,sigma_a,sigma_b):
    """连续白噪声的精确离散 F/Q，包含位置、速度与偏差交叉协方差。"""
    I=np.eye(3); F=np.eye(9)
    F[:3,3:6]=dt*I; F[:3,6:]=-.5*dt*dt*I; F[3:6,6:]=-dt*I
    qa=sigma_a**2; qb=sigma_b**2
    small=np.array([[qa*dt**3/3+qb*dt**5/20,qa*dt**2/2+qb*dt**4/8,-qb*dt**3/6],
                    [qa*dt**2/2+qb*dt**4/8,qa*dt+qb*dt**3/3,-qb*dt**2/2],
                    [-qb*dt**3/6,-qb*dt**2/2,qb*dt]])
    return F,np.kron(small,I)


def range_model(state,anchors,height):
    delta=state[:2]-anchors
    vertical=height+state[2]
    predicted=np.sqrt(np.sum(delta*delta,axis=1)+vertical*vertical)
    # 标签恰在基站正下方且高度为0时，使用有限分母避免0/0。
    H=np.zeros((4,9)); H[:,:2]=delta/np.maximum(predicted,1e-12)[:,None]
    H[:,2]=vertical/np.maximum(predicted,1e-12)
    return predicted,H


class NavigationEKF:
    def __init__(self,cfg,L,W,H):
        self.cfg=cfg; self.height=float(H)
        self.anchors=np.array([[0,0],[L,0],[L,W],[0,W]],dtype=float)
        self.x=np.zeros(9)
        self.P=np.diag([cfg.initial_position_sigma_m**2]*2+[cfg.pressure_sigma_m**2]+
                       [cfg.segment_velocity_sigma_mps**2]*3+[cfg.initial_bias_sigma_mps2**2]*3)
        self.initialized=False; self.pause_ns=None; self.last_update={}; self.last_pressure_update={}

    def predict(self,dt,acceleration):
        if not math.isfinite(dt) or dt<=0: raise ValueError('EKF dt 必须为正数')
        acceleration=np.asarray(acceleration,dtype=float)
        if acceleration.shape!=(3,) or not np.isfinite(acceleration).all():
            raise ValueError('EKF acceleration 无效')
        c=self.cfg
        F,Q=transition(dt,c.accel_noise_density,c.bias_random_walk if c.enabled else 0.)
        self.x=F@self.x
        self.x[:3]+=.5*acceleration*dt*dt; self.x[3:6]+=acceleration*dt
        if c.enabled:
            self.P=F@self.P@F.T+Q
            self.P=.5*(self.P+self.P.T)

    def pause(self,t_ns):
        if self.pause_ns is None: self.pause_ns=int(t_ns)

    def resume(self):
        self.pause_ns=None

    def update_depth(self,depth):
        """校准后的压力深度为高权重一维观测，同时约束z/vz/bz。"""
        if not math.isfinite(depth): raise ValueError('EKF pressure depth 无效')
        innovation=float(depth-self.x[2]); R=self.cfg.pressure_sigma_m**2
        S=float(self.P[2,2]+R); nis=innovation**2/S
        accepted=nis<=self.cfg.pressure_innovation_gate
        self.last_pressure_update=dict(innovation_m=innovation,nis=nis,accepted=accepted)
        if not self.cfg.enabled:
            self.x[2]=depth; self.x[5]=0.
        elif accepted:
            K=self.P[:,2]/S
            self.x+=K*innovation
            A=np.eye(9); A[:,2]-=K
            self.P=A@self.P@A.T+np.outer(K,K)*R
            self.P=.5*(self.P+self.P.T)
        return self.last_pressure_update

    def surface_fix(self,t_ns,xy,ranges):
        """先用平均四路距离学习偏差，再严格以平均测距解算坐标为新起点。

        水面观测为平均四路距离加深度0。暂停时均值冻结，但加入未知运动方差，避免把
        上浮位移全部解释成 IMU 偏差。NIS 只限制偏差学习，不阻止有效均值重定位。
        """
        xy=np.asarray(xy,dtype=float); ranges=np.asarray(ranges,dtype=float); c=self.cfg
        if xy.shape!=(2,) or ranges.shape!=(4,) or not np.isfinite(xy).all() or not np.isfinite(ranges).all() or np.any(ranges<=0):
            raise ValueError('EKF surface fix 无效')
        pause_s=0. if self.pause_ns is None else max(0.,(t_ns-self.pause_ns)/1e9)
        if c.enabled and self.initialized:
            self.P[:3,:3]+=np.eye(3)*(c.pause_velocity_sigma_mps*pause_s)**2
            self.P[6:,6:]+=np.eye(3)*c.bias_random_walk**2*pause_s
        prior=self.x.copy(); cov=self.P.copy()
        info=dict(accepted=False,nis=None,pause_s=pause_s,prior_xy=prior[:2].tolist(),
                  prior_xyz=prior[:3].tolist(),prior_position_variance=np.diag(cov)[:3].tolist(),
                  bias_before=prior[6:].tolist())
        if c.enabled and self.initialized:
            # 相邻UWB帧可能相关；平均后仍使用配置R，不擅自除以n低估噪声。
            R=np.diag([c.range_sigma_m**2]*4+[c.pressure_sigma_m**2])
            z=np.r_[ranges,0.]
            def observation(state):
                pred,H=range_model(state,self.anchors,self.height)
                depth_H=np.zeros((1,9)); depth_H[0,2]=1.
                return np.r_[pred,state[2]],np.vstack((H,depth_H))
            pred,H=observation(prior)
            innovation=z-pred; S=H@cov@H.T+R
            nis=float(innovation@np.linalg.solve(S,innovation))
            info.update(nis=nis,accepted=nis<=c.innovation_gate)
            if info['accepted']:
                current=prior.copy()
                for _ in range(c.iterations):
                    pred,H=observation(current)
                    K=np.linalg.solve(H@cov@H.T+R,H@cov).T
                    current=prior+K@(z-pred+H@(current-prior))
                A=np.eye(9)-K@H
                self.x=current; self.P=A@cov@A.T+K@R@K.T  # Joseph形式
        self.x[:3]=[*xy,0.]; self.x[3:6]=0.
        if not c.enabled or not c.retain_bias_on_fix:
            self.x[6:]=0.; self.P[6:,:]=0.; self.P[:,6:]=0.
            self.P[6:,6:]=np.eye(3)*c.initial_bias_sigma_mps2**2
        # 位置由本次UWB重新确定，速度为有不确定性的零速度先验。
        _,H=range_model(self.x,self.anchors,self.height)
        information=H[:,:3].T@H[:,:3]/c.range_sigma_m**2
        information[2,2]+=1/c.pressure_sigma_m**2
        position_cov=(np.linalg.solve(information,np.eye(3)) if np.linalg.matrix_rank(information)==3
                      else np.diag([c.initial_position_sigma_m**2]*2+[c.pressure_sigma_m**2]))
        bias_cov=self.P[6:,6:].copy()
        self.P=np.zeros((9,9)); self.P[:3,:3]=position_cov
        self.P[3:6,3:6]=np.eye(3)*c.segment_velocity_sigma_mps**2
        self.P[6:,6:]=bias_cov; self.P=.5*(self.P+self.P.T)
        if not np.isfinite(self.x).all() or not np.isfinite(self.P).all():
            raise ValueError('EKF 非有限状态')
        info.update(bias_after=self.x[6:].tolist(),position_anchored=True,surface_depth_m=0.)
        self.initialized=True; self.pause_ns=int(t_ns); self.last_update=info
        return info
