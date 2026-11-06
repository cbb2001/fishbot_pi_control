"""Bench-only intermittent rest experiment. No actuator or hardware imports.

Rest information is applied ONCE at t4, 2*t4, ... after initialization.
Between boundaries every estimator predicts freely, even though the bench is
physically stationary. Position anchoring is a separately labelled oracle.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from .imu_static_calibration_20261031 import StaticESKF, rotation, finite_vec


class VendorIntegrator:
    """Navigation's vendor quaternion + two-stage LPF + trapezoid XY integral.

    This bench diagnostic also integrates Z; production navigation uses pressure
    for Z. XY is equivalent to InertialEstimator without its pool yaw rotation.
    """
    def __init__(self,bias,tau=.2):
        self.bias=np.array(bias,dtype=float).copy();self.tau=float(tau)
        self.p=np.zeros(3);self.v=np.zeros(3);self.a1=None;self.a2=None

    def predict(self,acc,R,dt):
        a=R@(acc-self.bias)-[0,0,9.80665]
        if self.a1 is None:self.a1=a.copy();self.a2=a.copy()
        previous=self.a2.copy()
        gain=1. if self.tau==0 else -np.expm1(-dt/self.tau)
        self.a1+=gain*(a-self.a1);self.a2+=gain*(self.a1-self.a2)
        old=self.v.copy();self.v+=.5*(previous+self.a2)*dt
        self.p+=.5*(old+self.v)*dt

    def reset_velocity(self,position=False,bias=None):
        self.v.fill(0.)
        if position:self.p.fill(0.)
        if bias is not None:self.bias=np.array(bias).copy()
        # Exactly the navigation surface-fix policy: clear both filter stages.
        self.a1=np.zeros(3);self.a2=np.zeros(3)


class PeriodicStaticExperiment:
    METHODS=('free_vendor','periodic_velocity','periodic_velocity_bias',
             'periodic_surface_anchor','periodic_eskf_zupt')

    def __init__(self,noise,*,t4=10.,window_s=2.,warmup_s=5.,startup_s=5.,tau=.2,
                 externally_confirmed_static=False):
        if not 0 < window_s <= min(t4,startup_s) or warmup_s<0:
            raise ValueError('Need 0 < window <= t4 and startup duration')
        self.noise=noise;self.t4=float(t4);self.window_s=float(window_s)
        self.warmup_s=float(warmup_s);self.startup_s=float(startup_s);self.tau=tau
        self.confirmed=bool(externally_confirmed_static)
        self.rows=deque();self.models={};self.origin=None;self.last_t=None
        self.boundary=1;self.corrections=0;self.initialization=None

    def snapshot(self):
        return {name:dict(p_m=model.p.tolist(),v_mps=model.v.tolist(),
                         bias_body_mps2=(model.ba if isinstance(model,StaticESKF) else model.bias).tolist())
                for name,model in self.models.items()}

    def _quiet(self,rows,bg):
        if len(rows)<2 or rows[-1][0]-rows[0][0]<self.window_s-.02:return False
        acc=np.array([r[1] for r in rows]);gyro=np.array([r[2] for r in rows])
        return bool(np.max(acc.std(0))<self.noise['quiet_acc_std_max_mps2'] and
                    np.linalg.norm(gyro.mean(0)-bg)<self.noise['quiet_gyro_norm_max_radps'])

    def step(self,t,data):
        t=float(t)
        if self.last_t is not None and not 0<t-self.last_t<=.05:
            raise ValueError('Non-contiguous IMU time')
        acc=finite_vec(data['acc_mps2']);gyro=finite_vec(data['gyro_radps']);R=rotation(data['quat'])
        residual=acc-R[2,:]*9.80665
        self.rows.append((t,acc,gyro,residual))
        keep=max(self.startup_s,self.window_s)
        while self.rows and self.rows[0][0]<t-keep-1e-8:self.rows.popleft()
        event=None
        if self.origin is None:
            if t+1e-8>=self.warmup_s+self.startup_s:
                init=[r for r in self.rows if r[0]+1e-8>=self.warmup_s]
                if not self.confirmed:raise ValueError('Bench initialization needs explicit external stillness')
                if init[-1][0]-init[0][0]<self.startup_s-.02:raise ValueError('Incomplete startup window')
                ba=np.mean([r[3] for r in init],axis=0);self.bg=np.mean([r[2] for r in init],axis=0)
                if not self._quiet(init,self.bg):raise ValueError('Startup IMU not quiet')
                self.models={name:VendorIntegrator(ba,self.tau) for name in self.METHODS[:-1]}
                self.models['periodic_eskf_zupt']=StaticESKF(data['quat'],ba,self.bg,self.noise)
                # Initialize navigation LPF at the actual current acceleration.
                for model in self.models.values():
                    if isinstance(model,VendorIntegrator):model.predict(acc,R,0.)
                self.origin=t;self.initialization=dict(event='initialized',capture_time_s=t,
                    accel_bias_body_mps2=ba.tolist(),gyro_bias_body_radps=self.bg.tolist(),samples=len(init))
                event=self.initialization
            self.last_t=t;self.previous=(acc,gyro)
            return event
        dt=t-self.last_t
        for model in self.models.values():
            if isinstance(model,StaticESKF):
                model.predict(.5*(self.previous[0]+acc),.5*(self.previous[1]+gyro),dt)
            else:model.predict(acc,R,dt)
        elapsed=t-self.origin
        if elapsed+1e-8>=self.boundary*self.t4:
            # One causal boundary update; no hidden updates during the segment.
            rows=[r for r in self.rows if r[0]+1e-8>=t-self.window_s]
            quiet=self._quiet(rows,self.bg)
            accepted=self.confirmed and quiet
            before=self.snapshot()
            local_bias=np.mean([r[3] for r in rows],axis=0)
            local_gyro=np.mean([r[2] for r in rows],axis=0)
            if accepted:
                self.models['periodic_velocity'].reset_velocity()
                self.models['periodic_velocity_bias'].reset_velocity(bias=local_bias)
                self.models['periodic_surface_anchor'].reset_velocity(position=True)
                self.models['periodic_eskf_zupt'].update_known_rest(local_gyro,
                    externally_confirmed_static=True,quiet=True)
                self.corrections+=1
            event=dict(event='periodic_boundary',cycle=self.boundary,time_s=elapsed,
                       scheduled_time_s=self.boundary*self.t4,accepted=accepted,
                       window_start_capture_s=rows[0][0],window_end_capture_s=t,window_samples=len(rows),
                       local_effective_bias_mps2=local_bias.tolist(),before=before,after=self.snapshot())
            self.boundary+=1
        self.last_t=t;self.previous=(acc,gyro)
        return event
