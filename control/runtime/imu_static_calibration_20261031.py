"""Opt-in IMU calibration utilities; no hardware access or navigation side effects.

Conventions: wxyz, body-to-world, static specific force R.T @ [0,0,+g].
One-pose accelerometer bias is EFFECTIVE bias (also contains attitude/g error).
ESKF: nominal p,v,q,ba,bg; right-local error [dp,dv,dtheta,dba,dbg].
Reference: Sola, https://arxiv.org/abs/1711.02508 .
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or not .8 <= n <= 1.2:
        raise ValueError('Expected finite unit quaternion wxyz')
    return q/n


def quat_mul(a, b):
    return np.r_[a[0]*b[0]-a[1:]@b[1:], a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def quat_exp(v):
    angle = np.linalg.norm(v)
    return np.r_[np.cos(angle/2), v*(.5 if angle < 1e-10 else np.sin(angle/2)/angle)]


def rotation(q):
    w, x, y, z = quat_normalize(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def finite_vec(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError('Expected finite three-vector')
    return x


class FrozenIMUCalibration:
    """Apply one stored body bias, then actual current quaternion, then gravity.

    Do not also subtract the navigation session's b_body: that double-corrects.
    Gyro correction affects raw gyro only, never the vendor's existing quaternion.
    """
    def __init__(self, params):
        self.params = params
        if params.get('quaternion_order') != 'wxyz' or params.get('rotation') != 'body_to_world':
            raise ValueError('Calibration convention mismatch')
        self.ba = finite_vec(params['accel_bias_body_mps2'])
        self.bg = finite_vec(params['gyro_bias_body_radps'])
        self.g = float(params['gravity_mps2'])
        if not np.isfinite(self.g) or self.g <= 0:
            raise ValueError('Invalid gravity')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf8')))

    def correct(self, data):
        return (rotation(data['quat']) @ (finite_vec(data['acc_mps2'])-self.ba)-[0, 0, self.g],
                finite_vec(data['gyro_radps'])-self.bg)


class StaticESKF:
    """15-state error-state EKF, raw gyro propagation and optional known-rest aid.

    No vendor attitude measurement updates; no hidden velocity clamp or leakage.
    At rest, apply v=0 and gyro=bg at 10 Hz. Bias/attitude are not fully observable
    from one pose. Process densities are engineering estimates, not Allan fits.
    Never set externally_confirmed_static from an IMU quietness test alone.
    """
    def __init__(self, quat, ba, bg, noise, gravity=9.80665):
        self.q = quat_normalize(quat)
        self.ba, self.bg = finite_vec(ba).copy(), finite_vec(bg).copy()
        self.p, self.v = np.zeros(3), np.zeros(3)
        self.g = float(gravity)
        self.noise = noise
        self.P = np.diag(np.r_[np.full(3, .01**2), np.full(3, .02**2),
                              np.full(3, np.deg2rad(.1)**2),
                              np.full(3, .02**2), np.full(3, .003**2)])
        self.update_count = 0

    def predict(self, acc, gyro, dt):
        if not 0 < dt <= .05:
            raise ValueError('IMU dt must be in (0, .05] seconds')
        f, w = finite_vec(acc)-self.ba, finite_vec(gyro)-self.bg
        r = rotation(self.q)
        r_mid = rotation(quat_mul(self.q, quat_exp(w*dt*.5)))
        a = r_mid @ f - [0, 0, self.g]
        self.p += self.v*dt + .5*a*dt*dt
        self.v += a*dt
        self.q = quat_normalize(quat_mul(self.q, quat_exp(w*dt)))
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -r @ skew(f)
        F[3:6, 9:12] = -r
        F[6:9, 6:9] = -skew(w)
        F[6:9, 12:15] = -np.eye(3)
        Phi = np.eye(15) + F*dt + .5*(F@F)*dt*dt
        G = np.zeros((15, 12))
        G[3:6, 0:3] = r
        G[6:9, 3:6] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        density = np.r_[self.noise['accel_density'], self.noise['gyro_density'],
                        self.noise['accel_bias_rw'], self.noise['gyro_bias_rw']]
        # Midpoint mapping includes velocity-noise contribution to position.
        L = (np.eye(15)+.5*F*dt) @ G
        Q = (L*density**2) @ L.T * dt
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = .5*(self.P+self.P.T)

    def _update(self, residual, H, variances):
        S = H @ self.P @ H.T + np.diag(variances)
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ residual
        I_KH = np.eye(15)-K@H
        self.P = I_KH @ self.P @ I_KH.T + (K*variances)@K.T
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = quat_normalize(quat_mul(self.q, quat_exp(dx[6:9])))
        self.ba += dx[9:12]
        self.bg += dx[12:15]
        reset = np.eye(15)
        reset[6:9, 6:9] -= .5*skew(dx[6:9])
        self.P = reset @ self.P @ reset.T
        self.P = .5*(self.P+self.P.T)

    def update_known_rest(self, mean_gyro, *, externally_confirmed_static=False, quiet=False):
        if not externally_confirmed_static or not quiet:
            return False
        H = np.zeros((6, 15))
        H[:3, 3:6] = np.eye(3)
        H[3:, 12:15] = np.eye(3)
        residual = np.r_[-self.v, finite_vec(mean_gyro)-self.bg]
        variances = np.r_[np.full(3, self.noise['zupt_velocity_std_mps']**2),
                          np.asarray(self.noise['rest_gyro_std_radps'])**2]
        self._update(residual, H, variances)
        self.update_count += 1
        return True
