import tensorflow as tf
import boundaries as bd
import cut_profiles
import numpy as np
import time


def _delta_r2_eq_dt1(r_traj):
    return tf.reduce_sum(tf.square(r_traj[1:] - r_traj[:-1]))


def _delta_r2_eq_nobd(r_traj, dt=1):
    if dt >= len(r_traj): return tf.constant(0,r_traj.dtype)
    return tf.reduce_sum(tf.square(r_traj[dt:] - r_traj[:-dt])) / dt


def _delta_r2_ne_dt1(r_traj, i_traj):
    delta_r = tf.where(i_traj[1:] == i_traj[:-1], r_traj[1:] - r_traj[:-1], tf.cast(0, dtype=r_traj.dtype))
    return tf.reduce_sum(tf.square(delta_r))


def _delta_r2_ne_nobd(r_traj, i_traj, dt=1):
    delta_r = tf.where(i_traj[dt:] == i_traj[:-dt], r_traj[dt:] - r_traj[:-dt], tf.cast(0, dtype=r_traj.dtype))
    return tf.reduce_sum(tf.square(delta_r)) / dt

def _delta_r2_slow_exact(r_traj, b_traj, i_traj, dt=1):
    trajs, indices = tf.unique(i_traj)
    s=0
    for i in trajs:
        mask=i_traj==i
        s= s + _delta_r2(r_traj[mask], b_traj[mask], dt=dt)
    return s

def _delta_r2(r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None, dt=1):
    if dt == 1:
        if i_traj is None:
            return _delta_r2_eq_dt1(r_traj)
        else:
            return _delta_r2_ne_dt1(r_traj, i_traj)
    if dt > 1 and (b_traj is None) and (future_boundary is None) and (past_boundary is None):
        b_traj = tf.zeros_like(r_traj)
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    delta_t = tf.cast(dt, dtype=r_traj.dtype)
    i_future_boundary_crossed = (tf.cast(future_boundary.index2 > -1, dtype=r_traj.dtype) *
                                 tf.cast(future_boundary.delta_t2 <= delta_t, dtype=r_traj.dtype))
    i_past_boundary_crossed = (tf.cast(past_boundary.index2 > -1, dtype=r_traj.dtype) *
                               tf.cast(past_boundary.delta_t2 >= -delta_t, dtype=r_traj.dtype))

    # no crossing of boundaries
    if i_traj is None:
        loss = tf.reduce_sum((1 - i_past_boundary_crossed[dt:]) * (1 - i_future_boundary_crossed[:-dt]) *
                             tf.square(r_traj[dt:] - r_traj[:-dt]))
    else:
        loss = tf.reduce_sum((1 - i_past_boundary_crossed[dt:]) * (1 - i_future_boundary_crossed[:-dt]) *
                             tf.square(r_traj[dt:] - r_traj[:-dt]) *
                             tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype))

    # crossing of the future boundary
    loss += tf.reduce_sum(i_future_boundary_crossed * (1 - b_traj) * tf.square(future_boundary.r2 - r_traj) *
                          tf.cast(future_boundary.delta_i_to_end >= delta_t, dtype=r_traj.dtype))

    # crossing of the past boundary
    loss += tf.reduce_sum(i_past_boundary_crossed * (1 - b_traj) * tf.square(r_traj - past_boundary.r2) *
                          tf.cast(past_boundary.delta_i_from_start >= delta_t, dtype=r_traj.dtype))

    # transitions between boundaries
    loss += tf.reduce_sum(i_future_boundary_crossed * b_traj * (delta_t - future_boundary.delta_t2 + 1) *
                          tf.square(future_boundary.r2 - r_traj))# *

    return loss / dt


ldt0 = [2 ** i for i in range(16)]


def _comp_max_delta_zq(r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None, ldt=None):
    if ldt is None:
        ldt = ldt0
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    max_m2 = 0, 0
    max_abs = 0, 0
    for dt in ldt:
        lx, lz = cut_profiles.comp_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, dt=tf.constant(dt))
        lz = lz[:-1].numpy()
        m1 = np.mean(lz)
        mabs = np.max(abs(lz - m1))
        if mabs > max_abs[0]:
            max_abs = mabs, dt
        m2 = np.mean((lz - m1) ** 2) ** 0.5
        if m2 > max_m2[0]:
            max_m2 = m2, dt
    return max_abs[0], max_abs[1], max_m2[0], max_m2[1]


def _mse(r_traj, b_traj=None, i_traj=None, future_boundary=None):
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    return tf.math.reduce_mean(tf.where(future_boundary.index > -1, future_boundary.r-r_traj, 0)**2)



def _mse_eq(r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None):
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)
    return (tf.math.reduce_mean(tf.where(future_boundary.index > -1, future_boundary.r - r_traj, 0)**2) +
            tf.math.reduce_mean(tf.where(past_boundary.index > -1, r_traj-past_boundary.r, 0) ** 2))/2


def _cross_entropy(r_traj, b_traj=None, i_traj=None, future_boundary=None, eps=1e-6):
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    return -tf.math.reduce_mean(tf.where(future_boundary.index > -1,
                               future_boundary.r*tf.math.log(r_traj+eps)+(1-future_boundary.r)*tf.math.log(1-r_traj+eps), 0))

def _auc(r_traj, b_traj=None, i_traj=None, future_boundary=None):
    import sklearn.metrics
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)

    ok = future_boundary.index > -1
    return sklearn.metrics.roc_auc_score(future_boundary.r[ok], r_traj[ok])

def _delta_x(r1_traj, r2_traj):
    return tf.math.reduce_mean((r1_traj-r2_traj) ** 2) ** 0.5


def _low_bound_delta_r2_eq(r_traj, b_traj, i_traj=None, future_boundary=None):
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    return tf.reduce_sum(tf.where(future_boundary.index[1:] > -1,
                                  (future_boundary.r[1:] - future_boundary.r[:-1]) ** 2, 0))


def _imfpt_eq(r_traj, dt=1):
    return tf.math.reduce_sum(tf.square(r_traj[dt:] - r_traj[:-dt]) - 2 * dt * (r_traj[:-dt] + r_traj[dt:]))


def _min_imfpt_eq(b_traj, i_traj=None, future_boundary=None):
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(b_traj, b_traj, i_traj=i_traj)
    return tf.reduce_sum(tf.where(future_boundary.index > -1, b_traj * (future_boundary.delta_t + 1) ** 2, 0))


def delta_r2(rc):
    return _delta_r2(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary, dt=1)

def max_delta_zq(rc):
    return _comp_max_delta_zq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[0]

def max_sd_zq(rc):
    return _comp_max_delta_zq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[2]**0.5

def mse(rc):
    return _mse(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary)

def mse_eq(rc):
    return _mse_eq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)

def cross_entropy(rc):
    return _cross_entropy(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary)

def auc(rc):
    return _auc(rc.r_traj, future_boundary=rc.future_boundary)

def low_bound_delta_r2_eq(rc):
    return _low_bound_delta_r2_eq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary)

def time_elapsed(rc):
    return time.time() - rc.time_start

def delta_x(rc):
    return _delta_x(rc.r_traj, rc.r_traj_old)

def iter(rc):
    return rc.iter


metric2function = {'delta_r2': delta_r2, 'max_delta_zq': max_delta_zq,
                        'mse': mse, 'mse_eq': mse_eq, 'iter': iter,
                        'cross_entropy': cross_entropy, 'time_elapsed': time_elapsed,
                        'delta_x': delta_x, 'auc': auc,
                        'max_sd_zq': max_sd_zq}
metrics_short_name = {'delta_r2': 'dr2', 'max_delta_zq': 'maxdzq', 'mse': 'mse', 'mse_eq': 'mseeq',
                           'cross_entropy': 'xent', 'time_elapsed': 'time', 'delta_x': '|dx|',
                           'iter': '#', 'auc': 'auc', 'max_sd_zq': 'sdzq'}
