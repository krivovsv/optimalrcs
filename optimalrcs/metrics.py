import cupy as cp
from . import boundaries as bd
from . import cut_profiles
import numpy as np
import time


def _delta_r2_eq_dt1(r_traj):
    """
    Compute the mean squared displacement (MSD) for a trajectory with dt=1.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.

    Returns
    -------
    float
        The mean squared displacement.
    """
    return cp.sum(cp.square(r_traj[1:] - r_traj[:-1])).get()


def _delta_r2_eq_nobd(r_traj, dt=1):
    """
    Compute the mean squared displacement (MSD) for a trajectory without boundaries.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    dt : int, optional
        The time step, by default 1.

    Returns
    -------
    float
        The mean squared displacement.
    """
    if dt >= len(r_traj): return 0
    return cp.sum(cp.square(r_traj[dt:] - r_traj[:-dt])).get() / dt


def _delta_r2_ne_dt1(r_traj, i_traj):
    """
    Compute the mean squared displacement (MSD) for a trajectory with non-equilibrium conditions and dt=1.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    i_traj : np.ndarray
        The index trajectory.

    Returns
    -------
    float
        The mean squared displacement.
    """
    delta_r = cp.where(i_traj[1:] == i_traj[:-1], r_traj[1:] - r_traj[:-1], 0)
    return cp.sum(cp.square(delta_r)).get()


def _delta_r2_ne_nobd(r_traj, i_traj, dt=1):
    """
    Compute the mean squared displacement (MSD) for a trajectory with non-equilibrium conditions and without boundaries.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    i_traj : np.ndarray
        The index trajectory.
    dt : int, optional
        The time step, by default 1.

    Returns
    -------
    float
        The mean squared displacement.
    """
    delta_r = cp.where(i_traj[dt:] == i_traj[:-dt], r_traj[dt:] - r_traj[:-dt], 0)
    return cp.sum(cp.square(delta_r)).get() / dt

def _delta_r2_slow_exact(r_traj, b_traj, i_traj, dt=1):
    """
    Compute the mean squared displacement (MSD) for a trajectory with slow exact calculation.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The boundary trajectory.
    i_traj : np.ndarray
        The index trajectory.
    dt : int, optional
        The time step, by default 1.

    Returns
    -------
    float
        The mean squared displacement.
    """
    trajs, indices = cp.unique(i_traj)
    s=0
    for i in trajs:
        mask=i_traj==i
        s= s + _delta_r2(r_traj[mask], b_traj[mask], dt=dt)
    return s

def _delta_r2(r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None, dt=1):
    """
    Compute the mean squared displacement (MSD) for a trajectory.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray, optional
        The boundary trajectory, by default None.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    past_boundary : bd.PastBoundary, optional
        The past boundary object, by default None.
    dt : int, optional
        The time step, by default 1.

    Returns
    -------
    float
        The mean squared displacement.
    """
    if dt == 1:
        if i_traj is None:
            return _delta_r2_eq_dt1(r_traj)
        else:
            return _delta_r2_ne_dt1(r_traj, i_traj)
    if dt > 1 and (b_traj is None) and (future_boundary is None) and (past_boundary is None):
        b_traj = cp.zeros_like(r_traj)
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    i_future_boundary_crossed = ((future_boundary.index2 > -1).astype(r_traj.dtype) *
                                 (future_boundary.delta_t2 <= dt).astype(r_traj.dtype))
    i_past_boundary_crossed = ((past_boundary.index2 > -1).astype(r_traj.dtype) *
                               (past_boundary.delta_t2 >= -dt).astype(r_traj.dtype))

    # no crossing of boundaries
    if i_traj is None:
        loss = cp.sum((1 - i_past_boundary_crossed[dt:]) * (1 - i_future_boundary_crossed[:-dt]) *
                             cp.square(r_traj[dt:] - r_traj[:-dt]))
    else:
        loss = cp.sum((1 - i_past_boundary_crossed[dt:]) * (1 - i_future_boundary_crossed[:-dt]) *
                             cp.square(r_traj[dt:] - r_traj[:-dt]) *
                             (i_traj[dt:] == i_traj[:-dt]).astype(r_traj.dtype))

    # crossing of the future boundary
    loss += cp.sum(i_future_boundary_crossed * (1 - b_traj) * cp.square(future_boundary.r2 - r_traj) *
                          (future_boundary.delta_i_to_end >= dt).astype(r_traj.dtype))

    # crossing of the past boundary
    loss += cp.sum(i_past_boundary_crossed * (1 - b_traj) * cp.square(r_traj - past_boundary.r2) *
                          (past_boundary.delta_i_from_start >= dt).astype(r_traj.dtype))

    # transitions between boundaries
    loss += cp.sum(i_future_boundary_crossed * b_traj * (dt - future_boundary.delta_t2 + 1) *
                          cp.square(future_boundary.r2 - r_traj))# *

    return loss.get() / dt


ldt0 = [2 ** i for i in range(16)]


def _comp_max_delta_zq(r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None, w_traj=None, ldt=None):
    """
    Compute the maximum delta and standard deviation of ZQ profiles.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray, optional
        The boundary trajectory, by default None.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    past_boundary : bd.PastBoundary, optional
        The past boundary object, by default None.
    w_traj : np.ndarray, optional
        The weight trajectory, by default None.
    ldt : list, optional
        List of time steps for ZQ profile computation, by default None.

    Returns
    -------
    tuple
        A tuple containing the maximum absolute deviation, corresponding dt, maximum standard deviation,
        corresponding dt, and total variance.
    """
    if ldt is None:
        ldt = ldt0
    #if cp.is_tensor(r_traj):
    #    r_traj = r_traj.get()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    max_m2 = 0, 0
    max_abs = 0, 0
    total_m2 = 0
    for dt in ldt:
        lx, lz = cut_profiles.comp_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, w_traj=w_traj, dt=dt)
        lz = lz[:-1].get()
        m1 = np.mean(lz)
        mabs = np.max(abs(lz - m1))
        if mabs > max_abs[0]:
            max_abs = mabs, dt
        m2 = np.mean((lz - m1) ** 2) ** 0.5
        total_m2 += np.mean((lz - m1) ** 2)
        if m2 > max_m2[0]:
            max_m2 = m2, dt
    return max_abs[0], max_abs[1], max_m2[0], max_m2[1], total_m2

def _comp_max_grad_zq(r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, d=100, 
                      log_scale=False, log_scale_pmin=1e-4):
    """
    Compute the maximum gradient of ZQ profiles.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray, optional
        The boundary trajectory, by default None.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    past_boundary : bd.PastBoundary, optional
        The past boundary object, by default None.
    ldt : list, optional
        List of time steps for ZQ profile computation, by default None.
    d : int, optional
        The window size for gradient calculation, by default 100.
    log_scale : bool, optional
        Whether to use a logarithmic scale, by default False.
    log_scale_pmin : float, optional
        The minimum value for the logarithmic scale, by default 1e-4.

    Returns
    -------
    tuple
        A tuple containing the maximum gradient and corresponding dt.
    """
    if ldt is None:
        ldt = ldt0
    #if cp.is_tensor(r_traj):
    #    r_traj = r_traj.get()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    stats = 0, 0
    for dt in ldt:
        lx, lz = cut_profiles.comp_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, dt=dt, 
                                      log_scale=log_scale, log_scale_pmin=log_scale_pmin)
        lz = lz[:-1].get()
        lx = lx[:-1].get()
        avgrad = np.mean((lz[d:] - lz[:-d]) ** 2/(lx[d:]-lx[:-d])) ** 0.5
        if avgrad > stats[0]: stats = avgrad, dt
    return stats

def _comp_max_delta_zt(r_traj, b_traj, t_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None):
    """
    Compute the maximum delta and standard deviation of ZT profiles.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The boundary trajectory.
    t_traj : np.ndarray
        The time trajectory.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    past_boundary : bd.PastBoundary, optional
        The past boundary object, by default None.
    ldt : list, optional
        List of time steps for ZT profile computation, by default None.

    Returns
    -------
    tuple
        A tuple containing the maximum absolute deviation, corresponding dt, maximum standard deviation,
        and corresponding dt.
    """
    if ldt is None:
        ldt = ldt0
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, t_traj=t_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, t_traj=t_traj, i_traj=i_traj)

    max_m2 = 0, 0
    max_abs = 0, 0
    for dt in ldt:
        lx, lz = cut_profiles.comp_zt(r_traj, b_traj, t_traj, i_traj, future_boundary, past_boundary, dt=dt)
        lz = lz[:-1].get()
        m1 = np.mean(lz)
        mabs = np.max(abs(lz - m1))
        if mabs > max_abs[0]:
            max_abs = mabs, dt
        m2 = np.mean((lz - m1) ** 2) ** 0.5
        if m2 > max_m2[0]:
            max_m2 = m2, dt
    return max_abs[0], max_abs[1], max_m2[0], max_m2[1]

def _comp_max_grad_zt(r_traj, b_traj, t_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, d=100, 
                      log_scale=False, log_scale_tmin=1e-4):
    """
    Compute the maximum gradient of ZT profiles.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The boundary trajectory.
    t_traj : np.ndarray
        The time trajectory.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    past_boundary : bd.PastBoundary, optional
        The past boundary object, by default None.
    ldt : list, optional
        List of time steps for ZT profile computation, by default None.
    d : int, optional
        The window size for gradient calculation, by default 100.
    log_scale : bool, optional
        Whether to use a logarithmic scale, by default False.
    log_scale_tmin : float, optional
        The minimum value for the logarithmic scale, by default 1e-4.

    Returns
    -------
    tuple
        A tuple containing the maximum gradient and corresponding dt.
    """
    if ldt is None:
        ldt = ldt0
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    stats = 0, 0
    for dt in ldt:
        lx, lz = cut_profiles.comp_zt(r_traj, b_traj, t_traj, i_traj, future_boundary, past_boundary, dt=dt, 
                                      log_scale=log_scale, log_scale_tmin=log_scale_tmin)
        lz = lz[:-1].get()
        lx = lx[:-1].get()
        avgrad = np.mean((lz[d:] - lz[:-d]) ** 2/(lx[d:]-lx[:-d])) ** 0.5
        if avgrad > stats[0]: stats = avgrad, dt
    return stats


def _mse(r_traj, b_traj=None, i_traj=None, future_boundary=None):
    """
    Compute the mean squared error (MSE) for a trajectory.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray, optional
        The boundary trajectory, by default None.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.

    Returns
    -------
    float
        The mean squared error.
    """
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    return cp.mean(cp.where(future_boundary.index > -1, future_boundary.r-r_traj, 0)**2)



def _mse_eq(r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None):
    """
    Compute the mean squared error (MSE) for a trajectory with equilibrium conditions.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray, optional
        The boundary trajectory, by default None.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    past_boundary : bd.PastBoundary, optional
        The past boundary object, by default None.

    Returns
    -------
    float
        The mean squared error for equilibrium conditions.
    """
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)
    return (cp.mean(cp.where(future_boundary.index > -1, future_boundary.r - r_traj, 0)**2) +
            cp.mean(cp.where(past_boundary.index > -1, r_traj-past_boundary.r, 0) ** 2))/2


def _cross_entropy(r_traj, b_traj=None, i_traj=None, future_boundary=None, eps=1e-6):
    """
    Compute the cross entropy for a trajectory.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray, optional
        The boundary trajectory, by default None.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    eps : float, optional
        Small value to avoid log(0), by default 1e-6.

    Returns
    -------
    float
        The cross entropy.
    """
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    return -cp.mean(cp.where(future_boundary.index > -1,
                               future_boundary.r*cp.log(r_traj+eps)+(1-future_boundary.r)*cp.log(1-r_traj+eps), 0))


def _auc(r_traj, b_traj=None, i_traj=None, future_boundary=None, skip_boundaries=False):
    """
    Compute the area under the curve (AUC) for a trajectory.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray, optional
        The boundary trajectory, by default None.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.
    skip_boundaries : bool, optional
        Whether to skip boundaries in the calculation, by default False.

    Returns
    -------
    float
        The area under the curve.
    """
    import sklearn.metrics
    #if cp.is_tensor(r_traj):
    #    r_traj = r_traj.get()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)

    ok = future_boundary.index > -1
    if skip_boundaries: ok = ok & b_traj==0 
    return sklearn.metrics.roc_auc_score(future_boundary.r[ok].get(), r_traj[ok].get())

def _delta_x(r1_traj, r2_traj):
    """
    Compute the Euclidean distance between two trajectories.

    Parameters
    ----------
    r1_traj : np.ndarray
        The first reaction coordinate trajectory.
    r2_traj : np.ndarray
        The second reaction coordinate trajectory.

    Returns
    -------
    float
        The Euclidean distance.
    """
    return cp.mean((r1_traj-r2_traj) ** 2).get() ** 0.5


def _low_bound_delta_r2_eq(r_traj, b_traj, i_traj=None, future_boundary=None):
    """
    Compute the lower bound of mean squared displacement (MSD) for equilibrium conditions.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The boundary trajectory.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.

    Returns
    -------
    float
        The lower bound of mean squared displacement.
    """
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    return cp.sum(cp.where(future_boundary.index[1:] > -1,
                                  (future_boundary.r[1:] - future_boundary.r[:-1]) ** 2, 0))


def _imfpt_eq(r_traj, dt=1):
    """
    Compute the mean first passage time (MFPT) for equilibrium conditions.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    dt : int, optional
        The time step, by default 1.

    Returns
    -------
    float
        The mean first passage time.
    """
    return cp.sum(cp.square(r_traj[dt:] - r_traj[:-dt]) - 2 * dt * (r_traj[:-dt] + r_traj[dt:]))


def _min_imfpt_eq(b_traj, i_traj=None, future_boundary=None):
    """
    Compute the minimum mean first passage time (MFPT) for equilibrium conditions.

    Parameters
    ----------
    b_traj : np.ndarray
        The boundary trajectory.
    i_traj : np.ndarray, optional
        The index trajectory, by default None.
    future_boundary : bd.FutureBoundary, optional
        The future boundary object, by default None.

    Returns
    -------
    float
        The minimum mean first passage time.
    """
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(b_traj, b_traj, i_traj=i_traj)
    return cp.reduce_sum(cp.where(future_boundary.index > -1, b_traj * (future_boundary.delta_t + 1) ** 2, 0))


def delta_r2(rc):
    """
    Compute the mean squared displacement (MSD) for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The mean squared displacement.
    """
    return _delta_r2(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary, dt=1)

def max_delta_zq(rc):
    """
    Compute the maximum delta of ZQ profiles for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The maximum delta.
    """
    return _comp_max_delta_zq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[0]

def max_sd_zq(rc):
    """
    Compute the maximum standard deviation of ZQ profiles for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The maximum standard deviation.
    """
    return _comp_max_delta_zq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[2]

def total_sd_zq(rc):
    """
    Compute the total variance of ZQ profiles for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The total variance.
    """
    return _comp_max_delta_zq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[4]

def max_grad_zq(rc):
    """
    Compute the maximum gradient of ZQ profiles for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The maximum gradient.
    """
    return _comp_max_grad_zq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[0]

def max_sd_zt(rc):
    """
    Compute the maximum standard deviation of ZT profiles for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The maximum standard deviation.
    """
    return _comp_max_delta_zt(rc.r_traj, rc.b_traj, rc.t_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[2]

def max_grad_zt(rc):
    """
    Compute the maximum gradient of ZT profiles for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The maximum gradient.
    """
    return _comp_max_grad_zt(rc.r_traj, rc.b_traj, rc.t_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)[0]

def mse(rc):
    """
    Compute the mean squared error (MSE) for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The mean squared error.
    """
    return _mse(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary)

def mse_eq(rc):
    """
    Compute the mean squared error (MSE) for a reaction coordinate object with equilibrium conditions.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The mean squared error for equilibrium conditions.
    """
    return _mse_eq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary, rc.past_boundary)

def cross_entropy(rc):
    """
    Compute the cross entropy for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The cross entropy.
    """
    return _cross_entropy(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary)

def auc(rc):
    """
    Compute the area under the curve (AUC) for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The area under the curve.
    """
    return _auc(rc.r_traj, future_boundary=rc.future_boundary)

def low_bound_delta_r2_eq(rc):
    """
    Compute the lower bound of mean squared displacement (MSD) for a reaction coordinate object with equilibrium conditions.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The lower bound of mean squared displacement.
    """
    return _low_bound_delta_r2_eq(rc.r_traj, rc.b_traj, rc.i_traj, rc.future_boundary)

def time_elapsed(rc):
    """
    Compute the elapsed time for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The elapsed time.
    """
    return time.time() - rc.time_start

def delta_x(rc):
    """
    Compute the Euclidean distance between the current and old trajectories for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The Euclidean distance.
    """
    return _delta_x(rc.r_traj, rc.r_traj_old)

def iter(rc):
    """
    Get the iteration count for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    int
        The iteration count.
    """
    return rc.iter

def i_mfpt(rc):
    """
    Compute the mean first passage time (MFPT) functional for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The mean first passage time.
    """
    dti=(cp.square(rc.r_traj[1:] - rc.r_traj[:-1]) - 2 *
                                    (rc.t_traj[1:] - rc.t_traj[:-1]) * (rc.r_traj[:-1] + rc.r_traj[1:]))
    if rc.i_traj is not None:
        dti=dti*(rc.i_traj[1:]==rc.i_traj[:-1]).astype(rc.prec)
    return cp.mean(dti).get()
    
def low_bound_i_mfpt_eq(rc):
    """
    Compute the lower bound of mean first passage time (MFPT) for a reaction coordinate object with equilibrium conditions.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The lower bound of mean first passage time.
    """
    dti = rc.future_boundary.delta_t[1:][rc.b_traj[:-1] > 0][:-1] + 1
    return -sum(dti**2) / len(rc.b_traj)

def max_rc(rc):
    """
    Compute the maximum value of the reaction coordinate for a reaction coordinate object.

    Parameters
    ----------
    rc : object
        The reaction coordinate object containing trajectory data.

    Returns
    -------
    float
        The maximum value of the reaction coordinate.
    """
    return cp.max(rc.r_traj)



metric2function = {'delta_r2': delta_r2, 'max_delta_zq': max_delta_zq,
                        'mse': mse, 'mse_eq': mse_eq, 'iter': iter,
                        'cross_entropy': cross_entropy, 'time_elapsed': time_elapsed,
                        'delta_x': delta_x, 'auc': auc, 'imfpt': i_mfpt,
                        'max_sd_zq': max_sd_zq, 'max_grad_zq': max_grad_zq,
                        'max_sd_zt': max_sd_zt, 'max_grad_zt': max_grad_zt, 'max_rc': max_rc}

metrics_short_name = {'delta_r2': 'dr2', 'max_delta_zq': 'maxdzq', 'mse': 'mse', 'mse_eq': 'mseeq',
                           'cross_entropy': 'xent', 'time_elapsed': 'time', 'delta_x': '|dx|',
                           'iter': '#', 'auc': 'auc', 'max_sd_zq': 'sdzq', 'max_grad_zq': 'dzq',
                           'max_sd_zt': 'sdzt', 'max_grad_zt': 'dzt', 'imfpt': 'it', 'max_rc': 'max'}
