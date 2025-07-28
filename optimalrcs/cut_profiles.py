"""
cut_profiles.py

This module provides functions for computing various cut-based reaction coordinate profiles
from trajectory data, including Z_C1, Z_q, Z_t, and Z_{C,a} profiles. These profiles are used
to analyze the dynamics of systems with respect to boundaries defined in reaction coordinate space.

The module supports both regularly and irregularly sampled trajectories, and allows for
custom weighting, indexing, and boundary handling.

Functions
---------
comp_zc1 :
    Computes the Z_C1(r, dt) profile for regularly sampled trajectories.

comp_zc1_irreg :
    Computes the Z_C1(r, dt) profile for irregularly sampled trajectories with compensation.

comp_zq :
    Computes the Z_q(r, dt) profile, a simplified cut-based profile.

comp_zt :
    Computes the Z_t(r, dt) profile using mean first passage time (MFPT) information.

comp_zca :
    Computes the Z_{C,a}(r, dt) profile with a general exponent `a`.

All functions return a tuple of bin edges and the corresponding profile values.
"""


import numpy as np
import tensorflow as tf
from . import boundaries as bd


def comp_zc1(r_traj: np.ndarray, b_traj: np.ndarray, future_boundary: bd.FutureBoundary = None,
             past_boundary: bd.PastBoundary = None, i_traj: np.ndarray = None, w_traj : np.ndarray = None,
             dt = 1, nbins = 1000):
    """
    Compute the Z_C1(r, dt) cut-based profile for a trajectory.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The boundary indicator values for the trajectory.
    future_boundary : bd.FutureBoundary, optional
        Boundary object for future boundaries. If None, it will be constructed from inputs.
    past_boundary : bd.PastBoundary, optional
        Boundary object for past boundaries. If None, it will be constructed from inputs.
    i_traj : np.ndarray, optional
        The index trajectory to distinguish between different trajectories.
    w_traj : np.ndarray, optional
        The weight trajectory for reweighting.
    dt : int, optional
        The lag time to compute Z_C1(r, dt). Default is 1.
    nbins : int, optional
        Number of bins for the profile. Default is 1000.

    Returns
    -------
    bin_edges : tf.Tensor
        The edges of the histogram bins (length nbins + 1).
    zc1 : tf.Tensor
        The Z_C1 profile values (length nbins).
    """
    r_min = tf.math.reduce_min(r_traj)
    r_max = tf.math.reduce_max(r_traj)
    bin_edges = tf.linspace(r_min, r_max, nbins + 1)
    delta_t_prec = tf.cast(dt, dtype=r_traj.dtype)

    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    delta_t = tf.cast(dt, dtype=r_traj.dtype)
    i_future_boundary_crossed = (tf.cast(future_boundary.index2 > -1, dtype=r_traj.dtype) *
                                 tf.cast(future_boundary.delta_i2 <= delta_t, dtype=r_traj.dtype))
    i_past_boundary_crossed = (tf.cast(past_boundary.index2 > -1, dtype=r_traj.dtype) *
                               tf.cast(past_boundary.delta_i2 >= -delta_t, dtype=r_traj.dtype))

    # no crossings of boundaries
    delta_r = (1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) * (r_traj[dt:] - r_traj[:-dt])
    if i_traj is not None:
        delta_r = delta_r * tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj[:-dt]
    bin_indices_r = tf.searchsorted(bin_edges, r_traj, side='right') - 1
    bin_indices_r = np.clip(bin_indices_r, 0, len(bin_edges) - 2)
    hist = tf.math.bincount(bin_indices_r[:-dt], minlength=nbins, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r[dt:], minlength=nbins, weights=-delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj
    hist += tf.math.bincount(bin_indices_r, minlength=nbins, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_t2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 = delta_r01*w_traj
    hist += tf.math.bincount(bin_indices_r, minlength=nbins, weights=delta_r01)

    bin_indices = tf.searchsorted(bin_edges, future_boundary.r2, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=-delta_r)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=-delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * tf.gather(w_traj, tf.where(past_boundary.index2>-1, past_boundary.index2, 0))
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r, minlength=nbins, weights=-delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec / 2
    return bin_edges, zc1

def comp_zc1_irreg(r_traj: np.ndarray, b_traj: np.ndarray, future_boundary: bd.FutureBoundary = None,
             past_boundary: bd.PastBoundary = None, i_traj: np.ndarray = None, w_traj : np.ndarray = None,
             dt = 1, nbins = 1000, dtmin=1):
    """
    Compute the Z_C1(r, dt) profile for irregularly sampled trajectory data.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The committor state array.
    future_boundary : bd.FutureBoundary, optional
        Future boundary object. If None, it will be constructed from inputs.
    past_boundary : bd.PastBoundary, optional
        Past boundary object. If None, it will be constructed from inputs.
    i_traj : np.ndarray, optional
        Index trajectory to distinguish between different trajectories.
    w_traj : np.ndarray, optional
        Weight trajectory for reweighting.
    dt : int, optional
        Lag time to compute Z_C1(r, dt). Default is 1.
    nbins : int, optional
        Number of bins for the profile. Default is 1000.
    dtmin : float, optional
        Minimum time step allowed for valid crossings. Default is 1.

    Returns
    -------
    bin_edges : tf.Tensor
        The edges of the histogram bins (length nbins + 1).
    zc1 : tf.Tensor
        The Z_C1 profile values (length nbins).
    """
    r_min = tf.math.reduce_min(r_traj)
    r_max = tf.math.reduce_max(r_traj)
    bin_edges = tf.linspace(r_min, r_max, nbins + 1)
    delta_t_prec = tf.cast(dt, dtype=r_traj.dtype)

    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    delta_t = tf.cast(dt, dtype=r_traj.dtype)
    i_future_boundary_crossed = (tf.cast(future_boundary.index2 > -1, dtype=r_traj.dtype) *
                                 tf.cast(future_boundary.delta_i2 <= delta_t, dtype=r_traj.dtype))
    i_past_boundary_crossed = (tf.cast(past_boundary.index2 > -1, dtype=r_traj.dtype) *
                               tf.cast(past_boundary.delta_i2 >= -delta_t, dtype=r_traj.dtype))
    # compensation for irregular sampling interval
    N=tf.cast(future_boundary.delta_i_to_end+past_boundary.delta_i_from_start+1, dtype=r_traj.dtype)
    N1=tf.where(N>delta_t_prec+0.1, (N-1.)/(N-delta_t_prec+0.0000001), 0)
    N=tf.cast(tf.where(N>dtmin, N1, 0),dtype=r_traj.dtype)

    # no crossings of boundaries
    delta_r = (1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) * (r_traj[dt:] - r_traj[:-dt])
    if i_traj is not None:
        delta_r = delta_r * tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj[:-dt]
    delta_r *= N[:-dt]
    bin_indices_r = tf.searchsorted(bin_edges, r_traj, side='right') - 1
    bin_indices_r = np.clip(bin_indices_r, 0, len(bin_edges) - 2)
    hist = tf.math.bincount(bin_indices_r[:-dt], minlength=nbins, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r[dt:], minlength=nbins, weights=-delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj
    delta_r *= N
    hist += tf.math.bincount(bin_indices_r, minlength=nbins, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_t2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 = delta_r01*w_traj
    delta_r01 *= N
    hist += tf.math.bincount(bin_indices_r, minlength=nbins, weights=delta_r01)

    bin_indices = tf.searchsorted(bin_edges, future_boundary.r2, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=-delta_r)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=-delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * tf.gather(w_traj, tf.where(past_boundary.index2>-1, past_boundary.index2, 0))
    delta_r *= N
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r, minlength=nbins, weights=-delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec / 2
    return bin_edges, zc1


def comp_zq(r_traj: np.ndarray, b_traj: np.ndarray, i_traj: np.ndarray = None,
            future_boundary: bd.FutureBoundary = None, past_boundary: bd.PastBoundary = None, w_traj : np.ndarray = None,
            dt=1, nbins=1000, log_scale=False, log_scale_pmin=1e-4):
    """
    Compute the Z_q(r, dt) profile for a trajectory.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The boundary indicator values.
    i_traj : np.ndarray, optional
        Index trajectory to distinguish between different trajectories.
    future_boundary : bd.FutureBoundary, optional
        Future boundary object. If None, it will be constructed from inputs.
    past_boundary : bd.PastBoundary, optional
        Past boundary object. If None, it will be constructed from inputs.
    w_traj : np.ndarray, optional
        Weight trajectory for reweighting.
    dt : int, optional
        Lag time to compute Z_q(r, dt). Default is 1.
    nbins : int, optional
        Number of bins for the profile. Default is 1000.
    log_scale : bool, optional
        Whether to use logarithmic binning. Default is False.
    log_scale_pmin : float, optional
        Minimum value for log-scale binning. Default is 1e-4.

    Returns
    -------
    bin_edges : tf.Tensor
        The edges of the histogram bins (length nbins + 1).
    zq : tf.Tensor
        The Z_q profile values (length nbins).
    """
    r_min = tf.math.reduce_min(r_traj)
    r_max = tf.math.reduce_max(r_traj)
    bin_edges = tf.linspace(r_min, r_max, nbins + 1)
    if log_scale:
        if log_scale_pmin is None:
            r_min = tf.math.reduce_min(tf.where(r_traj >0, r_traj, r_max))
        else:
            r_min = max(r_min, log_scale_pmin)
        bin_edges = tf.exp(tf.linspace(np.log(r_min), np.log(r_max),
                                       nbins + 1))  # linear in log space
    delta_t_prec = tf.cast(dt, dtype=r_traj.dtype)

    if i_traj is None:
        i_traj = tf.ones_like(r_traj)
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    i_future_boundary_crossed = (tf.cast(future_boundary.index2 > -1, dtype=r_traj.dtype) *
                                 tf.cast(future_boundary.delta_i2 <= delta_t_prec, dtype=r_traj.dtype))
    i_past_boundary_crossed = (tf.cast(past_boundary.index2 > -1, dtype=r_traj.dtype) *
                               tf.cast(past_boundary.delta_i2 >= -delta_t_prec, dtype=r_traj.dtype))

    # no crossings of boundaries
    delta_r = ((1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) * (r_traj[dt:] - r_traj[:-dt])
               * tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype))
    if w_traj is not None:
        delta_r *= w_traj[:-dt]
    bin_indices = tf.searchsorted(bin_edges, r_traj, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist = tf.math.bincount(bin_indices[:-dt], minlength=nbins, weights=delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r *= w_traj
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 *= w_traj
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r *= tf.gather(w_traj, tf.where(past_boundary.index2>-1, past_boundary.index2, 0))
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec
    return bin_edges, zc1

def comp_zt(r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray, i_traj: np.ndarray = None,
            future_boundary: bd.FutureBoundary = None, past_boundary: bd.PastBoundary = None, dt=1, nbins=1000,
            log_scale=False, log_scale_tmin=1e-4):
    """
    Compute the Z_t(r, dt) profile based on MFPT reaction coordinate.

    Parameters
    ----------
    r_traj : np.ndarray
        The reaction coordinate trajectory.
    b_traj : np.ndarray
        The boundary indicator values.
    t_traj : np.ndarray
        The time trajectory used for MFPT computation.
    i_traj : np.ndarray, optional
        Index trajectory to distinguish between different trajectories.
    future_boundary : bd.FutureBoundary, optional
        Future boundary object. If None, it will be constructed from inputs.
    past_boundary : bd.PastBoundary, optional
        Past boundary object. If None, it will be constructed from inputs.
    dt : int, optional
        Lag time to compute Z_t(r, dt). Default is 1.
    nbins : int, optional
        Number of bins for the profile. Default is 1000.
    log_scale : bool, optional
        Whether to use logarithmic binning. Default is False.
    log_scale_tmin : float, optional
        Minimum value for log-scale binning. Default is 1e-4.

    Returns
    -------
    bin_edges : tf.Tensor
        The edges of the histogram bins (length nbins + 1).
    zt : tf.Tensor
        The Z_t profile values (length nbins).
    """
    r_min = tf.math.reduce_min(r_traj)
    r_max = tf.math.reduce_max(r_traj)
    bin_edges = tf.linspace(r_min, r_max, nbins + 1)
    if log_scale:
        if log_scale_tmin is None:
            r_min = tf.math.reduce_min(tf.where(r_traj >0, r_traj, r_max))
        else:
            r_min = max(r_min, log_scale_tmin)
        bin_edges = tf.exp(tf.linspace(np.log(r_min), np.log(r_max),
                                       nbins + 1))  # linear in log space
    delta_t_prec = tf.cast(dt, dtype=r_traj.dtype)

    if i_traj is None:
        i_traj = tf.ones_like(r_traj)
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    i_future_boundary_crossed = (tf.cast(future_boundary.index2 > -1, dtype=r_traj.dtype) *
                                 tf.cast(future_boundary.delta_i2 <= delta_t_prec, dtype=r_traj.dtype))
    i_past_boundary_crossed = (tf.cast(past_boundary.index2 > -1, dtype=r_traj.dtype) *
                               tf.cast(past_boundary.delta_i2 >= -delta_t_prec, dtype=r_traj.dtype))

    # no crossings of boundaries
    delta_r = ((1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) *
                (r_traj[dt:] - r_traj[:-dt] + t_traj[dt:]-t_traj[:-dt]) *
                tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype))
    bin_indices = tf.searchsorted(bin_edges, r_traj, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist = tf.math.bincount(bin_indices[:-dt], minlength=nbins, weights=delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj + future_boundary.delta_t2)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj + future_boundary.delta_t2))
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2-past_boundary.delta_t2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    hist += tf.math.bincount(bin_indices, minlength=nbins, weights=delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec
    return bin_edges, zc1

def comp_zca(r_traj, a, i_traj=None, w_traj=None, t_traj=None, nbins=1000, eps=1e-3, dt=1):
    """
    Compute the Z_{C,a}(r, dt) cut profile with exponent a.

    Parameters
    ----------
    r_traj : tf.Tensor
        Reaction coordinate trajectory.
    a : float
        Exponent for the cut profile.
    i_traj : tf.Tensor, optional
        Index trajectory to distinguish between different trajectories.
    w_traj : tf.Tensor, optional
        Weight trajectory for reweighting.
    t_traj : tf.Tensor, optional
        Time trajectory for non-uniform time steps.
    nbins : int, optional
        Number of bins for the profile. Default is 1000.
    eps : float, optional
        Small value to avoid division by zero or log of zero. Default is 1e-3.
    dt : int, optional
        Lag time to compute Z_{C,a}(r, dt). Default is 1.

    Returns
    -------
    bin_edges : tf.Tensor
        The edges of the histogram bins (length nbins + 1).
    zca : tf.Tensor
        The Z_{C,a} profile values (length nbins).
    """
    r_min = tf.math.reduce_min(r_traj)
    r_max = tf.math.reduce_max(r_traj)
    bin_edges = tf.linspace(r_min, r_max, nbins + 1)

    # Find the bin indices for each data point
    bin_indices = tf.searchsorted(bin_edges, r_traj, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

    delta_r = r_traj[dt:] - r_traj[:-dt]
    if a > 0:
        delta_ra = tf.math.multiply(tf.math.sign(delta_r),
                                    tf.math.pow(tf.math.abs(delta_r), a))
    else:
        delta_ra = tf.math.multiply(
            tf.math.sign(delta_r),
            tf.math.pow(tf.math.maximum(tf.math.abs(delta_r), eps), a))

    if i_traj is not None:
        delta_ra = tf.where(i_traj[dt:] == i_traj[:-dt], delta_ra, 0.)
    if w_traj is not None:
        delta_ra = delta_ra*w_traj[:-dt]
    if t_traj is not None:
        delta_ra=delta_ra*(t_traj[dt:]-t_traj[:-dt])

    delta_ra=tf.cast(delta_ra,tf.float64)
    # Compute the histogram counts
    hist = tf.math.bincount(bin_indices[:-dt], minlength=nbins, weights=delta_ra)
    hist += tf.math.bincount(bin_indices[dt:], minlength=nbins, weights=-delta_ra)
    zca = tf.cumsum(hist)/2/dt
    return bin_edges, zca
