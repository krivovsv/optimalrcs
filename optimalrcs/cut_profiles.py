import numpy as np
import tensorflow as tf
from . import boundaries as bd

def comp_zc1(r_traj: np.ndarray, b_traj: np.ndarray, future_boundary: bd.FutureBoundary = None,
             past_boundary: bd.PastBoundary = None, i_traj: np.ndarray = None, w_traj : np.ndarray = None,
             dt = 1, nbins = 1000):
    """
    Compute the Z_C1(r,dt) cut-based profile for a trajectory.

    Parameters:
        r_traj (np.ndarray): The reaction coordinate trajectory.
        b_traj (np.ndarray): The boundary indicator values for the trajectory.
        future_boundary (bd.FutureBoundary, optional): Boundary object for future boundaries. Defaults to None.
        past_boundary (bd.PastBoundary, optional): Boundary object for past boundaries. Defaults to None.
        i_traj (np.ndarray, optional): The index trajectory. Defaults to None.
        w_traj (np.ndarray, optional): The weight trajectory. Defaults to None.
        dt (int, optional): The lag time to compute Z_{C,}(r,dt). Defaults to 1.
        nbins (int, optional): Number of bins for the profile. Defaults to 1000.

    Returns:
        tuple: A tuple containing bin edges and the Z_C1 profile.
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
    hist = tf.math.bincount(bin_indices_r[:-dt], minlength=nbins + 1, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r[dt:], minlength=nbins + 1, weights=-delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj
    hist += tf.math.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_t2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 = delta_r01*w_traj
    hist += tf.math.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r01)

    bin_indices = tf.searchsorted(bin_edges, future_boundary.r2, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r)
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * tf.gather(w_traj, tf.where(past_boundary.index2>-1, past_boundary.index2, 0))
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r, minlength=nbins + 1, weights=-delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec / 2
    return bin_edges, zc1

def comp_zc1_irreg(r_traj: np.ndarray, b_traj: np.ndarray, future_boundary: bd.FutureBoundary = None,
             past_boundary: bd.PastBoundary = None, i_traj: np.ndarray = None, w_traj : np.ndarray = None,
             dt = 1, nbins = 1000, dtmin=1):
    """
    Computes the Z_C1(r,dt) profile for irregularly sampled trajectory data.

    Parameters:
        r_traj (np.ndarray): The reaction coordinate trajectory, array of shape (num_steps,).
        b_traj (np.ndarray): The committor state array of shape (num_steps,).
        future_boundary (bd.FutureBoundary): An instance of FutureBoundary for boundary information.
        past_boundary (bd.PastBoundary): An instance of PastBoundary for boundary information.
        i_traj (np.ndarray): The index trajectory array of shape (num_steps,). If provided, it will be used to enforce consistency in the data handling.
        w_traj (np.ndarray): The weight trajectory array of shape (num_steps,). If provided, it will be used to account for different weights in the calculation.
        dt (int, optional): The lag time to compute Z_{C,1}(r,dt). Defaults to 1.
        nbins (int): Number of bins for the Z_C1 profile.
        dtmin (float): Minimum time step allowed for valid crossings, defaults to 1.

    Returns:
        tuple: A tuple containing two elements:
            - bin_edges (np.ndarray): The edges of the histogram bins.
            - zc1 (np.ndarray): The Z_C1 profile array of shape (nbins,).
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
    hist = tf.math.bincount(bin_indices_r[:-dt], minlength=nbins + 1, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r[dt:], minlength=nbins + 1, weights=-delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj
    delta_r *= N
    hist += tf.math.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_t2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 = delta_r01*w_traj
    delta_r01 *= N
    hist += tf.math.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r01)

    bin_indices = tf.searchsorted(bin_edges, future_boundary.r2, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r)
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * tf.gather(w_traj, tf.where(past_boundary.index2>-1, past_boundary.index2, 0))
    delta_r *= N
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r, minlength=nbins + 1, weights=-delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec / 2
    return bin_edges, zc1


def comp_zq(r_traj: np.ndarray, b_traj: np.ndarray, i_traj: np.ndarray = None,
            future_boundary: bd.FutureBoundary = None, past_boundary: bd.PastBoundary = None, w_traj : np.ndarray = None,
            dt=1, nbins=1000, log_scale=False, log_scale_pmin=1e-4):
    """
    Compute the Z_q profile for a trajectory given by `r_traj` and boundary conditions defined by `b_traj`.

    Parameters:
        r_traj (np.ndarray): The trajectory data as an array of shape (num_points,).
        b_traj (np.ndarray): Boundary condition data as an array of shape (num_points,).
        i_traj (np.ndarray, optional): Indicator function for the trajectory. Defaults to None.
        future_boundary (bd.FutureBoundary, optional): Future boundary object. Defaults to None.
        past_boundary (bd.PastBoundary, optional): Past boundary object. Defaults to None.
        w_traj (np.ndarray, optional): Weight array for trajectory data. Defaults to None.
        dt (int, optional): The lag time to compute Z_q(r,dt). Defaults to 1.
        nbins (int, optional): Number of bins for histogramming. Defaults to 1000.
        log_scale (bool, optional): Whether to use logarithmic scale for bin edges. Defaults to False.
        log_scale_pmin (float, optional): Minimum value for logarithmic scale. Defaults to 1e-4.

    Returns:
        tuple: A tuple containing the bin edges and the Z_q profile array.
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
    hist = tf.math.bincount(bin_indices[:-dt], minlength=nbins + 1, weights=delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r *= w_traj
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 *= w_traj
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    if w_traj is not None:
        delta_r *= tf.gather(w_traj, tf.where(past_boundary.index2>-1, past_boundary.index2, 0))
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec
    return bin_edges, zc1

def comp_zt(r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray, i_traj: np.ndarray = None,
            future_boundary: bd.FutureBoundary = None, past_boundary: bd.PastBoundary = None, dt=1, nbins=1000, 
            log_scale=False, log_scale_tmin=1e-4):
    """
    Compute the Z,t profile from MFPT RC.

    Parameters:
        r_traj (np.ndarray): The main trajectory array representing the state variable.
        b_traj (np.ndarray): The bias trajectory array, indicating whether a boundary is crossed.
        t_traj (np.ndarray): The target trajectory array, used to compute transitions between boundaries.
        i_traj (np.ndarray, optional): The indicator array for the main trajectory. Defaults to ones if not provided.
        future_boundary (bd.FutureBoundary, optional): An instance of FutureBoundary class representing future boundaries.
        past_boundary (bd.PastBoundary, optional): An instance of PastBoundary class representing past boundaries.
        dt (int, optional): The lag time to compute Z_t(r,dt). Defaults to 1.
        nbins (int, optional): The number of bins for histogramming. Defaults to 1000.
        log_scale (bool, optional): Whether to use logarithmic scale for bin edges. Defaults to False.
        log_scale_tmin (float, optional): A minimum value for the logarithmic scale if provided. Defaults to 1e-4.
        
    Returns:
        tuple: A tuple containing two elements:
            - bin_edges (np.ndarray): The edges of the histogram bins.
            - Z_t (np.ndarray): The computed z profile along the trajectory.
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
    hist = tf.math.bincount(bin_indices[:-dt], minlength=nbins + 1, weights=delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj + future_boundary.delta_t2)
    delta_r = delta_r * tf.cast(future_boundary.delta_i_to_end >= delta_t_prec, dtype=r_traj.dtype)
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (delta_t_prec - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj + future_boundary.delta_t2))
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2-past_boundary.delta_t2)
    delta_r = delta_r * tf.cast(past_boundary.delta_i_from_start >= delta_t_prec, dtype=r_traj.dtype)
    bin_indices = tf.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    zc1 = tf.cumsum(hist) / delta_t_prec
    return bin_edges, zc1

def comp_zca(r_traj, a, i_traj=None, w_traj=None, t_traj=None, nbins=1000, eps=1e-3, dt=1):
    """
    Computes the $Z_{C,a}$ cut profile.

    Parameters:
        r_traj (TensorFlow Tensor): RC timeseries data.
        a (int or float): Exponent of the cut profile.
        i_traj (TensorFlow Tensor, optional): Array mapping from total aggregated trajectory frame to trajectory number. Defaults to None.
        w_traj (TensorFlow Tensor, optional): Re-weighting factor. Defaults to None.
        t_traj (TensorFlow Tensor, optional): Time along trajectories for non-constant delta t. Defaults to None.
        nbins (int, optional): Number of bins in the histogram. Defaults to 1000.
        eps (float, optional): Lower bound for delta_r in computing delta_r^a when delta_r < 0. Defaults to 1e-3.
        dt (int, optional): The lag time to compute Z_{C,a}(r,dt). Defaults to 1.
        
    Returns:
        tuple: A tuple containing:
            bin_edges (TensorFlow Tensor): Array of bin edges positions.
            Z_{C,a} (TensorFlow Tensor): Array of values of ZCa at these positions.
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
    hist = tf.math.bincount(bin_indices[:-dt], minlength=nbins + 1, weights=delta_ra)
    hist += tf.math.bincount(bin_indices[dt:], minlength=nbins + 1, weights=-delta_ra)
    zca = tf.cumsum(hist)/2/dt
    return bin_edges, zca
