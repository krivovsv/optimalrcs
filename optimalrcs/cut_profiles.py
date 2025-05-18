import numpy as np
import cupy as cp
import boundaries as bd


def comp_zc1_logic(r_traj: np.ndarray, b_traj: np.ndarray, future_boundary: bd.FutureBoundary = None,
                   past_boundary: bd.PastBoundary = None, dt=1, nbins=1000):
    r_min = cp.min(r_traj)
    r_max = cp.max(r_traj)
    bin_edges = cp.linspace(r_min, r_max, nbins + 1)
    zero = cp.astype(0, dtype=r_traj.dtype)

    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj)

    future_boundary_crossed = cp.logical_and(future_boundary.index2 > -1, future_boundary.delta_t2 <= dt)
    past_boundary_crossed = cp.logical_and(past_boundary.index2 > -1, past_boundary.delta_t2 >= -dt)
    delta_r = cp.where(cp.logical_or(future_boundary_crossed[:-dt], past_boundary_crossed[dt:]),
                       zero, r_traj[dt:] - r_traj[:-dt])
    bin_indices_r = cp.searchsorted(bin_edges, r_traj, side='right') - 1
    hist = cp.bincount(bin_indices_r[:-dt], minlength=nbins + 1, weights=delta_r)
    hist += cp.bincount(bin_indices_r[dt:], minlength=nbins + 1, weights=-delta_r)

    delta_r = cp.where(cp.logical_not(future_boundary_crossed), zero, future_boundary.r2 - r_traj)
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r)

    delta_r01 = delta_r * b_traj * (dt - future_boundary.delta_t2 - 1)
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r01)

    bin_indices = cp.searchsorted(bin_edges, future_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r)
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r01)

    delta_r = cp.where(cp.logical_not(past_boundary_crossed), zero, r_traj - past_boundary.r2)
    bin_indices = cp.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=-delta_r)

    zc1 = cp.cumsum(hist) / dt / 2
    return bin_edges, zc1


def comp_zc1(r_traj: np.ndarray, b_traj: np.ndarray, future_boundary: bd.FutureBoundary = None,
             past_boundary: bd.PastBoundary = None, i_traj: np.ndarray = None, w_traj : np.ndarray = None,
             dt = 1, nbins = 1000):
    r_min = cp.min(r_traj)
    r_max = cp.max(r_traj)
    bin_edges = cp.linspace(r_min, r_max, nbins + 1)

    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    i_future_boundary_crossed = ((future_boundary.index2 > -1).astype(r_traj.dtype) *
                                 (future_boundary.delta_i2 <= dt).astype(r_traj.dtype))
    i_past_boundary_crossed = ((past_boundary.index2 > -1).astype(r_traj.dtype) *
                               (past_boundary.delta_i2 >= -dt).astype(r_traj.dtype))

    # no crossings of boundaries
    delta_r = (1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) * (r_traj[dt:] - r_traj[:-dt])
    if i_traj is not None:
        delta_r = delta_r * (i_traj[dt:] == i_traj[:-dt]).astype(r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj[:-dt]
    bin_indices_r = cp.searchsorted(bin_edges, r_traj, side='right') - 1
    hist = cp.bincount(bin_indices_r[:-dt], minlength=nbins + 1, weights=delta_r)
    hist += cp.bincount(bin_indices_r[dt:], minlength=nbins + 1, weights=-delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * (future_boundary.delta_i_to_end >= dt).astype(r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * w_traj
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (dt - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 = delta_r01*w_traj
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r01)

    bin_indices = cp.searchsorted(bin_edges, future_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r)
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * (past_boundary.delta_i_from_start >= dt).astype(r_traj.dtype)
    if w_traj is not None:
        delta_r = delta_r * cp.gather(w_traj, cp.where(past_boundary.index2>-1, past_boundary.index2, 0))
    bin_indices = cp.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=-delta_r)

    zc1 = cp.cumsum(hist) / dt / 2
    return bin_edges, zc1

def comp_zc1_irreg(r_traj: np.ndarray, b_traj: np.ndarray, future_boundary: bd.FutureBoundary = None,
             past_boundary: bd.PastBoundary = None, i_traj: np.ndarray = None, w_traj : np.ndarray = None,
             dt = 1, nbins = 1000, dtmin=1):
    r_min = cp.min(r_traj)
    r_max = cp.max(r_traj)
    bin_edges = cp.linspace(r_min, r_max, nbins + 1)

    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    i_future_boundary_crossed = (cp.cast(future_boundary.index2 > -1, dtype=r_traj.dtype).astype(cp.int32) *
                                 cp.cast(future_boundary.delta_i2 <= dt, dtype=r_traj.dtype).astype(cp.int32))
    i_past_boundary_crossed = (cp.cast(past_boundary.index2 > -1, dtype=r_traj.dtype).astype(cp.int32) *
                               cp.cast(past_boundary.delta_i2 >= -dt, dtype=r_traj.dtype).astype(cp.int32))
    # compensation for irregular sampling interval
    N=cp.astype(future_boundary.delta_i_to_end+past_boundary.delta_i_from_start+1, dtype=r_traj.dtype)
    N1=cp.where(N>dt+0.1, (N-1.)/(N-dt+0.0000001), 0)
    N=cp.where(N>dtmin, N1, 0).astype(r_traj.dtype)
    
    
    # no crossings of boundaries
    delta_r = (1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) * (r_traj[dt:] - r_traj[:-dt])
    if i_traj is not None:
        delta_r = delta_r * cp.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype).astype(cp.int32)
    if w_traj is not None:
        delta_r = delta_r * w_traj[:-dt]
    delta_r *= N[:-dt]
    bin_indices_r = cp.searchsorted(bin_edges, r_traj, side='right') - 1
    hist = cp.bincount(bin_indices_r[:-dt], minlength=nbins + 1, weights=delta_r)
    hist += cp.bincount(bin_indices_r[dt:], minlength=nbins + 1, weights=-delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * cp.cast(future_boundary.delta_i_to_end >= dt, dtype=r_traj.dtype).astype(cp.int32)
    if w_traj is not None:
        delta_r = delta_r * w_traj
    delta_r *= N
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (dt - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 = delta_r01*w_traj
    delta_r01 *= N
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=delta_r01)

    bin_indices = cp.searchsorted(bin_edges, future_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r)
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=-delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * cp.cast(past_boundary.delta_i_from_start >= dt, dtype=r_traj.dtype).astype(cp.int32)
    if w_traj is not None:
        delta_r = delta_r * cp.gather(w_traj, cp.where(past_boundary.index2>-1, past_boundary.index2, 0))
    delta_r *= N
    bin_indices = cp.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    hist += cp.bincount(bin_indices_r, minlength=nbins + 1, weights=-delta_r)

    zc1 = cp.cumsum(hist) / dt / 2
    return bin_edges, zc1


def comp_zq(r_traj: np.ndarray, b_traj: np.ndarray, i_traj: np.ndarray = None,
            future_boundary: bd.FutureBoundary = None, past_boundary: bd.PastBoundary = None, w_traj : np.ndarray = None,
            dt=1, nbins=1000, log_scale=False, log_scale_pmin=1e-4):
    r_min = cp.min(r_traj)
    r_max = cp.max(r_traj)
    bin_edges = cp.linspace(r_min, r_max, nbins + 1)
    if log_scale:
        if log_scale_pmin is None:
            r_min = cp.max(cp.where(r_traj >0, r_traj, r_max))
        else:
            r_min = max(r_min, log_scale_pmin)
        bin_edges = cp.exp(cp.linspace(np.log(r_min), np.log(r_max),
                                       nbins + 1))  # linear in log space

    if i_traj is None:
        i_traj = cp.ones_like(r_traj)
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    i_future_boundary_crossed = ((future_boundary.index2 > -1).astype(r_traj.dtype) *
                                 (future_boundary.delta_i2 <= dt).astype(r_traj.dtype))
    i_past_boundary_crossed = ((past_boundary.index2 > -1).astype(r_traj.dtype) *
                               (past_boundary.delta_i2 >= -dt).astype(r_traj.dtype))

    # no crossings of boundaries
    delta_r = ((1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) * (r_traj[dt:] - r_traj[:-dt])
               * (i_traj[dt:] == i_traj[:-dt]).astype(r_traj.dtype))
    if w_traj is not None:
        delta_r *= w_traj[:-dt]
    bin_indices = cp.searchsorted(bin_edges, r_traj, side='right') - 1
    hist = cp.bincount(bin_indices[:-dt], minlength=nbins + 1, weights=delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj)
    delta_r = delta_r * (future_boundary.delta_i_to_end >= dt).astype(r_traj.dtype)
    if w_traj is not None:
        delta_r *= w_traj
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (dt - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj))
    if w_traj is not None:
        delta_r01 *= w_traj
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2)
    delta_r = delta_r * (past_boundary.delta_i_from_start >= dt).astype(r_traj.dtype)
    if w_traj is not None:
        delta_r *= cp.gather(w_traj, cp.where(past_boundary.index2>-1, past_boundary.index2, 0))
    bin_indices = cp.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    zc1 = cp.cumsum(hist) / dt
    return bin_edges, zc1

def comp_zt(r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray, i_traj: np.ndarray = None,
            future_boundary: bd.FutureBoundary = None, past_boundary: bd.PastBoundary = None, dt=1, nbins=1000, 
            log_scale=False, log_scale_tmin=1e-4):
    r_min = cp.min(r_traj)
    r_max = cp.max(r_traj)
    bin_edges = cp.linspace(r_min, r_max, nbins + 1)
    if log_scale:
        if log_scale_tmin is None:
            r_min = cp.max(cp.where(r_traj >0, r_traj, r_max))
        else:
            r_min = max(r_min, log_scale_tmin)
        bin_edges = cp.exp(cp.linspace(np.log(r_min), np.log(r_max),
                                       nbins + 1))  # linear in log space

    if i_traj is None:
        i_traj = cp.ones_like(r_traj)
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    i_future_boundary_crossed = ((future_boundary.index2 > -1).astype(r_traj.dtype) *
                                 (future_boundary.delta_i2 <= dt).astype(r_traj.dtype))
    i_past_boundary_crossed = ((past_boundary.index2 > -1).astype(r_traj.dtype) *
                               (past_boundary.delta_i2 >= -dt).astype(r_traj.dtype))

    # no crossings of boundaries
    delta_r = ((1 - i_future_boundary_crossed[:-dt]) * (1 - i_past_boundary_crossed[dt:]) * 
                (r_traj[dt:] - r_traj[:-dt] + t_traj[dt:]-t_traj[:-dt]) *
                (i_traj[dt:] == i_traj[:-dt]).astype(r_traj.dtype))
    bin_indices = cp.searchsorted(bin_edges, r_traj, side='right') - 1
    hist = cp.bincount(bin_indices[:-dt], minlength=nbins + 1, weights=delta_r)

    # crossing of the future boundary
    delta_r = i_future_boundary_crossed * (1 - b_traj) * (future_boundary.r2 - r_traj + future_boundary.delta_t2)
    delta_r = delta_r * (future_boundary.delta_i_to_end >= dt).astype(r_traj.dtype)
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    # transitions between the boundaries
    delta_r01 = (i_future_boundary_crossed * b_traj * (dt - future_boundary.delta_i2 + 1) *
                 (future_boundary.r2 - r_traj + future_boundary.delta_t2))
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r01)

    # crossing of the past boundary
    delta_r = i_past_boundary_crossed * (1 - b_traj) * (r_traj - past_boundary.r2-past_boundary.delta_t2)
    delta_r = delta_r * (past_boundary.delta_i_from_start >= dt).astype(r_traj.dtype)
    bin_indices = cp.searchsorted(bin_edges, past_boundary.r2, side='right') - 1
    hist += cp.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    zc1 = cp.cumsum(hist) / dt
    return bin_edges, zc1

def comp_zca(r_traj, a, i_traj=None, w_traj=None, t_traj=None, nbins=1000, eps=1e-3, dt=1):
    """ computes $Z_{C,a}$ cut profile

    :param r_traj: RC timeseries
    :param a: exponent of the cut profile
    :param i_traj: array mapping from the total aggregated trajectory frame to trajectory number
    :param w_traj: re-weighting factor
    :param t_traj: time along trajectories for non-constant delta t
    :param nbins: number of bins in the histogram
    :param eps: lower bound for delta_r in computing delta_r^a, when delta_r<0
    :param dt: delta_t used to compute cut profiles

    returns
    :lx : array of binedges postions
    :ZCa : array of values of ZCa at these postions
    """
    r_min = cp.min(r_traj)
    r_max = cp.max(r_traj)
    bin_edges = cp.linspace(r_min, r_max+0.001, nbins + 1)

    # Find the bin indices for each data point
    bin_indices = cp.searchsorted(bin_edges, r_traj, side='right') - 1

    delta_r = r_traj[dt:] - r_traj[:-dt]
    if a > 0:
        delta_ra = cp.multiply(cp.sign(delta_r),
                                    cp.power(cp.abs(delta_r), a))
    else:
        delta_ra = cp.multiply(
            cp.sign(delta_r),
            cp.power(cp.maximum(cp.abs(delta_r), eps), a))

    if i_traj is not None:
        delta_ra = cp.where(i_traj[dt:] == i_traj[:-dt], delta_ra, 0)
    if w_traj is not None:
        delta_ra = delta_ra*w_traj[:-dt]
    if t_traj is not None:
        delta_ra=delta_ra*(t_traj[dt:]-t_traj[:-dt])

    # Compute the histogram counts
    hist = cp.bincount(bin_indices[:-dt], minlength=nbins + 1, weights=delta_ra)
    hist += cp.bincount(bin_indices[dt:], minlength=nbins + 1, weights=-delta_ra)
    zca = cp.cumsum(hist)/2/dt
    return bin_edges, zca
