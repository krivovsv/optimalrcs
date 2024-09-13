import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics

ldt = [2**i for i in range(16)]


def comp_auxiliary_arrays(orc):
    """computes auxiliary arrays for a single trajectory,
    that describe how the trajectory visits boundaries,
    namely:
        index_boundary_future
        index_boundary_past
        r_boundary_future
        r_boundary_past
        delta_i_boundary_future
        delta_i_boundary_past
    
    :param orc: an optimal RC class
    """
    orc.index_boundary_future = np.zeros_like(
        orc.r, 'int32')  #index of the boundary in the future
    orc.index_boundary_past = np.zeros_like(
        orc.r, 'int32')  #index the boundary in the past

    n = len(orc.Ib)
    n1 = len(orc.Ib)-1
    for i in range(n1, -1, -1):
        if i == n1 or orc.i_traj[i] != orc.i_traj[i + 1]:
            index_current_boundary = -1
        if orc.Ib[i] > 0: index_current_boundary = i
        orc.index_boundary_future[i] = index_current_boundary

    for i in range(n):
        if i == 0 or orc.i_traj[i] != orc.i_traj[i - 1]:
            index_current_boundary = -1
        if orc.Ib[i] > 0: index_current_boundary = i
        orc.index_boundary_past[i] = index_current_boundary

    orc.r_boundary_future = np.where(orc.index_boundary_future > -1,
                                     orc.r[orc.index_boundary_future], 0)
    orc.r_boundary_past = np.where(orc.index_boundary_past > -1,
                                   orc.r[orc.index_boundary_past], 0)

    index_frame = range(n)  # frame index along the trajectory
    orc.delta_i_boundary_future = np.where(
        orc.index_boundary_future > -1,
        orc.index_boundary_future - index_frame, 1e10)
    orc.delta_i_boundary_past = np.where(orc.index_boundary_past > -1,
                                         index_frame - orc.index_boundary_past,
                                         1e10)


def comp_auxiliary_arrays2(orc):
    """computes auxiliary arrays for a single trajectory,
    that describe how the trajectory visits boundaries,
    namely:
        index_boundary_future
        index_boundary_past
        r_boundary_future
        r_boundary_past
        delta_i_boundary_future
        delta_i_boundary_past
    
    difference with the comp_auxiliary_arrays function is that 
    for states at the boundary, the index points not to the 
    current state, but to the next boundary state, which is 
    useful in calculating cut profiles.
    
    :param orc: an optimal RC class
    """
    orc.index_boundary_future2 = np.zeros_like(
        orc.r, 'int32')  #index of the boundary in the future
    orc.index_boundary_past2 = np.zeros_like(
        orc.r, 'int32')  #index the boundary in the past

    n = len(orc.Ib)
    n1 = len(orc.Ib)-1
    for i in range(n - 1, -1, -1):
        if i == n1 or orc.i_traj[i] != orc.i_traj[i + 1]:
            index_current_boundary = -1
        orc.index_boundary_future2[i] = index_current_boundary
        if orc.Ib[i] > 0: index_current_boundary = i

    for i in range(n):
        if i == 0 or orc.i_traj[i] != orc.i_traj[i - 1]:
            index_current_boundary = -1
        orc.index_boundary_past2[i] = index_current_boundary
        if orc.Ib[i] > 0: index_current_boundary = i

    orc.r_boundary_future2 = np.where(orc.index_boundary_future2 > -1,
                                      orc.r[orc.index_boundary_future2], 0)
    orc.r_boundary_past2 = np.where(orc.index_boundary_past2 > -1,
                                    orc.r[orc.index_boundary_past2], 0)

    index_frame = range(n)  # frame index along the trajectory
    orc.delta_i_boundary_future2 = np.where(
        orc.index_boundary_future2 > -1,
        orc.index_boundary_future2 - index_frame, 1e10)
    orc.delta_i_boundary_past2 = np.where(
        orc.index_boundary_past2 > -1, index_frame - orc.index_boundary_past2,
        1e10)


def comp_auxiliary_arrays_ne_vardt(orc):
    """computes auxiliary arrays for multiple trajectories
    that describe how the trajectories visit boundaries,
    namely:
        index_boundary_future
        index_boundary_past
        r_boundary_future
        r_boundary_past
        delta_i_boundary_future
        delta_i_boundary_past
        delta_t_boundary_future
        delta_t_boundary_past
    
    :param orc: an optimal RC class
    """
    orc.index_boundary_future = np.zeros_like(
        orc.r, 'int32')  #index of the boundary in the future
    orc.index_boundary_past = np.zeros_like(
        orc.r, 'int32')  #index the boundary in the past

    n1 = len(orc.Ib) - 1
    n = len(orc.Ib)
    for i in range(n1, -1, -1):
        if i == n1 or orc.i_traj[i] != orc.i_traj[i + 1]:
            index_current_boundary = -1
        if orc.Ib[i] > 0: index_current_boundary = i
        orc.index_boundary_future[i] = index_current_boundary

    for i in range(n):
        if i == 0 or orc.i_traj[i] != orc.i_traj[i - 1]:
            index_current_boundary = -1
        if orc.Ib[i] > 0: index_current_boundary = i
        orc.index_boundary_past[i] = index_current_boundary

    orc.r_boundary_future = np.where(orc.index_boundary_future > -1,
                                     orc.r[orc.index_boundary_future], 0)
    orc.r_boundary_past = np.where(orc.index_boundary_past > -1,
                                   orc.r[orc.index_boundary_past], 0)

    index_frame = range(n)  # frame index along the trajectory
    orc.delta_i_boundary_future = np.where(
        orc.index_boundary_future > -1,
        orc.index_boundary_future - index_frame, 1e10)
    orc.delta_i_boundary_past = np.where(orc.index_boundary_past > -1,
                                         index_frame - orc.index_boundary_past,
                                         1e10)

    orc.delta_t_boundary_future = np.where(
        orc.index_boundary_future > -1,
        orc.t_traj[orc.index_boundary_future] - orc.t_traj[index_frame], 1e10)
    orc.delta_t_boundary_past = np.where(
        orc.index_boundary_past > -1,
        orc.t_traj[index_frame] - orc.t_traj[orc.index_boundary_past], 1e10)


def comp_auxiliary_arrays_ne_vardt2(orc):
    """computes auxiliary arrays for multiple trajectories
    that describe how the trajectories visit boundaries,
    namely:
        index_boundary_future
        index_boundary_past
        r_boundary_future
        r_boundary_past
        delta_i_boundary_future
        delta_i_boundary_past
        delta_t_boundary_future
        delta_t_boundary_past
    
    difference with the comp_auxiliary_arrays_ne_vardt function is that 
    for states at the boundary, the index points not to the 
    current state, but to the next boundary state, which is 
    useful in calculating cut profiles.
    
    :param orc: an optimal RC class
    """
    orc.index_boundary_future2 = np.zeros_like(
        orc.r, 'int32')  #index of the boundary in the future
    orc.index_boundary_past2 = np.zeros_like(
        orc.r, 'int32')  #index the boundary in the past

    n1 = len(orc.Ib) - 1
    n = len(orc.Ib)
    for i in range(n1, -1, -1):
        if i == n1 or orc.i_traj[i] != orc.i_traj[i + 1]:
            index_current_boundary = -1
        orc.index_boundary_future2[i] = index_current_boundary
        if orc.Ib[i] > 0: index_current_boundary = i

    for i in range(n):
        if i == 0 or orc.i_traj[i] != orc.i_traj[i - 1]:
            index_current_boundary = -1
        orc.index_boundary_past2[i] = index_current_boundary
        if orc.Ib[i] > 0: index_current_boundary = i

    orc.r_boundary_future2 = np.where(orc.index_boundary_future2 > -1,
                                      orc.r[orc.index_boundary_future2], 0)
    orc.r_boundary_past2 = np.where(orc.index_boundary_past2 > -1,
                                    orc.r[orc.index_boundary_past2], 0)

    index_frame = range(n)  # frame index along the trajectory
    orc.delta_i_boundary_future2 = np.where(
        orc.index_boundary_future2 > -1,
        orc.index_boundary_future2 - index_frame, 1e10)
    orc.delta_i_boundary_past2 = np.where(
        orc.index_boundary_past2 > -1, index_frame - orc.index_boundary_past2,
        1e10)

    orc.delta_t_boundary_future2 = np.where(
        orc.index_boundary_future2 > -1,
        orc.t_traj[orc.index_boundary_future2] - orc.t_traj[index_frame], 1e10)
    orc.delta_t_boundary_past2 = np.where(
        orc.index_boundary_past2 > -1,
        orc.t_traj[index_frame] - orc.t_traj[orc.index_boundary_past2], 1e10)


def comp_ZCa(r, a, nbins=1000, eps=1e-3):
    """ computes $Z_{C,a}$ cut profile for a single trajectory
    
    :param r: RC timeseries
    :param a: exponent of the cut profile
    :param nbins: number of bins in the histogram
    :param eps: lower bound for delta_r in computing delta_r^a, when delta_r<0
    
    returns 
    :lx : array of binedges postions
    :ZCa : array of values of ZCa at these postions
    """
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)

    # Find the bin indices for each data point
    bin_indices = tf.searchsorted(bin_edges, r, side='right') - 1

    delta_r = r[1:] - r[:-1]
    if a > 0:
        delta_ra = tf.math.multiply(tf.math.sign(delta_r),
                                    tf.math.pow(tf.math.abs(delta_r), a))
    else:
        delta_ra = tf.math.multiply(
            tf.math.sign(delta_r),
            tf.math.pow(tf.math.maximum(tf.math.abs(delta_r), eps), a))
    delta_ra = delta_ra / 2

    # Compute the histogram counts
    hist = tf.math.bincount(bin_indices[:-1],
                            minlength=nbins + 1,
                            weights=delta_ra)
    hist += tf.math.bincount(bin_indices[1:],
                             minlength=nbins + 1,
                             weights=-delta_ra)
    ZCa = tf.cumsum(hist)
    return bin_edges, ZCa

def comp_ZCa_w(r, a, i_traj, w, nbins=1000, eps=1e-3):
    """ computes $Z_{C,a}$ cut profile for multiple trjectories
    
    :param r: RC timeseries
    :param a: exponent of the cut profile
    :param i_traj: array mapping from the total aggregated trajectory frame 
        to trajectory number
    :param nbins: number of bins in the histogram
    :param eps: lower bound for delta_r in computing delta_r^a, when delta_r<0
    
    returns 
    :lx : array of binedges postions
    :ZCa : array of values of ZCa at these postions
    """
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)

    # Find the bin indices for each data point
    bin_indices = tf.searchsorted(bin_edges, r, side='right') - 1

    delta_r = r[1:] - r[:-1]
    if a > 0:
        delta_ra = tf.math.multiply(tf.math.sign(delta_r),
                                    tf.math.pow(tf.math.abs(delta_r), a))
    else:
        delta_ra = tf.math.multiply(
            tf.math.sign(delta_r),
            tf.math.pow(tf.math.maximum(tf.math.abs(delta_r), eps), a))
    It = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r.dtype)
    delta_ra = delta_ra * w[:-1] * It

    # Compute the histogram counts
    hist = tf.math.bincount(bin_indices[:-1],
                            minlength=nbins + 1,
                            weights=delta_ra)
    hist += tf.math.bincount(bin_indices[1:],
                             minlength=nbins + 1,
                             weights=-delta_ra)
    ZCa = tf.cumsum(hist) / 2
    return bin_edges, ZCa


def comp_ZCa_w_vardt(r, a, i_traj, t_traj, w, nbins=1000, eps=1e-3):
    """ computes $Z_{C,a}$ cut profile for multiple trjectories,
    with variable saving time interval
    
    :param r: RC timeseries
    :param a: exponent of the cut profile
    :param i_traj: array mapping from the total aggregated trajectory frame 
        to trajectory number
    :param t_traj: array of timestamps along the trajectory 
    :param nbins: number of bins in the histogram
    :param eps: lower bound for delta_r in computing delta_r^a, when delta_r<0
    
    returns 
    :lx : array of binedges postions
    :ZCa : array of values of ZCa at these positions
    """
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)

    # Find the bin indices for each data point
    bin_indices = tf.searchsorted(bin_edges, r, side='right') - 1

    delta_r = r[1:] - r[:-1]
    if a > 0:
        delta_ra = tf.math.multiply(tf.math.sign(delta_r),
                                    tf.math.pow(tf.math.abs(delta_r), a))
    else:
        delta_ra = tf.math.multiply(
            tf.math.sign(delta_r),
            tf.math.pow(tf.math.maximum(tf.math.abs(delta_r), eps), a))
    It = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r.dtype)

    delta_t = t_traj[1:] - t_traj[:-1]
    delta_ra = delta_ra * w[:-1] * delta_t * It

    # Compute the histogram counts
    hist = tf.math.bincount(bin_indices[:-1],
                            minlength=nbins + 1,
                            weights=delta_ra)
    hist += tf.math.bincount(bin_indices[1:],
                             minlength=nbins + 1,
                             weights=-delta_ra)
    ZCa = tf.cumsum(hist) / 2
    return bin_edges, ZCa

#@tf.function
def comp_ZC1(r, Ib, r_boundary_future, r_boundary_past, delta_t_boundary_future, delta_t_boundary_past, dt=1, nbins=1000):
    """ computes $Z_{C,1}(r,dt)$ cut profile for a single trjectory,
    using TP summation scheme for dt>1. It is used as a committor
    optimiality validation criterion.
    
    :param r: RC timeseries
    :param Ib: array indicator of boundaries
    :param r_boundary_future, r_boundary_past, delta_t_boundary_future,
        delta_t_boundary_past: auxiliary arrays
    :param dt: time interval 
    :param nbins: number of bins in the histogram
    
    returns 
    :lx : array of binedges postions
    :ZC1 : array of values of ZC1 at these postions
    """
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)
    delta_t_prec = tf.cast(dt, dtype=r.dtype)
    zero = tf.cast(0, dtype=r.dtype)

    delta_r = tf.where(tf.math.logical_or(delta_t_boundary_future[:-dt] < delta_t_prec, delta_t_boundary_past[dt:] < delta_t_prec),
                       zero, r[dt:] - r[:-dt])
    bin_indices_r = tf.searchsorted(bin_edges, r, side='right') - 1
    hist = tf.math.bincount(bin_indices_r[:-dt],
                            minlength=nbins + 1,
                            weights=delta_r)
    hist += tf.math.bincount(bin_indices_r[dt:],
                             minlength=nbins + 1,
                             weights=-delta_r)

    delta_r = tf.where(delta_t_boundary_future >= delta_t_prec, zero, r_boundary_future - r)
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=delta_r)
    delta_r01 = delta_r * Ib * (delta_t_prec - delta_t_boundary_future - 1)
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=delta_r01)
    bin_indices = tf.searchsorted(bin_edges, r_boundary_future, side='right') - 1
    hist += tf.math.bincount(bin_indices,
                             minlength=nbins + 1,
                             weights=-delta_r)
    hist += tf.math.bincount(bin_indices,
                             minlength=nbins + 1,
                             weights=-delta_r01)

    delta_r = tf.where(delta_t_boundary_past >= delta_t_prec, zero, r - r_boundary_past)
    bin_indices = tf.searchsorted(bin_edges, r_boundary_past, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=-delta_r)

    ZC1 = tf.cumsum(hist) / delta_t_prec / 2
    return bin_edges, ZC1

#@tf.function
def comp_ZC1_w(r,
               Ib,
               r_boundary_future,
               r_boundary_past,
               delta_t_boundary_future,
               delta_t_boundary_past,
               index_boundary_past,
               i_traj,
               wtraj,
               dt=1,
               nbins=1000):
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)
    delta_t_prec = tf.cast(dt, dtype=r.dtype)
    zero = tf.cast(0, dtype=r.dtype)

    delta_r = tf.where(tf.math.logical_or(delta_t_boundary_future[:-dt] < delta_t_prec, delta_t_boundary_past[dt:] < delta_t_prec),
                       zero, r[dt:] - r[:-dt])

    Itw = tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r.dtype) * wtraj[:-dt]
    delta_r = delta_r * Itw
    bin_indices_r = tf.searchsorted(bin_edges, r, side='right') - 1
    hist = tf.math.bincount(bin_indices_r[:-dt],
                            minlength=nbins + 1,
                            weights=delta_r)
    hist += tf.math.bincount(bin_indices_r[dt:],
                             minlength=nbins + 1,
                             weights=-delta_r)

    delta_r = tf.where(delta_t_boundary_future >= delta_t_prec, zero, r_boundary_future - r)
    delta_r = delta_r * wtraj
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=delta_r)
    delta_r01 = delta_r * Ib * (delta_t_prec - delta_t_boundary_future - 1)
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=delta_r01)
    bin_indices = tf.searchsorted(bin_edges, r_boundary_future, side='right') - 1
    hist += tf.math.bincount(bin_indices,
                             minlength=nbins + 1,
                             weights=-delta_r)
    hist += tf.math.bincount(bin_indices,
                             minlength=nbins + 1,
                             weights=-delta_r01)

    delta_r = tf.where(delta_t_boundary_past >= delta_t_prec, zero, r - r_boundary_past)
    delta_r = delta_r * tf.gather(wtraj, index_boundary_past)  # w[r_boundary_past]
    bin_indices = tf.searchsorted(bin_edges, r_boundary_past, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=-delta_r)

    ZC1 = tf.cumsum(hist) / delta_t_prec / 2
    return bin_edges, ZC1



def comp_Zmfpt(r, Ib, r_boundary_future, r_boundary_past, delta_t_boundary_future, delta_t_boundary_past, dt=1, nbins=1000):
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)
    delta_t_prec = tf.cast(dt, dtype=r.dtype)
    zero = tf.cast(0, dtype=r.dtype)

    delta_r = tf.where(tf.math.logical_or(delta_t_boundary_future[:-dt] < delta_t_prec, delta_t_boundary_past[dt:] < delta_t_prec),
                       zero, r[dt:] - r[:-dt] + delta_t_prec)
    bin_indices = tf.searchsorted(bin_edges, r, side='right') - 1
    hist = tf.math.bincount(bin_indices[:-dt],
                            minlength=nbins + 1,
                            weights=delta_r)

    delta_r = tf.where(delta_t_boundary_future >= delta_t_prec, zero, r_boundary_future - r + delta_t_boundary_future)
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    delta_r00 = delta_r * Ib * (
        delta_t_prec - delta_t_boundary_future - 1
    )  # -1 because one transition is already counted in the delta_t_boundary_future above
    hist += tf.math.bincount(bin_indices,
                             minlength=nbins + 1,
                             weights=delta_r00)

    delta_r = tf.where(delta_t_boundary_past >= delta_t_prec, zero, r - r_boundary_past + delta_t_boundary_past)
    bin_indices = tf.searchsorted(bin_edges, r_boundary_past, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    Zmfpt = tf.cumsum(hist) / delta_t_prec
    return bin_edges, Zmfpt

#@tf.function
def comp_Zmfpt_ne(r,
                  Ib,
                  r_boundary_future,
                  r_boundary_past,
                  delta_t_boundary_future,
                  delta_t_boundary_past,
                  i_traj,
                  t_traj,
                  dt=1,
                  nbins=1000):
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)
    delta_t_prec = tf.cast(dt, dtype=r.dtype)
    zero = tf.cast(0, dtype=r.dtype)

    #$\sum_i (\tau(i)-\tau(k)+\Delta t)P(i|k)=0$

    delta_r = tf.where(tf.math.logical_or(delta_t_boundary_future[:-dt] < delta_t_prec, delta_t_boundary_past[dt:] < delta_t_prec),
                       zero, r[dt:] - r[:-dt] + t_traj[dt:] - t_traj[:-dt])
    delta_r = tf.where(i_traj[dt:] == i_traj[:-dt], delta_r, zero)
    bin_indices = tf.searchsorted(bin_edges, r, side='right') - 1
    hist = tf.math.bincount(bin_indices[:-dt],
                            minlength=nbins + 1,
                            weights=delta_r)

    delta_r = tf.where(delta_t_boundary_future >= delta_t_prec, zero, r_boundary_future - r + delta_t_boundary_future)
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)
    delta_r00 = delta_r * Ib * (
        delta_t_prec - delta_t_boundary_future - 1
    )  # -1 because one transition is already counted in the delta_t_boundary_future above
    hist += tf.math.bincount(bin_indices,
                             minlength=nbins + 1,
                             weights=delta_r00)

    delta_r = tf.where(delta_t_boundary_past >= delta_t_prec, zero, r - r_boundary_past + delta_t_boundary_past)
    bin_indices = tf.searchsorted(bin_edges, r_boundary_past, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    Zmfpt = tf.cumsum(hist) / delta_t_prec
    return bin_edges, Zmfpt



#@tf.function
def comp_Zq_ne(r, Ib, r_boundary_future, r_boundary_past, delta_t_boundary_future, delta_t_boundary_past, i_traj, dt=1, nbins=1000):
    rmin = tf.math.reduce_min(r)
    rmax = tf.math.reduce_max(r)
    bin_edges = tf.linspace(rmin, rmax, nbins + 1)
    delta_t_prec = tf.cast(dt, dtype=r.dtype)
    zero = tf.cast(0, dtype=r.dtype)

    delta_r = tf.where(tf.math.logical_or(delta_t_boundary_future[:-dt] < delta_t_prec, delta_t_boundary_past[dt:] < delta_t_prec),
                       zero, r[dt:] - r[:-dt])
    delta_r = tf.where(i_traj[dt:] == i_traj[:-dt], delta_r, zero)
    bin_indices_r = tf.searchsorted(bin_edges, r, side='right') - 1
    hist = tf.math.bincount(bin_indices_r[:-dt],
                            minlength=nbins + 1,
                            weights=delta_r)

    delta_r = tf.where(delta_t_boundary_future >= delta_t_prec, zero, r_boundary_future - r)
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=delta_r)
    delta_r01 = delta_r * Ib * (delta_t_prec - delta_t_boundary_future - 1)
    hist += tf.math.bincount(bin_indices_r,
                             minlength=nbins + 1,
                             weights=delta_r01)

    delta_r = tf.where(delta_t_boundary_past >= delta_t_prec, zero, r - r_boundary_past)
    bin_indices = tf.searchsorted(bin_edges, r_boundary_past, side='right') - 1
    hist += tf.math.bincount(bin_indices, minlength=nbins + 1, weights=delta_r)

    ZC1 = tf.cumsum(hist) / delta_t_prec
    return bin_edges, ZC1

def comp_r2s(r, nbins=10000):
    """computes transformation from r to s - the natural
    coordinate, where diffusion coefficient is D(s)=1
    
    returns cubic splines, that approximate functions 
    r->s and r->delta_r/ds
    
    r - the putative RC timeseries
    nbins - number of bins used to compute histograms, controls discretization
    """
    from scipy.interpolate import UnivariateSpline

    lx, lzh = comp_ZCa(r, a=-1, nbins=nbins)
    lzh = lzh * 2

    lx1, lzc1 = comp_ZCa(r, a=1, nbins=nbins)
    ld = np.sqrt((lzc1 + 1) / (lzh + 1))

    r2delta_rds = UnivariateSpline(lx, ld, s=0)

    ld1 = 1 / ld
    r2s = UnivariateSpline(lx, ld1, s=0).antiderivative()

    return r2s, r2delta_rds

def comp_r2s_w(r, i_traj, w, nbins=10000):
    """computes transformation from r to s - the natural
    coordinate, where diffusion coefficient is D(s)=1
    
    returns cubic splines, that approximate functions 
    r->s and r->delta_r/ds
    
    r - the putative RC timeseries
    dx - discretization step
    """
    from scipy.interpolate import UnivariateSpline

    lx, lzh = comp_ZCa_w(r, a=-1, i_traj=i_traj, w=w, nbins=nbins)
    lzh = lzh * 2

    lx1, lzc1 = comp_ZCa_w(r, a=1, i_traj=i_traj, w=w, nbins=nbins)
    ld = np.sqrt((lzc1 + 1) / (lzh + 1))

    r2delta_rds = UnivariateSpline(lx, ld, s=0)

    ld1 = 1 / ld
    r2s = UnivariateSpline(lx, ld1, s=0).antiderivative()

    return r2s, r2delta_rds


def comp_r2s_w_dt(r, i_traj, t_traj, w, nbins=10000):
    """computes transformation from r to s - the natural
    coordinate, where diffusion coefficient is D(s)=1
    
    returns cubic splines, that approximate functions 
    r->s and r->delta_r/ds
    
    r - the putative RC timeseries
    dx - discretization step
    """
    from scipy.interpolate import UnivariateSpline

    lx, lzh = comp_ZCa_w_vardt(r,
                            a=-1,
                            i_traj=i_traj,
                            t_traj=t_traj,
                            w=w,
                            nbins=nbins)
    lzh = lzh * 2

    lx1, lzc1 = comp_ZCa_w(r, a=1, i_traj=i_traj, w=w, nbins=nbins)
    ld = np.sqrt((lzc1 + 1) / (lzh + 1))

    r2delta_rds = UnivariateSpline(lx, ld, s=0)

    ld1 = 1 / ld
    r2s = UnivariateSpline(lx, ld1, s=0).antiderivative()

    return r2s, r2delta_rds


@tf.function
def basis_poly_ry(r, y, n, fenv=None):
    """computes basis functions as terms of polynomial of variables r and y

    r is the putative RC time-series
    y is a randomly chosen collective variable or coordinate to improve r
    n is the degree of the polynomial
    fenv is common envelope to focus optimization on a particular region
    """
    r = r / tf.math.reduce_max(tf.math.abs(r))
    y = y / tf.math.reduce_max(tf.math.abs(y))

    if fenv is None:
        f = tf.ones_like(r)
    else:
        f = tf.identity(fenv)

    fk = []
    for iy in range(n + 1):
        fr = tf.identity(f)
        for ir in range(n + 1 - iy):
            fk.append(fr)
            fr = fr * r
        f = f * y
    return tf.stack(fk)


@tf.function
def basis_sigmoidpoly_ry(r, y, n, fenv=None, rescale=0):
    """computes basis functions to emulate optimization of q=f(r)
    where f is fixed and equals sigmoid: f(r)=1/(1+exp(-r))
    basis functions are terms of polynomial P(r,y) of variables r
    and y of degree n with common envelope.
        
    r is the putative RC time-series
    y is a randomly chosen collective variable or coordinate to improve r
    n is the degree of the polynomial
    rescale > 0 is used to rescale coordinate other than the committor
        to [0,1] range. Maximum and minimum are determined by subsampling
        the rc, to remove boundary points that are too distant from the rest
        For example, maximum is defined as minimum of all the maxima
        of rc[ioff:-1:rescale], for ioff in range(0,rescale)
    """
    rr = r
    if rescale > 0:
        k = int(len(r) / rescale)
        rr = rr - tf.math.reduce_max(
            tf.math.reduce_min(tf.reshape(rr[:k * rescale], [rescale, k]), 1))
        rr = rr / tf.math.reduce_min(
            tf.math.reduce_max(tf.reshape(rr[:k * rescale], [rescale, k]), 1))
    rr = tf.clip_by_value(rr, 1e-10, 1 - 1e-10)
    dfdr = rr * (1 - rr)
    rr = -tf.math.log(1 / rr - 1)
    rr = rr / tf.math.reduce_max(tf.math.abs(rr))
    y = y / tf.math.reduce_max(tf.math.abs(y))

    fk = []
    f = dfdr
    if fenv is not None: f = f * fenv
    for ir in range(n + 1):
        fy = tf.identity(f)
        for iy in range(n + 1 - ir):
            fk.append(fy)
            fy = fy * y
        f = f * rr
    return tf.stack(fk)




@tf.function
def committor_loss_dt(r, Ib, delta_t_boundary_future, delta_t_boundary_past, r_boundary_future, r_boundary_past, dt=1):
    # Calculate committor loss for any dt
    # neglects the transitions between the boundary states, which
    # dont contribute to the gradient
    Ip = tf.cast(delta_t_boundary_future > dt, dtype=r.dtype)
    Im = tf.cast(delta_t_boundary_past > dt, dtype=r.dtype)
    loss = tf.reduce_sum(Im[dt:] * Ip[:-dt] *
                         tf.square(r[dt:] - r[:-dt]))
    loss += tf.reduce_sum((1 - Ip) * (1 - Ib) * tf.square(r - r_boundary_future))
    loss += tf.reduce_sum((1 - Im) * (1 - Ib) * tf.square(r - r_boundary_past))
    return loss / (len(r) - dt) / dt


@tf.function
def committor_loss(r, It):
    return tf.reduce_sum((r[1:] - r[:-1])**2 * It) / len(It)


@tf.function
def NPq(r, fk, Itw=None):
    """ implements NPq (non-parametric committor optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if Itw is None: Itw = tf.ones_like(r[:-1])

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = tf.tensordot(dfk * Itw, dfk, axes=[1, 1])

    delta_r = r[1:] - r[:-1]
    b = tf.tensordot(dfk, -delta_r * Itw, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn


@tf.function
def NPlmbd(r, fk, IA, IB, lmbdA, lmbdB, Itw=None):
    """ implements NPlmbd (non-parametric committor optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if Itw is None: Itw = tf.ones_like(r[:-1])

    dfk = fk[:, 1:] - fk[:, :-1]

    akj = tf.tensordot(dfk * Itw, dfk, axes=[1, 1])
    akj = akj + tf.tensordot(fk * (lmbdA * IA + lmbdB * IB), fk, axes=[1, 1])

    dr = -(r[1:] - r[:-1])

    b = tf.tensordot(dfk, dr * Itw, 1)
    b = b + tf.tensordot(fk, (lmbdB * IB * (1 - r) - r * lmbdA * IA), 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn


@tf.function
def NPlmbd_b(r, fk, IA, IB, lmbdA, lmbdB, Itw=None, batch_size=100000):
    """ implements NPlmbd (non-parametric committor optimization) iteration
        using batches for large datasets.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    num_samples = fk.shape[1]
    num_batches = (num_samples - 1) // batch_size + 1

    if Itw is None: Itw = tf.ones_like(r[:-1])

    _ = (lmbdA * IA + lmbdB * IB)
    _2 = (lmbdB * IB * (1 - r) - r * lmbdA * IA)
    delta_r = -(r[1:] - r[:-1]) * Itw

    akj = tf.zeros((fk.shape[0], fk.shape[0]), dtype=fk.dtype)
    b = tf.zeros((fk.shape[0]), dtype=fk.dtype)
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_samples - 1)
        end_idx2 = min((batch + 1) * batch_size, num_samples)
        dfk = fk[:, start_idx + 1:end_idx + 1] - fk[:, start_idx:end_idx]
        akj = akj + tf.tensordot(
            dfk * Itw[start_idx:end_idx], dfk, axes=[1, 1])
        akj = akj + tf.tensordot(
            fk[:, start_idx:end_idx2] * _[start_idx:end_idx2],
            fk[:, start_idx:end_idx2],
            axes=[1, 1])
        b = b + tf.tensordot(dfk, delta_r[start_idx:end_idx], 1)
        b = b + tf.tensordot(fk[:, start_idx:end_idx2], _2[start_idx:end_idx2],
                             1)

    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn


@tf.function
def NPmfpt(r, fk, Itw=None):
    """ implements NPmfpt (non-parametric mfpt optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if Itw is None: Itw = tf.ones_like(r[:-1])

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = tf.tensordot(dfk * Itw, dfk, axes=[1, 1])

    delta_r = -(r[1:] - r[:-1])
    b = tf.tensordot(dfk, delta_r * Itw, 1) + 2 * tf.math.reduce_sum(fk, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    return rn


@tf.function
def Imfpt(r, dt):
    return tf.math.reduce_mean(
        tf.square(r[dt:] - r[:-dt]) - tf.cast(2 * dt, dtype=r.dtype) *
        (r[:-dt] + r[dt:]))


def Imfpt_min(delta_t_boundary_future, Ib):
    # array of delta_t from the current boundary to the next, removing the last delta t
    dti = delta_t_boundary_future[1:][Ib[:-1] > 0][:-1] + 1
    return sum(dti**2) / len(Ib)

envelope_scale=0.01

def comp_env(r):
    i0 = tf.random.uniform(shape=[], maxval=r.shape[0], dtype=tf.int32)
    r0 = r[i0]
    delta_r = tf.math.reduce_max(tf.math.abs(r)) - tf.math.reduce_min(
        tf.math.abs(r))
    if delta_r < 1e-5: delta_r = 1e-5
    if tf.random.uniform([], maxval=1) < 0.5:
        return tf.math.sigmoid((r - r0) / envelope_scale / delta_r)
    return tf.math.sigmoid(-(r - r0) / envelope_scale / delta_r)

def comp_env(r):
    i0 = np.random.randint(r.shape[0])
    r0 = r[i0]
    delta_r = tf.math.reduce_max(tf.math.abs(r)) - tf.math.reduce_min(
        tf.math.abs(r))
    if delta_r < 1e-5: delta_r = 1e-5
    if np.random.random() < 0.5:
        return tf.math.sigmoid((r - r0) / envelope_scale / delta_r)
    return tf.math.sigmoid(-(r - r0) / envelope_scale / delta_r)

class Committor:

    def __init__(self, IndA, IndB, i_traj=None, prec=np.float64):
        self.IndA = IndA
        self.IndB = IndB
        self.Ib = np.zeros_like(IndA, prec)
        self.Ib[self.IndA | self.IndB] = 1
        if i_traj is None:
            self.i_traj = np.ones_like(IndA, prec)
        else:
            self.i_traj = i_traj
        self.It = np.ones_like(IndA[:-1], prec)
        self.It[self.i_traj[1:] != self.i_traj[:-1]] = 0
        self.len = len(IndA)
        self.r = np.ones_like(IndA, prec) / 2
        self.r[self.IndA] = 0
        self.r[self.IndB] = 1
        self.prec = prec
        comp_auxiliary_arrays(self)
        comp_auxiliary_arrays2(self)

    def min_loss(self):
        return sum((self.r_boundary_future[1:]-self.r_boundary_future[:-1])**2*self.It) / 2

    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      delta_r2min=None,
                      dxmin=None,
                      iprint=1000,
                      basis_functions=basis_poly_ry):

        ro = self.r

        self.metrics={'mse':[],'delta_r2':[],'dx':[],'iteration':[]}
        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            
            if i % 10 == 0: 
                fenv = comp_env(self.r) * (1 - self.Ib)
            
            fk = basis_functions(self.r, y, ny, fenv)  # compute basis functions
            self.r = NPq(self.r, fk, self.It)  # find the optimal update of the RC

            if i % iprint == 0:
                delta_r=tf.where(self.index_boundary_future > -1,self.r_boundary_future - self.r,0)
                mse=tf.math.reduce_mean(delta_r**2)
                delta_r=tf.where(self.index_boundary_past > -1,self.r_boundary_past - self.r,0)
                mse+=tf.math.reduce_mean(delta_r**2)
                mse=mse.numpy()/2
                
                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2
                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()

                print('iteration %i, mse=%g, loss=%g, |dx|=%g, time=%g' %
                      (i, mse, delta_r2, dx, time.time() - start))
                
                self.metrics['iteration'].append(i)
                self.metrics['mse'].append(mse)
                self.metrics['delta_r2'].append(delta_r2)
                self.metrics['dx'].append(dx)

                if delta_r2min is not None and delta_r2 < delta_r2min and i > 0:
                    break
                    
                if dxmin is not None and dx < dxmin and i > 0:
                    break

    def plots(self, ldt=ldt, dt_sim=1, suptitle=None):
        y_pred = self.r.numpy()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)

        #plot for ZH
        lx2, lzh2 = comp_ZCa(y_pred, a=-1)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$q$')
        ax1.grid()

        #plot for ZC1
        for dt in ldt:
            lx, ly = comp_ZC1(y_pred,
                              self.Ib,
                              self.r_boundary_future2,
                              self.r_boundary_past2,
                              self.delta_i_boundary_future2,
                              self.delta_i_boundary_past2,
                              dt=tf.constant(dt))
            ax3.plot(lx.numpy()[:-1], -np.log(ly.numpy()[:-1]))
        ax3.set(ylabel='$-\ln Z_{C,1}$', xlabel='$q$')
        ax3.grid()
        
        # plot for ZH along the natural coordinate
        r2si, r2delta_rdsi = comp_r2s(y_pred)
        s = r2si(y_pred) * dt_sim**0.5
        lx2, lzh2 = comp_ZCa(s, a=-1)
        ax2.plot(lx2[:-2], -np.log(2 * lzh2[:-2] + 1), 'r-')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{q}$')
        ax2.grid()
        fig.tight_layout()

    def plots_optimization(self,suptitle=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)

        n=len(self.metrics['iteration'])//2
        ax1.plot(self.metrics['iteration'],self.metrics['mse'],':b')
        ax1t=ax1.twinx()
        ax1t.plot(self.metrics['iteration'][n:],self.metrics['mse'][n:],'-r')
        ax1t.grid()
        ax1.set(xlabel='iteration',ylabel='MSE',yscale='log')

        ax2.plot(self.metrics['iteration'],self.metrics['delta_r2'],':b')
        ax2t=ax2.twinx()
        ax2t.plot(self.metrics['iteration'][n:],self.metrics['delta_r2'][n:],'-r')
        ax2t.grid()
        ax2.set(xlabel='iteration',ylabel='$\Delta r^2$',yscale='log')
        
        ax3.plot(self.metrics['iteration'],self.metrics['dx'],':b')
        ax3.set(xlabel='iteration',ylabel='$max <|\Delta Z_q|>$',yscale='log')
        ax3t=ax3.twinx()
        ax3t.plot(self.metrics['iteration'][n:],self.metrics['dx'][n:],'-r')
        ax3t.grid()
        fig.tight_layout()

    def plot_obs_pred(self, nbins=100, suptitle=None, halves=True):
        y_pred = self.r.numpy()
        y_test = self.r_boundary_future
        rmin = tf.math.reduce_min(y_pred)
        rmax = tf.math.reduce_max(y_pred)
        bin_edges = tf.linspace(rmin, rmax, nbins + 1)
        zero = tf.cast(0, self.prec)
        one = tf.cast(1, self.prec)

        nB = tf.where(self.index_boundary_future > -1, self.r_boundary_future,
                      zero)
        ni = tf.where(self.index_boundary_future > -1, one, zero)
        di = tf.where(self.index_boundary_future > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, y_pred, side='right') - 1
        hist_nB = tf.math.bincount(bin_indices,
                                   minlength=nbins + 1,
                                   weights=nB)
        hist_n = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ni)
        hist_d = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=di)
        hist_pB = hist_nB / hist_n
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        if halves:
            n2=len(y_pred)//2
            hist_nB1 = tf.math.bincount(bin_indices[:n2],
                                       minlength=nbins + 1,
                                       weights=nB[:n2])
            hist_n1 = tf.math.bincount(bin_indices[:n2], minlength=nbins + 1, weights=ni[:n2])
            hist_pB1 = hist_nB1 / hist_n1

            hist_nB2 = tf.math.bincount(bin_indices[n2:],
                                       minlength=nbins + 1,
                                       weights=nB[n2:])
            hist_n2 = tf.math.bincount(bin_indices[n2:], minlength=nbins + 1, weights=ni[n2:])
            hist_pB2 = hist_nB2 / hist_n2


        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)
        ax1.plot(lx, hist_pB[:-1], '-r', label='obs vs pred')
        if halves:
            ax1.plot(lx, hist_pB1[:-1], ':g', label='obs vs pred 1/2')
            ax1.plot(lx, hist_pB2[:-1], ':b', label='obs vs pred 2/2')
        ax1.plot((lx[0], lx[-1]), (lx[0], lx[-1]), ':k', label='obs = pred')
        ax1.set(xlabel='pB predicted', ylabel='pB observed')
        ax1.legend()
        ax1.grid()

        ax2.plot(lx, hist_n[:-1], '-r', label='hist observed')
        ax2.plot(lx, hist_d[:-1], '-k', label='hist discarded')
        ax2.set(xlabel='pB predicted', ylabel='N', yscale='log')
        ax2.legend()

        fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred)
        ax3.plot(fpr[1:], thresh[1:], 'r-', label='threshold')
        AUC = metrics.roc_auc_score(y_test, y_pred)
        ax3.plot(fpr, tpr, 'b-', label='tpr, AUC: %.2f%%' % (AUC * 100))
        ax3.set(xscale='log',
                xlabel='False positive rate',
                ylabel='True positive rate')
        ax3.legend()
        ax3.grid()

        fig.tight_layout()

    def plot_obs_pred_log(self, p_min=1e-4, nbins=100, suptitle=None):
        y_pred = self.r.numpy()
        y_pred_n0 = y_pred[y_pred > 0]
        rmin = tf.math.reduce_min(y_pred_n0)
        rmin = max(rmin, p_min)
        rmax = tf.math.reduce_max(y_pred)
        bin_edges = tf.exp(tf.linspace(np.log(rmin), np.log(rmax),
                                       nbins + 1))  # linear in log space
        zero = tf.cast(0, self.prec)
        one = tf.cast(1, self.prec)

        nB = tf.where(self.index_boundary_future > -1, self.r_boundary_future,
                      zero)
        ni = tf.where(self.index_boundary_future > -1, one, zero)
        di = tf.where(self.index_boundary_future > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, y_pred, side='right') - 1
        hist_nB = tf.math.bincount(bin_indices,
                                   minlength=nbins + 1,
                                   weights=nB)
        hist_n = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ni)
        hist_d = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=di)
        hist_pB = hist_nB / hist_n
        lx = (bin_edges[:-1] * bin_edges[1:])**0.5

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)
        ax1.plot(lx, hist_pB[:-1], '-r', label='obs vs pred')
        ax1.plot((lx[0], lx[-1]), (lx[0], lx[-1]), ':k', label='obs = pred')
        ax1.set(xlabel='pB predicted',
                ylabel='pB observed',
                xscale='log',
                yscale='log')
        ax1.legend()
        ax1.grid()

        ax2.plot(lx, hist_n[:-1], '-r', label='hist observed')
        ax2.plot(lx, hist_d[:-1], '-k', label='hist discarded')
        ax2.set(xlabel='pB predicted', ylabel='N', yscale='log', xscale='log')
        ax2.legend()

        dx = np.array(bin_edges[1:] - bin_edges[:-1])
        p = (hist_n.numpy()[:-1]) / dx
        ax3.plot(lx, p, 'r-', label='P')
        ax3.set(xscale='log',
                xlabel='pB predicted',
                ylabel='Probability density of patients')
        ax3.legend()
        ax3.grid()

        fig.tight_layout()

    def plot_time_CDF(self,
                      lr=np.linspace(0, 1, 11, True),
                      xlim=None,
                      ylim=None,
                      nbins=10000,
                      suptitle=None):
        y_pred = self.r.numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)
        for i in range(len(lr) - 1):
            r_in_range = (lr[i] < y_pred) & (y_pred < lr[i + 1]) & (
                self.r_boundary_future == 1)
            dt = self.delta_i_boundary_future[r_in_range]
            dt_max = tf.math.reduce_max(dt).numpy()
            bin_edges = tf.linspace(0, dt_max,
                                    nbins + 1)  # linear in log space

            bin_indices = tf.searchsorted(bin_edges, dt, side='right') - 1
            hist_dt = tf.math.bincount(bin_indices, minlength=nbins + 1)
            c_hist_dt = tf.cumsum(hist_dt)
            c_hist_dt /= c_hist_dt[-1]

            lx = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax1.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))
            ax2.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))

        ax1.legend(loc='lower right')
        ax1.set(xlabel='t', ylabel='CDF', yscale='log', ylim=(0.01, 0.1))
        ax1.grid(which='both')
        if xlim is not None: ax1.set(xlim=xlim)
        if ylim is not None: ax1.set(ylim=ylim)
        ax2.legend(loc='lower right')
        ax2.set(xlabel='t',
                ylabel='CDF',
                xscale='log',
                yscale='log',
                ylim=(0.01, 1))
        ax2.grid(which='both')

        fig.tight_layout()


class Soft_Committor:

    def __init__(self,
                 IndA,
                 IndB,
                 lmbdA,
                 lmbdB,
                 i_traj=None,
                 t_traj=None,
                 prec=np.float64):
        self.IndA = tf.cast(IndA, prec)
        self.IndB = tf.cast(IndB, prec)
        self.lmbdA = lmbdA
        self.lmbdB = lmbdB
        self.t_traj = t_traj
        if i_traj is None:
            self.i_traj = np.ones_like(IndA, prec)
        else:
            self.i_traj = i_traj
        self.It = np.ones_like(IndA[:-1], prec)
        self.It[self.i_traj[1:] != self.i_traj[:-1]] = 0
        self.len = len(IndA)
        self.r = np.ones_like(IndA, prec) / 2
        self.prec = prec

    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      delta_r2min=None,
                      dxmin=None,
                      iprint=1000,
                      batch_size=None):

        ro = self.r

        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r)
            fk = basis_poly_ry(self.r, y, ny, fenv)  # compute basis functions
            if batch_size == None:
                self.r = NPlmbd(
                    self.r,
                    fk,
                    self.IndA,
                    self.IndB,
                    self.lmbdA,
                    self.lmbdB,
                    Itw=self.It)  # find the optimal update of the RC
            else:
                self.r = NPlmbd_b(
                    self.r,
                    fk,
                    self.IndA,
                    self.IndB,
                    self.lmbdA,
                    self.lmbdB,
                    Itw=self.It,
                    batch_size=batch_size)  # find the optimal update of the RC

            if i % iprint == 0:
                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                delta_r = tf.math.reduce_max(tf.math.abs(
                    self.r)) - tf.math.reduce_min(tf.math.abs(self.r))
                delta_r = delta_r.numpy()
                print(
                    'iteration %i, delta_r2=%g, delta_r=%g, |dx|=%g, time=%g' %
                    (i, delta_r2 / delta_r / delta_r, delta_r, dx / delta_r,
                     time.time() - start))
                if delta_r2min != None and delta_r2 / delta_r / delta_r < delta_r2min and i > 0:
                    break
                if dxmin != None and dx / delta_r < dxmin and i > 0: break

    def plots(self, dt_sim=1):
        y_pred = self.r.numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        lx2, lzh2 = comp_ZCa(y_pred, a=-1)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$q$')
        ax1.grid()

        r2si, r2delta_rdsi = comp_r2s(y_pred)
        s = r2si(y_pred) * dt_sim**0.5
        self.rn=s
        lx2, lzh2 = comp_ZCa(s, a=-1)
        ax2.plot(lx2[:-2],
                 -np.log(2 * lzh2[:-2] + 1),
                 'r-',
                 label='$F_{C,-1}$')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{q}$')
        ax2.grid()
        fig.tight_layout()

    def comp_eq_weights(self, ny=6, miter=1000, dxmin=None, iprint=10):
        self.w = np.ones_like(self.r, self.prec)

        start = time.time()
        wo = self.w
        for i in range(miter):
            self.w = NPNEw(self.w, basis_poly_ry(self.w, self.r, ny), self.It)

            if i % iprint == 0:
                dx = self.w - wo
                wo = self.w
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                maxw = tf.math.reduce_max(self.w).numpy()
                minw = tf.math.reduce_min(self.w).numpy()
                print('iteration %i, max(w)=%g, min(w)=%g, |dx|=%g, time=%g' %
                      (i, maxw, minw, dx, time.time() - start))
                if dxmin != None and dx < dxmin: break

    def eq_plots(self, dt_sim=1):
        y_pred = self.r.numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        lx2, lzh2 = comp_ZCa_w(y_pred, a=-1, i_traj=self.i_traj, w=self.w)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$q$')
        ax1.grid()

        r2si, r2delta_rdsi = comp_r2s_w_dt(y_pred,
                                           i_traj=self.i_traj,
                                           t_traj=self.t_traj,
                                           w=self.w)
        s = r2si(y_pred) * dt_sim**0.5
        lx2, lzh2 = comp_ZCa_w(s, a=-1, i_traj=self.i_traj, w=self.w)
        ax2.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{q}$')
        ax2.grid()
        fig.tight_layout()


class MFPT:

    def __init__(self, IndA, i_traj=None, prec=np.float64):
        self.Ib = np.array(IndA, prec)
        if i_traj == None:
            self.i_traj = np.ones_like(IndA, prec)
        else:
            self.i_traj = i_traj
        self.It = np.ones_like(IndA[:-1], prec)
        self.It[self.i_traj[1:] != self.i_traj[:-1]] = 0
        self.len = len(IndA)
        self.r = np.ones_like(IndA, prec)
        self.r[IndA] = 0
        self.prec = prec
        comp_auxiliary_arrays(self)
        comp_auxiliary_arrays2(self)

    def min_loss(self):
        return -Imfpt_min(self.delta_i_boundary_future, self.Ib)

    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      Imin=None,
                      dxmin=None,
                      iprint=1000):

        ro = self.r

        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)
            fk = basis_poly_ry(self.r, y, ny, fenv)  # compute basis functions
            self.r = NPmfpt(self.r, fk,
                            self.It)  # find the optimal update of the RC
            self.r = tf.where(self.r < 0, tf.cast(0, self.prec), self.r)

            if i % iprint == 0:
                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                delta_r = tf.math.reduce_max(tf.math.abs(
                    self.r)) - tf.math.reduce_min(tf.math.abs(self.r))
                I = Imfpt(self.r, 1).numpy()
                print(
                    'iteration %i, loss=%g, delta_r=%g, |dx|=%g, time=%g' %
                    (i, I, delta_r.numpy(), dx / delta_r, time.time() - start))
                if Imin != None and I < Imin and i > 0: break
                if dxmin != None and dx / delta_r < dxmin and i > 0: break

    def plots(self, ldt=ldt, dt_sim=1):
        y_pred = self.r.numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        lx2, lzh2 = comp_ZCa(y_pred, a=-1)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$mfpt$')
        ax1.grid()

        for dt in ldt:
            lx, ly = comp_Zmfpt(y_pred,
                                self.Ib,
                                self.r_boundary_future2,
                                self.r_boundary_past2,
                                self.delta_i_boundary_future2,
                                self.delta_i_boundary_past2,
                                dt=tf.constant(dt))
            ax3.plot(lx.numpy()[:-1], -np.log(ly.numpy()[:-1]))
        ax3.set(ylabel='$-\ln Z_{mfpt}$', xlabel='$mfpt$')
        ax3.grid()

        r2si, r2delta_rdsi = comp_r2s(y_pred, nbins=100000)
        s = r2si(y_pred) * dt_sim**0.5
        lx2, lzh2 = comp_ZCa(s, a=-1)
        ax2.plot(lx2[:-2], -np.log(2 * lzh2[:-2] + 1), 'r-')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{mfpt}$')
        ax2.grid()
        fig.tight_layout()

    def plot_obs_pred(self, nbins=100):
        y_pred = self.r.numpy()
        rmin = tf.math.reduce_min(y_pred)
        rmax = tf.math.reduce_max(y_pred)
        bin_edges = tf.linspace(rmin, rmax, nbins + 1)
        zero = tf.cast(0, self.prec)
        one = tf.cast(1, self.prec)

        ti = tf.where(self.index_boundary_future > -1,
                      self.delta_i_boundary_future, zero)
        ni = tf.where(self.index_boundary_future > -1, one, zero)
        di = tf.where(self.index_boundary_future > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, y_pred, side='right') - 1
        hist_t = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ti)
        hist_n = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ni)
        hist_d = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=di)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        hist_t = hist_t / hist_n
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax1.plot(lx, hist_t[:-1], '-r', label='obs vs pred')
        ax1.plot((lx[0], lx[-1]), (lx[0], lx[-1]), ':k', label='obs = pred')
        ax1.set(xlabel='mfpt predicted', ylabel='mfpt observed')
        ax1.legend()

        ax2.plot(lx, hist_n[:-1], '-r', label='hist observed')
        ax2.plot(lx, hist_d[:-1], '-k', label='hist discarded')
        ax2.set(xlabel='mfpt predicted', ylabel='N', yscale='log')
        ax2.legend()
        fig.tight_layout()

    def plot_time_CDF(self,
                      lr=np.linspace(0, 1, 11, True),
                      xlim=None,
                      ylim=None,
                      nbins=10000):
        y_pred = self.r.numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(len(lr) - 1):
            r_in_range = (lr[i] < y_pred) & (y_pred < lr[i + 1]) & (
                self.index_boundary_future > -1)
            dt = self.delta_i_boundary_future[r_in_range]
            dt_max = tf.math.reduce_max(dt).numpy()
            bin_edges = tf.linspace(0, dt_max,
                                    nbins + 1)  # linear in log space

            bin_indices = tf.searchsorted(bin_edges, dt, side='right') - 1
            hist_dt = tf.math.bincount(bin_indices, minlength=nbins + 1)
            c_hist_dt = tf.cumsum(hist_dt)
            c_hist_dt /= c_hist_dt[-1]

            lx = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax1.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))
            ax2.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))

        ax1.legend(loc='lower right')
        ax1.set(xlabel='t', ylabel='CDF', yscale='log', ylim=(0.01, 1))
        ax1.grid(which='both')
        if xlim is not None: ax1.set(xlim=xlim)
        if ylim is not None: ax1.set(ylim=ylim)
        ax2.legend(loc='lower right')
        ax2.set(xlabel='t',
                ylabel='CDF',
                xscale='log',
                yscale='log',
                ylim=(0.01, 1))
        ax2.grid(which='both')

        fig.tight_layout()


@tf.function
def NPNEq(r, fk, It, gamma=0, stable=False):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    if stable:
        akj = -tf.tensordot(fk[:, :-1], fk[:, :-1] * It, axes=[1, 1])
    else:
        delta_fj = fk[:, 1:] - fk[:, :-1]*(1+gamma)
        akj = tf.tensordot(fk[:, :-1], delta_fj * It, axes=[1, 1])

    delta_r = r[1:] - r[:-1]
    b = tf.tensordot(fk[:, :-1], -delta_r * It, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn

@tf.function
def NPNEq_dt(r, fk, i_traj, t_traj, delta_t_boundary_future, r_boundary_future, gamma=0, dt=1):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    It = tf.cast(i_traj[dt:] == i_traj[:-dt], dtype = r.dtype)
    delta_t_prec = tf.cast(dt, dtype=r.dtype)
    Ip = tf.cast(delta_t_boundary_future[:-dt] > delta_t_prec, dtype = r.dtype)
 
    delta_fj = fk[:, dt:]*Ip - fk[:, :-dt]*(1+gamma)
    akj = tf.tensordot(fk[:, :-dt], delta_fj * It, axes=[1, 1])
 
    rp = tf.where(delta_t_boundary_future[:-dt] > delta_t_prec,
                       r[dt:], r_boundary_future[:-dt])
    delta_r = rp - r[:-dt]
    b = tf.tensordot(fk[:, :-dt], -delta_r * It, 1)
    b = tf.reshape(b, [b.shape[0], 1])
 
    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])
 
    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn

@tf.function
def NPNEq_dt1(r, fk, i_traj, t_traj, delta_t_boundary_future, r_boundary_future, gamma=0, dt=1):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    It = tf.cast(i_traj[dt:] == i_traj[:-dt], dtype = r.dtype)
    delta_t_prec = tf.cast(dt, dtype=r.dtype)
    Ip = tf.cast(delta_t_boundary_future[:-dt] > delta_t_prec, dtype = r.dtype)
 
    delta_fj = fk[:, dt:]*Ip - fk[:, :-dt]*(1+gamma)
    akj = tf.tensordot(fk[:, :-dt], delta_fj * It, axes=[1, 1])
 
    rp = tf.where(delta_t_boundary_future[:-dt] > delta_t_prec,
                       r[dt:], r_boundary_future[:-dt])
    delta_r = rp - r[:-dt]
    b = tf.tensordot(fk[:, :-dt], -delta_r * It, 1)
    
    #akj=akj/dt
    #b=b/dt

    It = tf.cast(i_traj[1:] == i_traj[:-1], dtype = r.dtype)
    delta_fj = fk[:, 1:] - fk[:, :-1]*(1+gamma)
    akj = akj + tf.tensordot(fk[:, :-1], delta_fj * It, axes=[1, 1])

    delta_r = r[1:] - r[:-1]
    b = b + tf.tensordot(fk[:, :-1], -delta_r * It, 1)

    
    b = tf.reshape(b, [b.shape[0], 1])
 
    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])
 
    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn


@tf.function
def NPNEw(r, fk, It):
    """ implements NPNEw (non-parametric non-equilibrium re-weighting factors 
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """

    dfk = fk[:, 1:] - fk[:, :-1]

    b = -tf.tensordot(dfk * It, r[:-1], 1)
    b = tf.reshape(b, [b.shape[0], 1])
    scale = tf.math.reduce_sum(1 - r[:-1] * It)
    scale = tf.reshape(scale, [1, 1])
    b = tf.concat((b, scale), 0)

    ones = tf.reshape(It, [1, It.shape[0]])
    dfk = tf.concat((dfk * It, ones), 0)
    akj = tf.tensordot(dfk, fk[:, :-1], axes=[1, 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = abs(rn)

    return rn


gamma=0.5
def comp_gamma(i,miter): 
    return gamma

def comp_maxdZq(self):
    y_pred=self.r.numpy()
    max_m2=0,0
    max_abs=0,0
    for dt in ldt:
        lx, ly = comp_Zq_ne(y_pred,
            self.Ib,
            self.r_boundary_future2,
            self.r_boundary_past2,
            self.delta_i_boundary_future2,
            self.delta_i_boundary_past2,
            self.i_traj,
            dt=tf.constant(dt))

        ly=ly[:-1]
        m1=np.mean(ly)
        mabs=np.mean(abs(ly/m1-1))
        if mabs>max_abs[0]:max_abs=mabs,dt
        m2=np.mean((ly/m1-1)**2)**0.5
        if m2>max_m2[0]:max_m2=m2,dt
    return max_abs[0],max_abs[1],max_m2[0],max_m2[1]


class CommittorNE:

    def __init__(self, IndA, IndB, i_traj, t_traj, prec=np.float64):
        self.IndA = IndA
        self.IndB = IndB
        self.Ib = np.zeros_like(IndA, prec)
        self.Ib[self.IndA | self.IndB] = 1
        self.i_traj = i_traj
        self.t_traj = np.array(t_traj, prec)
        self.It = np.array(i_traj[1:] == i_traj[:-1], prec)
        self.len = len(IndA)
        self.r = np.ones_like(IndA, prec) / 2
        self.r[self.IndA] = 0
        self.r[self.IndB] = 1
        self.prec = prec
        comp_auxiliary_arrays_ne_vardt(self)
        comp_auxiliary_arrays_ne_vardt2(self)

    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      delta_r2min=None,
                      iprint=1000,
                      comp_gamma=comp_gamma,
                      history=None,
                      metrics=None,
                      stable=False,
                      diff_type=0,
                      basis_functions=basis_poly_ry):

        ro = self.r
        self.metrics={'mse':[],'delta_r2':[],'dx':[],'iteration':[],'m1':[],'m2':[]}
        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)

            if history is None:
                fk = basis_functions(self.r, y, ny,
                                 fenv)  # compute basis functions
            else:
                d=np.random.choice(history)
                if d>0 and diff_type>0:
                    if diff_type==1: fk = basis_functions(tf.roll(self.r,d,0)-self.r, y-tf.roll(y,d,0), ny,
                                 fenv)  # compute basis functions
                    if diff_type==2: fk = basis_functions(tf.roll(self.r,d,0), y-tf.roll(y,d,0), ny,
                                 fenv)  # compute basis functions
                    if diff_type==3: fk = basis_functions(self.r, y-tf.roll(y,d,0), ny,
                                 fenv)  # compute basis functions
                    if diff_type==4: fk = basis_functions(tf.roll(self.r,d,0), tf.roll(y,d,0), ny,
                                 fenv)  # compute basis functions
                    if diff_type==5: fk = basis_functions(tf.roll(self.r,d,0)-self.r, tf.roll(y,d,0), ny,
                                 fenv)  # compute basis functions
                else: 
                    fk = basis_functions(self.r, y, ny,
                                 fenv)


            gamma=tf.constant(comp_gamma(i,miter),dtype=self.prec)
            self.r = NPNEq(self.r, fk,
                           self.It, gamma, stable)  # find the optimal update of the RC

            if i % iprint == 0:
                delta_r=tf.where(self.index_boundary_future > -1,self.r_boundary_future - self.r,0)
                mse=tf.math.reduce_mean(delta_r**2).numpy()

                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                m1,dt1,m2,dt2=comp_maxdZq(self)
                print('i=%i, mse=%g, delta_r2=%g, |dx|=%g, stdev=%g, time=%g' %
                      (i, mse, delta_r2, dx, m2, time.time() - start))
                if metrics is not None: metrics(self)
                self.metrics['iteration'].append(i)
                self.metrics['mse'].append(mse)
                self.metrics['delta_r2'].append(delta_r2)
                self.metrics['dx'].append(dx)
                self.metrics['m1'].append(m1)
                self.metrics['m2'].append(m2)
                if dxmin != None and dx < dxmin and i > 0: break
                if delta_r2min != None and delta_r2 < delta_r2min and i > 0: break

    def fit_transform2(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      delta_r2min=None,
                      iprint=1000,
                      comp_gamma=comp_gamma,
                      history=None,
                      metrics=None,
                      stable=False,
                      diff_type=[(2,2)],
                      dt=1,
                      basis_functions=basis_poly_ry):

        ro = self.r
        self.metrics={'mse':[],'delta_r2':[],'dx':[],'iteration':[],'m1':[],'m2':[]}
        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)

            if history is None:
                fk = basis_functions(self.r, y, ny,
                                 fenv)  # compute basis functions
            else:
                d=np.random.choice(history)
                if d>0:
                    if len(diff_type)>1:d1,d2=diff_type[np.random.randint(len(diff_type))]
                    else: d1,d2=diff_type[0]
                    if d1==1: dr=self.r
                    if d1==2: dr=tf.roll(self.r,d,0)
                    if d1==3: dr=self.r-tf.roll(self.r,d,0)
                    if d1==4: dr=tf.roll(y,d,0)
                    if d2==1: dy=y
                    if d2==2: dy=tf.roll(y,d,0)
                    if d2==3: dy=y-tf.roll(y,d,0)
                    if d2==4: dy=tf.roll(self.r,d,0)
                    fk = basis_functions(dr, dy, ny, fenv)
                else:
                    fk = basis_functions(self.r, y, ny, fenv)


            gamma=tf.constant(comp_gamma(i,miter),dtype=self.prec)
            if dt==1:
                self.r = NPNEq(self.r, fk,
                           self.It, gamma, stable)  # find the optimal update of the RC
            else:
                self.r = NPNEq_dt(self.r, fk,
                           self.i_traj, self.t_traj,
                           self.delta_t_boundary_future,
                           self.r_boundary_future, gamma, dt=dt)  # find the optimal update of the RC

            if i % iprint == 0:
                delta_r=tf.where(self.index_boundary_future > -1,self.r_boundary_future - self.r,0)
                mse=tf.math.reduce_mean(delta_r**2).numpy()

                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                m1,dt1,m2,dt2=comp_maxdZq(self)
                print('i=%i, mse=%g, delta_r2=%g, |dx|=%g, stdev=%g, time=%g' %
                      (i, mse, delta_r2, dx, m2, time.time() - start))
                if metrics is not None: metrics(self)
                self.metrics['iteration'].append(i)
                self.metrics['mse'].append(mse)
                self.metrics['delta_r2'].append(delta_r2)
                self.metrics['dx'].append(dx)
                self.metrics['m1'].append(m1)
                self.metrics['m2'].append(m2)
                if dxmin != None and dx < dxmin and i > 0: break
                if delta_r2min != None and delta_r2 < delta_r2min and i > 0: break
    def fit_transform3(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      delta_r2min=None,
                      iprint=1000,
                      comp_gamma=comp_gamma,
                      history=None,
                      metrics=None,
                      stable=False,
                      diff_type=[(2,2)],
                      dt=1,
                      basis_functions=basis_poly_ry):

        ro = self.r
        self.metrics={'mse':[],'delta_r2':[],'dx':[],'iteration':[],'m1':[],'m2':[]}
        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)

            if history is None:
                fk = basis_functions(self.r, y, ny,
                                 fenv)  # compute basis functions
            else:
                d=np.random.choice(history)
                if d>0:
                    if len(diff_type)>1:d1,d2=diff_type[np.random.randint(len(diff_type))]
                    else: d1,d2=diff_type[0]
                    if d1==1: y1=self.r
                    if d1==2: y1=y
                    if d1==3: y1=tf.roll(self.r,d,0)
                    if d1==4: y1=tf.roll(y,d,0)
                    if d2==1: y2=self.r
                    if d2==2: y2=y
                    if d2==3: y2=tf.roll(self.r,d,0)
                    if d2==4: y2=tf.roll(y,d,0)
                    fk = basis_functions(y1, y2, ny, fenv)
                else:
                    fk = basis_functions(self.r, y, ny, fenv)


            gamma=tf.constant(comp_gamma(i,miter),dtype=self.prec)
            if dt==1:
                self.r = NPNEq(self.r, fk,
                           self.It, gamma, stable)  # find the optimal update of the RC
            else:
                self.r = NPNEq_dt(self.r, fk,
                           self.i_traj, self.t_traj,
                           self.delta_t_boundary_future,
                           self.r_boundary_future, gamma, dt=dt)  # find the optimal update of the RC

            if i % iprint == 0:
                delta_r=tf.where(self.index_boundary_future > -1,self.r_boundary_future - self.r,0)
                mse=tf.math.reduce_mean(delta_r**2).numpy()

                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                m1,dt1,m2,dt2=comp_maxdZq(self)
                print('i=%i, mse=%g, delta_r2=%g, |dx|=%g, stdev=%g, time=%g' %
                      (i, mse, delta_r2, dx, m2, time.time() - start))
                if metrics is not None: metrics(self)
                self.metrics['iteration'].append(i)
                self.metrics['mse'].append(mse)
                self.metrics['delta_r2'].append(delta_r2)
                self.metrics['dx'].append(dx)
                self.metrics['m1'].append(m1)
                self.metrics['m2'].append(m2)
                if dxmin != None and dx < dxmin and i > 0: break
                if delta_r2min != None and delta_r2 < delta_r2min and i > 0: break
                    
    def plots(self, ldt=ldt, dt_sim=1, suptitle=None):
        y_pred = self.r.numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)

        lx2, lzh2 = comp_ZCa(y_pred, a=-1)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$q$')
        ax1.grid()

        for dt in ldt:
            lx, ly = comp_Zq_ne(y_pred,
                                self.Ib,
                                self.r_boundary_future2,
                                self.r_boundary_past2,
                                self.delta_i_boundary_future2,
                                self.delta_i_boundary_past2,
                                self.i_traj,
                                dt=tf.constant(dt))
            ax3.plot(lx.numpy()[:-1], ly.numpy()[:-1])
        ax3.set(ylabel='$Z_q$', xlabel='$q$')
        ax3.grid()

        r2si, r2delta_rdsi = comp_r2s(y_pred)
        s = r2si(y_pred) * dt_sim**0.5
        lx2, lzh2 = comp_ZCa(s, a=-1)
        ax2.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{q}$')
        ax2.grid()
        fig.tight_layout()
    
    def plots_optimization(self,suptitle=None):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)

        n=len(self.metrics['iteration'])//2
        ax1.plot(self.metrics['iteration'],self.metrics['mse'],':b')
        ax1t=ax1.twinx()
        ax1t.plot(self.metrics['iteration'][n:],self.metrics['mse'][n:],'-r')
        ax1t.grid()
        ax1.set(xlabel='iteration',ylabel='MSE',yscale='log')

        ax2.plot(self.metrics['iteration'],self.metrics['delta_r2'],':b')
        ax2t=ax2.twinx()
        ax2t.plot(self.metrics['iteration'][n:],self.metrics['delta_r2'][n:],'-r')
        ax2t.grid()
        ax2.set(xlabel='iteration',ylabel='$\Delta r^2$',yscale='log')
        
        ax3.plot(self.metrics['iteration'],self.metrics['dx'],':b')
        ax3.set(xlabel='iteration',ylabel='$\Delta x$',yscale='log')
        ax3t=ax3.twinx()
        ax3t.plot(self.metrics['iteration'][n:],self.metrics['dx'][n:],'-r')
        ax3t.grid()

        ax4.plot(self.metrics['iteration'],self.metrics['m1'],':b')
        ax4.plot(self.metrics['iteration'],self.metrics['m2'],':b')
        ax4.set(xlabel='iteration',ylabel='$max <|\Delta Z_q|>$',yscale='log')
        ax4t=ax4.twinx()
        ax4t.plot(self.metrics['iteration'][n:],self.metrics['m1'][n:],'-r')
        ax4t.plot(self.metrics['iteration'][n:],self.metrics['m2'][n:],'-r')
        ax4t.grid()
        fig.tight_layout()
    
        
    def comp_eq_weights(self,
                        ny=6,
                        miter=1000,
                        dxmin=None,
                        iprint=10,
                        verbose=1):
        self.w = np.ones_like(self.r, self.prec)

        start = time.time()
        wo = self.w
        for i in range(miter):
            self.w = NPNEw(self.w, basis_poly_ry(self.w, self.r, ny), self.It)

            if i % iprint == 0:
                dx = self.w - wo
                wo = self.w
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                maxw = tf.math.reduce_max(self.w).numpy()
                minw = tf.math.reduce_min(self.w).numpy()
                if verbose > 1:
                    print(
                        'iteration %i, max(w)=%g, min(w)=%g, |dx|=%g, time=%g'
                        % (i, maxw, minw, dx, time.time() - start))
                if dxmin != None and dx < dxmin: break
        if verbose == 1:
            print('iteration %i, max(w)=%g, min(w)=%g, |dx|=%g, time=%g' %
                  (i, maxw, minw, dx, time.time() - start))

    def eq_plots(self, ldt=ldt[:3], dt_sim=1, suptitle=None):
        y_pred = self.r.numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)

        lx2, lzh2 = comp_ZCa_w(y_pred, a=-1, i_traj=self.i_traj, w=self.w)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$q$')
        ax1.grid()

        for dt in ldt:
            lx, ly = comp_ZC1_w(y_pred,
                                self.Ib,
                                self.r_boundary_future2,
                                self.r_boundary_past2,
                                self.delta_i_boundary_future2,
                                self.delta_i_boundary_past2,
                                self.index_boundary_past2,
                                self.i_traj,
                                self.w,
                                dt=tf.constant(dt))
            ax3.plot(lx.numpy()[:-1], -np.log(ly.numpy()[:-1]))
        ax3.set(ylabel='$-\ln Z_{C,1}$', xlabel='$q$')
        ax3.grid()

        r2si, r2delta_rdsi = comp_r2s_w_dt(y_pred,
                                           i_traj=self.i_traj,
                                           t_traj=self.t_traj,
                                           w=self.w)
        s = r2si(y_pred) * dt_sim**0.5
        lx2, lzh2 = comp_ZCa_w(s, a=-1, i_traj=self.i_traj, w=self.w)
        ax2.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{q}$')
        ax2.grid()
        fig.tight_layout()

    def plot_obs_pred(self, xlim=None, nbins=100, suptitle=None, halves=True):
        y_pred = self.r.numpy()
        y_test = self.r_boundary_future
        rmin = tf.math.reduce_min(y_pred)
        rmax = tf.math.reduce_max(y_pred)
        bin_edges = tf.linspace(rmin, rmax, nbins + 1)
        zero = tf.cast(0, self.prec)
        one = tf.cast(1, self.prec)

        nB = tf.where(self.index_boundary_future > -1, self.r_boundary_future,
                      zero)
        ni = tf.where(self.index_boundary_future > -1, one, zero)
        di = tf.where(self.index_boundary_future > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, y_pred, side='right') - 1
        hist_nB = tf.math.bincount(bin_indices,
                                   minlength=nbins + 1,
                                   weights=nB)
        hist_n = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ni)
        hist_d = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=di)
        hist_pB = hist_nB / hist_n
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        if halves:
            n2=len(y_pred)//2
            hist_nB1 = tf.math.bincount(bin_indices[:n2],
                                       minlength=nbins + 1,
                                       weights=nB[:n2])
            hist_n1 = tf.math.bincount(bin_indices[:n2], minlength=nbins + 1, weights=ni[:n2])
            hist_pB1 = hist_nB1 / hist_n1

            hist_nB2 = tf.math.bincount(bin_indices[n2:],
                                       minlength=nbins + 1,
                                       weights=nB[n2:])
            hist_n2 = tf.math.bincount(bin_indices[n2:], minlength=nbins + 1, weights=ni[n2:])
            hist_pB2 = hist_nB2 / hist_n2


        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)
        ax1.plot(lx, hist_pB[:-1], '-r', label='obs vs pred')
        if halves:
            ax1.plot(lx, hist_pB1[:-1], ':g', label='obs vs pred 1/2')
            ax1.plot(lx, hist_pB2[:-1], ':b', label='obs vs pred 2/2')
        ax1.plot((lx[0], lx[-1]), (lx[0], lx[-1]), ':k', label='obs = pred')
        ax1.set(xlabel='pB predicted', ylabel='pB observed')
        if xlim: ax1.set(xlim=xlim, ylim=xlim)
        ax1.legend()
        ax1.grid()

        ax2.plot(lx, hist_nB[:-1], '-r', label='hist observed nB')
        ax2.plot(lx, hist_n[:-1], '-b', label='hist observed total')
        ax2.plot(lx, hist_d[:-1], '-k', label='hist discarded')
        ax2.set(xlabel='pB predicted', ylabel='N', yscale='log')
        if xlim: ax2.set(xlim=xlim)
        ax2.legend()

        fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred)
        ax3.plot(fpr[1:], thresh[1:], 'r-', label='threshold')
        AUC = metrics.roc_auc_score(y_test, y_pred)
        ax3.plot(fpr, tpr, 'b-', label='tpr, AUC: %.2f%%' % (AUC * 100))
        ax3.set(xscale='log',
                xlabel='False positive rate',
                ylabel='True positive rate')
        ax3.grid()
        ax3.legend()

        fig.tight_layout()

    def plot_obs_pred_log(self, p_min=1e-4, nbins=100, suptitle=None):
        y_pred = self.r.numpy()
        y_pred_n0 = y_pred[y_pred > 0]
        rmin = tf.math.reduce_min(y_pred_n0)
        rmin = max(rmin, p_min)
        rmax = tf.math.reduce_max(y_pred)
        bin_edges = tf.exp(tf.linspace(np.log(rmin), np.log(rmax),
                                       nbins + 1))  # linear in log space
        bin_edges = tf.cast(bin_edges,self.prec)
        zero = tf.cast(0, self.prec)
        one = tf.cast(1, self.prec)

        nB = tf.where(self.index_boundary_future > -1, self.r_boundary_future,
                      zero)
        ni = tf.where(self.index_boundary_future > -1, one, zero)
        di = tf.where(self.index_boundary_future > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, y_pred, side='right') - 1
        hist_nB = tf.math.bincount(bin_indices,
                                   minlength=nbins + 1,
                                   weights=nB)
        hist_n = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ni)
        hist_d = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=di)
        hist_pB = hist_nB / hist_n
        lx = (bin_edges[:-1] * bin_edges[1:])**0.5

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)
        ax1.plot(lx, hist_pB[:-1], '-r', label='obs vs pred')
        ax1.plot((lx[0], lx[-1]), (lx[0], lx[-1]), ':k', label='obs = pred')
        ax1.set(xlabel='pB predicted',
                ylabel='pB observed',
                xscale='log',
                yscale='log')
        ax1.legend()
        ax1.grid()

        ax2.plot(lx, hist_nB[:-1], '-r', label='hist observed nB')
        ax2.plot(lx, hist_n[:-1], '-b', label='hist observed total')
        ax2.plot(lx, hist_d[:-1], '-k', label='hist discarded')
        ax2.set(xlabel='pB predicted', ylabel='N', yscale='log', xscale='log')
        ax2.legend()

        dx = np.array(bin_edges[1:] - bin_edges[:-1])
        p = (hist_n.numpy()[:-1]) / dx
        ax3.plot(lx, p, 'r-', label='P')
        ax3.set(xscale='log',
                xlabel='pB predicted',
                ylabel='Probability density of patients')
        ax3.legend()
        ax3.grid()

        fig.tight_layout()

    def plot_time_CDF(self,
                      lr=np.linspace(0, 1, 11, True),
                      xlim=None,
                      ylim=None,
                      nbins=10000,
                      suptitle=None):
        y_pred = self.r.numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        if suptitle is not None: fig.suptitle(suptitle,fontsize=16)
        for i in range(len(lr) - 1):
            r_in_range = (lr[i] < y_pred) & (y_pred < lr[i + 1]) & (
                self.r_boundary_future == 1)
            dt = self.delta_t_boundary_future[r_in_range]
            dt_max = tf.math.reduce_max(dt).numpy()
            bin_edges = tf.linspace(0, dt_max,
                                    nbins + 1)  # linear in log space

            bin_indices = tf.searchsorted(bin_edges, dt, side='right') - 1
            hist_dt = tf.math.bincount(bin_indices, minlength=nbins + 1)
            c_hist_dt = tf.cumsum(hist_dt)
            c_hist_dt /= c_hist_dt[-1]

            lx = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax1.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))
            ax2.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))

        ax1.legend(loc='lower right')
        ax1.set(xlabel='t', ylabel='CDF', yscale='log', ylim=(0.01, 0.1))
        ax1.grid(which='both')
        if xlim is not None: ax1.set(xlim=xlim)
        if ylim is not None: ax1.set(ylim=ylim)
        ax2.legend(loc='lower right')
        ax2.set(xlabel='t',
                ylabel='CDF',
                xscale='log',
                yscale='log',
                ylim=(0.01, 1))
        ax2.grid(which='both')

        fig.tight_layout()


@tf.function
def NPNEt(r, fk, It, t, gamma=0):
    """ implements NPNEt (non-parametric non-equilibrium mfpt
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    dfj = fk[:, :-1] - fk[:, 1:]

    akj = tf.tensordot(fk[:, :-1] * It, dfj, axes=[1, 1])

    if gamma != 0:
        diag = tf.linalg.diag_part(akj) + gamma * tf.math.reduce_sum(It)
        akj = tf.linalg.set_diag(akj, diag)

    delta_r = r[1:] - r[:-1] + t[1:] - t[:-1]

    b = tf.tensordot(fk[:, :-1], delta_r * It, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)

    rn = tf.clip_by_value(rn, 0, 1e10)
    return rn



@tf.function
def Imfpt_ne_vardt(r, It, t):
    return tf.math.reduce_mean(It * (tf.square(r[1:] - r[:-1]) - 2 *
                                     (t[1:] - t[:-1]) * (r[:-1] + r[1:])))


class MFPTNE:

    def __init__(self, IndA, i_traj, t_traj, prec=np.float64):
        self.Ib = np.zeros_like(IndA, prec)
        self.Ib[IndA] = 1
        self.i_traj = i_traj
        self.It = np.ones_like(IndA[:-1], prec)
        self.It[self.i_traj[1:] != self.i_traj[:-1]] = 0
        self.t_traj = t_traj.astype(prec)
        self.len = len(IndA)
        self.r = np.ones_like(IndA, prec)
        self.r[IndA] = 0
        self.prec = prec
        comp_auxiliary_arrays_ne_vardt(self)
        comp_auxiliary_arrays_ne_vardt2(self)

    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      iprint=1000,
                      gamma=0):

        ro = self.r

        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)
            fk = basis_poly_ry(self.r, y, ny, fenv)  # compute basis functions
            self.r = NPNEt(self.r, fk, self.It, self.t_traj,
                           gamma=gamma)  # find the optimal update of the RC

            if i % iprint == 0:
                I = Imfpt_ne_vardt(self.r, self.It, self.t_traj).numpy()

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                delta_r = tf.math.reduce_max(tf.math.abs(
                    self.r)) - tf.math.reduce_min(tf.math.abs(self.r))
                print('iteration %i, loss=%g, delta_r=%g, |dx|=%g, time=%g' %
                      (i, I, delta_r.numpy(), dx / delta_r.numpy(),
                       time.time() - start))
                if dxmin != None and dx / delta_r.numpy() < dxmin and i > 0:
                    break

    def plots(self, ldt=ldt, lnzmfpt=True, dt_sim=1):
        y_pred = self.r.numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        lx2, lzh2 = comp_ZCa(y_pred, a=-1)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$mfpt$')
        ax1.grid()

        for dt in ldt:
            lx, ly = comp_Zmfpt_ne(y_pred,
                                   self.Ib,
                                   self.r_boundary_future2,
                                   self.r_boundary_past2,
                                   self.delta_t_boundary_future2,
                                   self.delta_t_boundary_past2,
                                   self.i_traj,
                                   self.t_traj,
                                   dt=tf.constant(dt))
            ly = ly.numpy()[:-1]
            if lnzmfpt: ly = -np.log(ly)
            ax3.plot(lx.numpy()[:-1], ly)
        if lnzmfpt: ax3.set(ylabel='$-\ln Z_{mfpt}$', xlabel='$mfpt$')
        else: ax3.set(ylabel='$Z_{mfpt}$', xlabel='$mfpt$')
        ax3.grid()

        r2si, r2delta_rdsi = comp_r2s(y_pred, nbins=100000)
        s = r2si(y_pred) * dt_sim**0.5
        lx2, lzh2 = comp_ZCa(s, a=-1)
        ax2.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{mfpt}$')
        ax2.grid()
        fig.tight_layout()

    def comp_eq_weights(self,
                        ny=6,
                        miter=1000,
                        dxmin=None,
                        iprint=10,
                        verbose=1):
        self.w = np.ones_like(self.r, self.prec)

        start = time.time()
        wo = self.w
        for i in range(miter):
            self.w = NPNEw(self.w, basis_poly_ry(self.w, self.r, ny), self.It)

            if i % iprint == 0:
                dx = self.w - wo
                wo = self.w
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                maxw = tf.math.reduce_max(self.w).numpy()
                minw = tf.math.reduce_min(self.w).numpy()
                if verbose > 1:
                    print(
                        'iteration %i, max(w)=%g, min(w)=%g, |dx|=%g, time=%g'
                        % (i, maxw, minw, dx, time.time() - start))
                if dxmin != None and dx < dxmin: break
        if verbose == 1:
            print('iteration %i, max(w)=%g, min(w)=%g, |dx|=%g, time=%g' %
                  (i, maxw, minw, dx, time.time() - start))

    def eq_plots(self, dt_sim=1):
        y_pred = self.r.numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        lx2, lzh2 = comp_ZCa_w(y_pred, a=-1, i_traj=self.i_traj, w=self.w)
        ax1.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax1.set(ylabel='$F/kT$', xlabel='$mfpt$')
        ax1.grid()

        r2si, r2delta_rdsi = comp_r2s_w_dt(y_pred,
                                           i_traj=self.i_traj,
                                           t_traj=self.t_traj,
                                           w=self.w,
                                           nbins=100000)
        s = r2si(y_pred) * dt_sim**0.5
        lx2, lzh2 = comp_ZCa_w(s, a=-1, i_traj=self.i_traj, w=self.w)
        ax2.plot(lx2[:-2], -np.log(2 * lzh2[:-2]), 'r-')
        ax2.set(ylabel='$F/kT$', xlabel='$\\tilde{mfpt}$')
        ax2.grid()
        fig.tight_layout()

    def plot_obs_pred(self, nbins=100):
        y_pred = self.r.numpy()
        rmin = tf.math.reduce_min(y_pred)
        rmax = tf.math.reduce_max(y_pred)
        bin_edges = tf.linspace(rmin, rmax, nbins + 1)
        zero = tf.cast(0, self.prec)
        one = tf.cast(1, self.prec)

        ti = tf.where(self.index_boundary_future > -1,
                      self.delta_t_boundary_future, zero)
        ni = tf.where(self.index_boundary_future > -1, one, zero)
        di = tf.where(self.index_boundary_future > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, y_pred, side='right') - 1
        hist_t = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ti)
        hist_n = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ni)
        hist_d = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=di)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        hist_t = hist_t / hist_n
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax1.plot(lx, hist_t[:-1], '-r', label='obs vs pred')
        ax1.plot((lx[0], lx[-1]), (lx[0], lx[-1]), ':k', label='obs = pred')
        ax1.set(xlabel='mfpt predicted', ylabel='mfpt observed')
        ax1.legend()

        ax2.plot(lx, hist_n[:-1], '-r', label='hist observed')
        ax2.plot(lx, hist_d[:-1], '-k', label='hist discarded')
        ax2.set(xlabel='mfpt predicted', ylabel='N', yscale='log')
        ax2.legend()
        fig.tight_layout()

    def plot_time_CDF(self,
                      lr=np.linspace(0, 1, 11, True),
                      xlim=None,
                      ylim=None,
                      nbins=10000):
        y_pred = self.r.numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(len(lr) - 1):
            r_in_range = (lr[i] < y_pred) & (y_pred < lr[i + 1]) & (
                self.index_boundary_future > -1)
            dt = self.delta_t_boundary_future[r_in_range]
            dt_max = tf.math.reduce_max(dt).numpy()
            bin_edges = tf.linspace(0, dt_max,
                                    nbins + 1)  # linear in log space

            bin_indices = tf.searchsorted(bin_edges, dt, side='right') - 1
            hist_dt = tf.math.bincount(bin_indices, minlength=nbins + 1)
            c_hist_dt = tf.cumsum(hist_dt)
            c_hist_dt /= c_hist_dt[-1]

            lx = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax1.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))
            ax2.plot(lx,
                     c_hist_dt[:-1],
                     '-',
                     label='%g<r<%g' % (lr[i], lr[i + 1]))

        ax1.legend(loc='lower right')
        ax1.set(xlabel='t', ylabel='CDF', yscale='log', ylim=(0.01, 1))
        ax1.grid(which='both')
        if xlim is not None: ax1.set(xlim=xlim)
        if ylim is not None: ax1.set(ylim=ylim)
        ax2.legend(loc='lower right')
        ax2.set(xlabel='t',
                ylabel='CDF',
                xscale='log',
                yscale='log',
                ylim=(0.01, 1))
        ax2.grid(which='both')

        fig.tight_layout()

@tf.function
def NPqInf(r, fk, r_boundary_future, i_boundary_future, r_boundary_past, i_boundary_past):
    """ implements NPq (non-parametric committor optimization) iteration for
    infinite \Delta t
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    r_boundary_future(i) is the value of the next boundary the trajectory visits starting from i
    """

    delta_r=tf.where(i_boundary_future > -1,r_boundary_future - r,0)
    delta_r+=tf.where(i_boundary_past > -1,r_boundary_past - r,0)
    akj = tf.tensordot(fk, fk, axes=[1, 1])

    b = tf.tensordot(fk, delta_r/2, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn


@tf.function
def NPqInfNE(r, fk, r_boundary_future, i_boundary_future):
    """ implements NPq (non-parametric committor optimization) iteration for
    infinite \Delta t
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    r_boundary_future(i) is the value of the next boundary the trajectory visits starting from i
    """

    delta_r=tf.where(i_boundary_future > -1,r_boundary_future - r,0)
    akj = tf.tensordot(fk, fk, axes=[1, 1])

    b = tf.tensordot(fk, delta_r, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn



class CommittorInf(Committor):
    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      iprint=1000):

        ro = self.r

        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)
            fk = basis_poly_ry(self.r, y, ny, fenv)  # compute basis functions
            self.r = NPqInf(self.r, fk, self.r_boundary_future,
                            self.index_boundary_future,
                            self.r_boundary_past,
                            self.index_boundary_past)  # find the optimal update of the RC

            if i % iprint == 0:

                delta_r=tf.where(self.index_boundary_future > -1,self.r_boundary_future - self.r,0)
                mse=tf.math.reduce_mean(delta_r**2)
                delta_r=tf.where(self.index_boundary_past > -1,self.r_boundary_past - self.r,0)
                mse+=tf.math.reduce_mean(delta_r**2)
                mse=mse.numpy()/2

                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                print('iteration %i, mse=%g, delta_r2=%g, |dx|=%g, time=%g' %
                      (i, mse, delta_r2, dx, time.time() - start))
                if dxmin != None and dx < dxmin and i > 0: break

class CommittorM(Committor):
    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      iprint=1000):

        ro = self.r

        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)
            fk = basis_poly_ry(self.r, y, ny, fenv)  # compute basis functions
            self.r = NPqInf(self.r, fk, self.r_boundary_future,
                            self.index_boundary_future,
                            self.r_boundary_past,
                            self.index_boundary_past)  # find the optimal update of the RC

            if i % iprint == 0:

                delta_r=tf.where(self.index_boundary_future > -1,self.r_boundary_future - self.r,0)
                mse=tf.math.reduce_mean(delta_r**2)
                delta_r=tf.where(self.index_boundary_past > -1,self.r_boundary_past - self.r,0)
                mse+=tf.math.reduce_mean(delta_r**2)
                mse=mse.numpy()/2

                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                print('iteration %i, mse=%g, delta_r2=%g, |dx|=%g, time=%g' %
                      (i, mse, delta_r2, dx, time.time() - start))
                if dxmin != None and dx < dxmin and i > 0: break

                    
class CommittorInfNE(CommittorNE):
    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      iprint=1000):

        ro = self.r

        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r) * (1 - self.Ib)
            fk = basis_poly_ry(self.r, y, ny, fenv)  # compute basis functions
            self.r = NPqInfNE(self.r, fk, self.r_boundary_future,
                            self.index_boundary_future)  # find the optimal update of the RC

            if i % iprint == 0:

                delta_r=tf.where(self.index_boundary_future > -1,self.r_boundary_future - self.r,0)
                mse=tf.math.reduce_mean(delta_r**2).numpy()

                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                print('iteration %i, mse=%g, delta_r2=%g, |dx|=%g, time=%g' %
                      (i, mse, delta_r2, dx, time.time() - start))
                if dxmin != None and dx < dxmin and i > 0: break

@tf.function
def NPlmbdNE(r, fk, IA, IB, lmbdA, lmbdB, It, gamma=0):
    """ implements NPlmbd (non-parametric committor optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    dfk = fk[:, 1:] - fk[:, :-1]*(1+gamma+lmbdA * IA[:-1] + lmbdB * IB[:-1])

    akj = tf.tensordot(fk[:, :-1], dfk*It, axes=[1, 1])

    dr = r[1:] - r[:-1]*(1+lmbdB * IB[:-1] + lmbdA * IA[:-1]) + lmbdB * IB[:-1]

    b = tf.tensordot(fk[:, :-1], -dr * It, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, 1)
    return rn

class Soft_CommittorNE(Soft_Committor):
    def fit_transform(self,
                      comp_y,
                      comp_env=comp_env,
                      ny=6,
                      miter=100000,
                      dxmin=None,
                      delta_r2min=None,
                      iprint=1000,
                      comp_gamma=comp_gamma,
                      history=None,
                      basis_functions=basis_poly_ry):

        ro = self.r
        start = time.time()
        for i in range(miter + 1):
            y = tf.cast(comp_y(), self.prec)
            if i % 10 == 0: fenv = comp_env(self.r)

            if history is None:
                fk = basis_functions(self.r, y, ny,
                                 fenv)  # compute basis functions
            else:
                d=np.random.choice(history)
                fk = basis_functions(tf.roll(self.r,d,0), tf.roll(y,d,0), ny,
                                 fenv)  # compute basis functions

            gamma=tf.constant(comp_gamma(i,miter),dtype=self.prec)
            self.r = NPlmbdNE(
                    self.r,
                    fk,
                    self.IndA,
                    self.IndB,
                    self.lmbdA,
                    self.lmbdB,
                    self.It,
                    gamma)  # find the optimal update of the RC

            if i % iprint == 0:
                delta_r2 = committor_loss(self.r, self.It) * (self.len - 1) / 2

                dx = self.r - ro
                ro = self.r
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                delta_r = tf.math.reduce_max(tf.math.abs(
                    self.r)) - tf.math.reduce_min(tf.math.abs(self.r))
                delta_r = delta_r.numpy()
                print(
                    'iteration %i, delta_r2=%g, delta_r=%g, |dx|=%g, time=%g' %
                    (i, delta_r2 / delta_r / delta_r, delta_r, dx / delta_r,
                     time.time() - start))
                if delta_r2min != None and delta_r2 / delta_r / delta_r < delta_r2min and i > 0:
                    break
                if dxmin != None and dx / delta_r < dxmin and i > 0: break