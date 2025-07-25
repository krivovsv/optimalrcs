import cupy as cp


def npq(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPq (non-parametric committor optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = cp.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1]
    b = cp.tensordot(dfk, -delta_r * itw, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    rn_traj = cp.clip(rn_traj, 0, 1)
    return rn_traj



def npqsoft(r_traj, fk, ia_traj, ib_traj, lmbd_a, lmbd_b, i_traj=None, w_traj=None):
    """ implements NPlmbd (non-parametric committor optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]

    akj = cp.tensordot(dfk * itw, dfk, axes=[1, 1])
    akj = akj + cp.tensordot(fk * (lmbd_a * ia_traj + lmbd_b * ib_traj), fk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])

    b = cp.tensordot(dfk, delta_r * itw, 1)
    b = b + cp.tensordot(fk, (lmbd_b * ib_traj * (1 - r_traj) - r_traj * lmbd_a * ia_traj), 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    rn_traj = cp.clip(rn_traj, 0, 1)
    return rn_traj



def npt(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPt (non-parametric mfpt optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a re-weighting factor,
        to use with multiple short trajectories. Default value is 1.
    """

    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = cp.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])
    b = cp.tensordot(dfk, delta_r * itw, 1) + 2 * cp.math.reduce_sum(fk, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    return rn_traj


def npneq(r_traj, fk, i_traj=None, gamma=0, stable=False):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indicator function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)

    if stable:
        akj = -cp.tensordot(fk[:, :-1], fk[:, :-1] * itw, axes=[1, 1])
    else:
        delta_fj = fk[:, 1:] - fk[:, :-1] * (1 + gamma)
        akj = cp.tensordot(fk[:, :-1], delta_fj * itw, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1]
    b = cp.tensordot(fk[:, :-1], -delta_r * itw, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    rn_traj = cp.clip(rn_traj, 0, 1)
    return rn_traj



def npneq_dt(r_traj, fk, i_traj, future_boundary, gamma=0, dt=1):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    it = (i_traj[dt:] == i_traj[:-dt]).astype(r_traj.dtype)
    delta_t_prec = cp.cast(dt, dtype=r_traj.dtype)
    not_crossed = (cp.logical_or(future_boundary.index[:-dt] == -1,
                                  future_boundary.delta_t[:-dt] > delta_t_prec)).astype(r_traj.dtype)

    delta_fj = fk[:, dt:] * not_crossed - fk[:, :-dt] * (1 + gamma)
    akj = cp.tensordot(fk[:, :-dt], delta_fj * it, axes=[1, 1])

    r_plus = cp.where(cp.logical_and(future_boundary.index[:-dt] > -1, future_boundary.delta_t[:-dt] <= delta_t_prec),
                      future_boundary.r[:-dt], r_traj[dt:])
    delta_r = r_plus - r_traj[:-dt]
    b = cp.tensordot(fk[:, :-dt], -delta_r * it, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    rn_traj = cp.clip(rn_traj, 0, 1)
    return rn_traj



def npnew(r, fk, it):
    """ implements NPNEw (non-parametric non-equilibrium re-weighting factors
    optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indicator function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """

    dfk = fk[:, 1:] - fk[:, :-1]

    b = -cp.tensordot(dfk * it, r[:-1], 1)
    b = cp.reshape(b, [b.shape[0], 1])
    scale = cp.sum(1 - r[:-1] * it)
    scale = cp.reshape(scale, [1, 1])
    b = cp.concatenate((b, scale), 0)

    ones = cp.reshape(it, [1, it.shape[0]])
    dfk = cp.concatenate((dfk * it, ones), 0)
    akj = cp.tensordot(dfk, fk[:, :-1], axes=[1, 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn = r + cp.tensordot(al_j, fk, 1)
    rn = abs(rn)

    return rn



def npnet(r_traj, fk, t_traj, i_traj, gamma=0, t_max=1e10, subsample=None):
    """ implements NPNEt (non-parametric non-equilibrium mfpt
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)

    dfj = fk[:, 1:] - fk[:, :-1] * (1 + gamma)
    

    akj = cp.tensordot(fk[:, :-1], dfj * itw, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1] + t_traj[1:] - t_traj[:-1]

    b = cp.tensordot(fk[:, :-1], -delta_r * itw, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn = r_traj + cp.tensordot(al_j, fk, 1)
    if subsample is not None:
        k = int(len(rn) / subsample)
        t_max = cp.math.reduce_min(t_max, cp.math.reduce_min(cp.math.reduce_max(cp.reshape(rn[:k * subsample], [subsample, k]), 1)))
    rn = cp.clip(rn, 0, t_max)
    return rn
