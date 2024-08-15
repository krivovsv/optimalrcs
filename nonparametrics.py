import tensorflow as tf


@tf.function
def npq(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPq (non-parametric committor optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if i_traj is None:
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = tf.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1]
    b = tf.tensordot(dfk, -delta_r * itw, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj


@tf.function
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
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]

    akj = tf.tensordot(dfk * itw, dfk, axes=[1, 1])
    akj = akj + tf.tensordot(fk * (lmbd_a * ia_traj + lmbd_a * ib_traj), fk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])

    b = tf.tensordot(dfk, delta_r * itw, 1)
    b = b + tf.tensordot(fk, (lmbd_b * ib_traj * (1 - r_traj) - r_traj * lmbd_a * ia_traj), 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj


@tf.function
def npmfpt(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPmfpt (non-parametric mfpt optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a re-weighting factor,
        to use with multiple short trajectories. Default value is 1.
    """

    if i_traj is None:
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = tf.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])
    b = tf.tensordot(dfk, delta_r * itw, 1) + 2 * tf.math.reduce_sum(fk, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    return rn_traj


@tf.function
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
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)

    if stable:
        akj = -tf.tensordot(fk[:, :-1], fk[:, :-1] * itw, axes=[1, 1])
    else:
        delta_fj = fk[:, 1:] - fk[:, :-1] * (1 + gamma)
        akj = tf.tensordot(fk[:, :-1], delta_fj * itw, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1]
    b = tf.tensordot(fk[:, :-1], -delta_r * itw, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj

@tf.function
def npneq(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None):
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
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)

    if train_mask is not None:
        itw = itw * train_mask

    if stable:
        akj = -tf.tensordot(fk[:, :-1], fk[:, :-1] * itw, axes=[1, 1])
    else:
        delta_fj = fk[:, 1:] - fk[:, :-1] * (1 + gamma)
        akj = tf.tensordot(fk[:, :-1], delta_fj * itw, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1]
    b = tf.tensordot(fk[:, :-1], -delta_r * itw, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj

@tf.function
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
    it = tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype)
    delta_t_prec = tf.cast(dt, dtype=r_traj.dtype)
    not_crossed = tf.cast(tf.logical_or(future_boundary.index[:-dt] == - 1,
                                        future_boundary.delta_t[:-dt] > delta_t_prec), dtype=r_traj.dtype)

    delta_fj = fk[:, dt:] * not_crossed - fk[:, :-dt] * (1 + gamma)
    akj = tf.tensordot(fk[:, :-dt], delta_fj * it, axes=[1, 1])

    r_plus = tf.where(tf.logical_and(future_boundary.index[:-dt] > - 1, future_boundary.delta_t[:-dt] <= delta_t_prec),
                      future_boundary.r[:-dt], r_traj[dt:])
    delta_r = r_plus - r_traj[:-dt]
    b = tf.tensordot(fk[:, :-dt], -delta_r * it, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj


@tf.function
def npnew(r, fk, it):
    """ implements NPNEw (non-parametric non-equilibrium re-weighting factors
    optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indicator function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """

    dfk = fk[:, 1:] - fk[:, :-1]

    b = -tf.tensordot(dfk * it, r[:-1], 1)
    b = tf.reshape(b, [b.shape[0], 1])
    scale = tf.math.reduce_sum(1 - r[:-1] * it)
    scale = tf.reshape(scale, [1, 1])
    b = tf.concat((b, scale), 0)

    ones = tf.reshape(it, [1, it.shape[0]])
    dfk = tf.concat((dfk * it, ones), 0)
    akj = tf.tensordot(dfk, fk[:, :-1], axes=[1, 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = abs(rn)

    return rn
