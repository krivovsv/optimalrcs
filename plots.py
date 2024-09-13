import cut_profiles
import boundaries as bd
import tensorflow as tf
import numpy as np

ldt0 = [2**i for i in range(16)]


def plot_zc1(ax, r_traj, b_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, xlabel='$q$',
             ln=True, w_traj=None):
    if ldt is None:
        ldt = ldt0
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    for dt in ldt:
        lx, ly = cut_profiles.comp_zc1(r_traj, b_traj, future_boundary, past_boundary, dt=tf.constant(dt),
                                       i_traj=i_traj, w_traj=w_traj)
        if ln:
            ax.plot(lx.numpy()[:-1], -np.log(ly.numpy()[:-1]))
        else:
            ax.plot(lx.numpy()[:-1], ly.numpy()[:-1])

    if ln:
        ax.set(ylabel='$-\ln Z_{C,1}$', xlabel=xlabel)
    else:
        ax.set(ylabel='$Z_{C,1}$', xlabel=xlabel)
    ax.grid()


def plot_zq(ax, r_traj, b_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, xlabel='$q$',
            ln=False):
    if ldt is None:
        ldt = ldt0
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    for dt in ldt:
        lx, ly = cut_profiles.comp_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, dt=tf.constant(dt))
        if ln:
            ax.plot(lx.numpy()[:-1], -np.log(ly.numpy()[:-1]))
            ylabel = '$-\ln Z_q$'
        else:
            ax.plot(lx.numpy()[:-1], ly.numpy()[:-1])
            ylabel = '$Z_q$'
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid()

def plot_zt(ax, r_traj, b_traj, t_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, xlabel='$mfpt$',
            ln=False):
    if ldt is None:
        ldt = ldt0
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, t_traj=t_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, t_traj=t_traj, i_traj=i_traj)

    for dt in ldt:
        lx, ly = cut_profiles.comp_zt(r_traj, b_traj, t_traj, i_traj, future_boundary, past_boundary, dt=tf.constant(dt))
        if ln:
            ax.plot(lx.numpy()[:-1], -np.log(ly.numpy()[:-1]))
            ylabel = '$-\ln Z_t$'
        else:
            ax.plot(lx.numpy()[:-1], ly.numpy()[:-1])
            ylabel = '$Z_t$'
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid()


def plot_fep(ax, r_traj, i_traj=None, t_traj=None, w_traj=None, xlabel='$q$', natural=False, dt_sim=1, lt='r-'):
    if natural:
        r_traj = transform_q2qn(r_traj, i_traj=i_traj, t_traj=t_traj, w_traj=w_traj, dt_sim=dt_sim)
    lx, lzh = cut_profiles.comp_zca(r_traj, a=-1, i_traj=i_traj, t_traj=t_traj, w_traj=w_traj)
    ax.plot(lx[:-2], -np.log(2 * lzh[:-2]), lt)
    ax.set(ylabel='$F/kT$', xlabel=xlabel)
    ax.grid()


def transform_q2qn(r_traj, i_traj=None, t_traj=None, w_traj=None, nbins=10000, dt_sim=1):
    """computes transformation from r to s - the natural
    coordinate, where diffusion coefficient is D(s)=1

    returns cubic splines, that approximate functions
    r->s and r->delta_r/ds

    r - the putative RC timeseries
    dx - discretization step
    """
    from scipy.interpolate import UnivariateSpline

    lx, lzh = cut_profiles.comp_zca(r_traj, a=-1, i_traj=i_traj, t_traj=t_traj, w_traj=w_traj, nbins=nbins)
    lzh = lzh * 2

    lx1, lzc1 = cut_profiles.comp_zca(r_traj, a=1, i_traj=i_traj, w_traj=w_traj, nbins=nbins)
    ld = np.sqrt((lzc1 + 1) / (lzh + 1))

    # r2delta_rds = UnivariateSpline(lx, ld, s=0)

    ld1 = 1 / ld
    r2s = UnivariateSpline(lx, ld1, s=0).antiderivative()
    return r2s(r_traj) * dt_sim ** 0.5


def plot_obs_pred_q(ax, r_traj, future_boundary, nbins=100, halves=True, ax2=None, log_scale=False, log_scale_pmin=None):
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()
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

    zero = tf.cast(0, dtype=r_traj.dtype)
    one = tf.cast(1, dtype=r_traj.dtype)

    def obs_pred(istart, iend, line_type, label, ax, ax2=None):

        nb = tf.where(future_boundary.index[istart:iend] > -1, future_boundary.r[istart:iend], zero)
        nab = tf.where(future_boundary.index[istart:iend] > -1, one, zero)
        nd = tf.where(future_boundary.index[istart:iend] > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, r_traj[istart:iend], side='right') - 1
        hist_nb = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=nb)
        hist_nab = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=nab)
        hist_nd = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=nd)
        hist_pb = hist_nb / hist_nab
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2
        if log_scale:
            lx = (bin_edges[:-1] * bin_edges[1:]) ** 0.5
        ax.plot(lx, hist_pb[:-1], line_type, label=label)
        if ax2 is not None:
            ax2.plot(lx, hist_nb[:-1], 'r-', label='nB' )
            ax2.plot(lx, hist_nab[:-1], 'b-', label='nA+nB' )
            ax2.plot(lx, hist_nd[:-1], 'k-', label='n discarded' )
    obs_pred(0, -1, '-r', 'obs vs pred', ax, ax2)
    if halves:
        obs_pred(0, len(r_traj)//2, ':g', 'obs vs pred 1/2', ax)
        obs_pred(len(r_traj)//2, -1, ':b', 'obs vs pred 2/2', ax)
    ax.plot((0, 1), (0, 1), ':k', label='obs = pred')
    ax.set(xlabel='pB predicted', ylabel='pB observed')
    if log_scale:
        ax.set(xscale='log', yscale='log')
    ax.legend()
    ax.grid()
    if ax2 is not None:
        ax2.set(xlabel='pB predicted', ylabel='#', yscale='log')
        #ax2.set_yscale('log')
        ax2.legend()
        ax2.grid()
        if log_scale:
            ax2.set(xscale='log', yscale='log')

def plot_obs_pred_t(ax, r_traj, future_boundary, nbins=100, halves=True, ax2=None, log_scale=False, log_scale_tmin=None):
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()
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

    zero = tf.cast(0, dtype=r_traj.dtype)
    one = tf.cast(1, dtype=r_traj.dtype)

    def obs_pred(istart, iend, line_type, label, ax, ax2=None):

        ta = tf.where(future_boundary.index[istart:iend] > -1, future_boundary.delta_t[istart:iend], zero)
        na = tf.where(future_boundary.index[istart:iend] > -1, one, zero)
        nd = tf.where(future_boundary.index[istart:iend] > -1, zero, one)
        bin_indices = tf.searchsorted(bin_edges, r_traj[istart:iend], side='right') - 1
        hist_ta = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=ta)
        hist_na = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=na)
        hist_nd = tf.math.bincount(bin_indices, minlength=nbins + 1, weights=nd)
        hist_ta = hist_ta / hist_na
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2
        if log_scale:
            lx = (bin_edges[:-1] * bin_edges[1:]) ** 0.5
        ax.plot(lx, hist_ta[:-1], line_type, label=label)
        if ax2 is not None:
            ax2.plot(lx, hist_na[:-1], 'r-', label='nA' )
            ax2.plot(lx, hist_nd[:-1], 'k-', label='n discarded' )
    obs_pred(0, -1, '-r', 'obs vs pred', ax, ax2)
    if halves:
        obs_pred(0, len(r_traj)//2, ':g', 'obs vs pred 1/2', ax)
        obs_pred(len(r_traj)//2, -1, ':b', 'obs vs pred 2/2', ax)
    ax.plot((r_min, r_max), (r_min, r_max), ':k', label='obs = pred')
    ax.set(xlabel='mfpt predicted', ylabel='mfpt observed')
    if log_scale:
        ax.set(xscale='log', yscale='log')
    ax.legend()
    ax.grid()
    if ax2 is not None:
        ax2.set(xlabel='mfpt predicted', ylabel='#', yscale='log')
        #ax2.set_yscale('log')
        ax2.legend()
        ax2.grid()
        if log_scale:
            ax2.set(xscale='log', yscale='log')


def plot_roc_curve(ax, r_traj, future_boundary, log_scale=False):
    import sklearn.metrics
    if tf.is_tensor(r_traj):
        r_traj = r_traj.numpy()

    ok = future_boundary.index > -1
    fpr, tpr, thresh = sklearn.metrics.roc_curve(future_boundary.r[ok], r_traj[ok])
    ax.plot(fpr[1:], thresh[1:], 'r-', label='threshold')
    auc = sklearn.metrics.roc_auc_score(future_boundary.r[ok], r_traj[ok])
    ax.plot(fpr, tpr, 'b-', label='tpr, AUC: %.2f%%' % (auc * 100))
    if log_scale:
        ax.set(xscale='log', xlabel='False positive rate', ylabel='True positive rate')
    else:
        ax.set(xlabel='False positive rate', ylabel='True positive rate')
    ax.legend()
    ax.grid()
