import cut_profiles, metrics
import boundaries as bd
import cupy as cp
import numpy as np

ldt0 = [2**i for i in range(16)]


def plot_zc1(ax, r_traj, b_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, xlabel='$q$',
             ln=True, w_traj=None, dtmin=1):
    if ldt is None:
        ldt = ldt0
#    if isinstance(r_traj, cp.ndarray):
#        r_traj = r_traj.get()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    for dt in ldt:
        lx, ly = cut_profiles.comp_zc1_irreg(r_traj, b_traj, future_boundary, past_boundary, dt=cp.array(dt),
                                       i_traj=i_traj, w_traj=w_traj, dtmin=dtmin)
        if ln:
            ax.plot(lx.get()[:-1], -np.log(ly.get()[:-1]))
        else:
            ax.plot(lx.get()[:-1], ly.get()[:-1])

    if ln:
        ax.set(ylabel='$-\\ln Z_{C,1}$', xlabel=xlabel)
    else:
        ax.set(ylabel='$Z_{C,1}$', xlabel=xlabel)
    ax.grid()


def plot_zq(ax, r_traj, b_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, xlabel='$q$',
            w_traj=None, ln=False, force0=False):
    if ldt is None:
        ldt = ldt0
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    for dt in ldt:
        lx, ly = cut_profiles.comp_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, dt=dt)
        if force0: ly -= ly[0]
#        if force0 : ly-=cp.mean(ly[:-1])
        if ln:
            ax.plot(lx.get()[:-1], -np.log(ly.get()[:-1]))
            ylabel = '$-\\ln Z_q$'
        else:
            ax.plot(lx.get()[:-1], ly.get()[:-1])
            ylabel = '$Z_q$'
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid()

def plot_zt(ax, r_traj, b_traj, t_traj, i_traj=None, future_boundary=None, past_boundary=None, ldt=None, xlabel='$\\tau$',
            ln=False):
    if ldt is None:
        ldt = ldt0
#    if isinstance(r_traj, cp.ndarray):
#        r_traj = r_traj.get()
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, t_traj=t_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, t_traj=t_traj, i_traj=i_traj)

    for dt in ldt:
        lx, ly = cut_profiles.comp_zt(r_traj, b_traj, t_traj, i_traj, future_boundary, past_boundary, dt=dt)
        if ln:
            ax.plot(lx.get()[:-1], -np.log(ly.get()[:-1]))
            ylabel = '$-\\ln Z_t$'
        else:
            ax.plot(lx.get()[:-1], ly.get()[:-1])
            ylabel = '$Z_t$'
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid()


def plot_fep(ax, r_traj, i_traj=None, t_traj=None, w_traj=None, xlabel='$q$', natural=False, dt_sim=1, lt='r-'):
    if natural:
        r_traj = transform_q2qn(r_traj, i_traj=i_traj, t_traj=t_traj, w_traj=w_traj, dt_sim=dt_sim)
    r_traj=cp.asarray(r_traj)
    lx, lzh = cut_profiles.comp_zca(r_traj, a=-1, i_traj=i_traj, t_traj=t_traj, w_traj=w_traj)
    ax.plot(lx[:-2].get(), -np.log(2 * lzh[:-2].get()), lt)
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
    r2s = UnivariateSpline(lx.get(), ld1.get(), s=0).antiderivative()
    return r2s(r_traj.get()) * dt_sim ** 0.5

#def transform_q2q(self,dt):
#    _envelope = (1 - self.b_traj)
#
#    fk = optimalrcs.basis_poly_ry(self.r_traj, self.r_traj, 3, _envelope)
#            # compute next update of the RC
#    return nonparametrics.npneq_dt(self.r_traj, fk, self.i_traj, self.future_boundary, gamma=0, dt=dt)


def plot_obs_pred_q(ax, r_traj, future_boundary, nbins=100, halves=True, ax2=None, log_scale=False, log_scale_pmin=None):
    r_min = cp.amin(r_traj)
    r_max = cp.amax(r_traj)
    bin_edges = cp.linspace(r_min, r_max, nbins + 1)
    if log_scale:
        if log_scale_pmin is None:
            r_min = cp.amin(cp.where(r_traj >0, r_traj, r_max))
        else:
            r_min = max(r_min, log_scale_pmin)
        bin_edges = cp.exp(cp.linspace(np.log(r_min), np.log(r_max),
                                       nbins + 1))  # linear in log space


    def obs_pred(istart, iend, line_type, label, ax, ax2=None):

        nb = cp.where(future_boundary.index[istart:iend] > -1, future_boundary.r[istart:iend], 0)
        nab = cp.where(future_boundary.index[istart:iend] > -1, 1, 0)
        nd = cp.where(future_boundary.index[istart:iend] > -1, 0, 1)
        bin_indices = cp.searchsorted(bin_edges, r_traj[istart:iend], side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        hist_nb = cp.bincount(bin_indices, minlength=nbins + 1, weights=nb)
        hist_nab = cp.bincount(bin_indices, minlength=nbins + 1, weights=nab)
        hist_nd = cp.bincount(bin_indices, minlength=nbins + 1, weights=nd)
        hist_pb = hist_nb / hist_nab
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2
        if log_scale:
            lx = (bin_edges[:-1] * bin_edges[1:]) ** 0.5
        ax.plot(lx.get(), hist_pb[:-1].get(), line_type, label=label)
        if ax2 is not None:
            ax2.plot(lx.get(), hist_nb[:-1].get(), 'r-', label='nB' )
            ax2.plot(lx.get(), hist_nab[:-1].get(), 'b-', label='nA+nB' )
            ax2.plot(lx.get(), hist_nd[:-1].get(), 'k-', label='n discarded' )
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
    r_min = cp.amin(r_traj)
    r_max = cp.amax(r_traj)
    bin_edges = cp.linspace(r_min, r_max, nbins + 1)
    if log_scale:
        if log_scale_tmin is None:
            r_min = cp.amin(cp.where(r_traj >0, r_traj, r_max))
        else:
            r_min = cp.maximum(r_min, log_scale_tmin)
        bin_edges = cp.exp(cp.linspace(np.log(r_min), np.log(r_max),
                                       nbins + 1))  # linear in log space
        r_traj=cp.maximum(r_traj,r_min)
    r_min=r_min.get()
    r_max=r_max.get()

    def obs_pred(istart, iend, line_type, label, ax, ax2=None):

        ta = cp.where(future_boundary.index[istart:iend] > -1, future_boundary.delta_t[istart:iend], 0)
        na = cp.where(future_boundary.index[istart:iend] > -1, 1, 0)
        nd = cp.where(future_boundary.index[istart:iend] > -1, 0, 1)
        bin_indices = cp.searchsorted(bin_edges, r_traj[istart:iend], side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        hist_ta = cp.bincount(bin_indices, minlength=nbins + 1, weights=ta)
        hist_na = cp.bincount(bin_indices, minlength=nbins + 1, weights=na)
        hist_nd = cp.bincount(bin_indices, minlength=nbins + 1, weights=nd)
        hist_ta = hist_ta / hist_na
        lx = (bin_edges[:-1] + bin_edges[1:]) / 2
        if log_scale:
            lx = (bin_edges[:-1] * bin_edges[1:]) ** 0.5
        ax.plot(lx.get(), hist_ta[:-1].get(), line_type, label=label)
        if ax2 is not None:
            ax2.plot(lx.get(), hist_na[:-1].get(), 'r-', label='nA' )
            ax2.plot(lx.get(), hist_nd[:-1].get(), 'k-', label='n discarded' )
    obs_pred(0, -1, '-r', 'obs vs pred', ax, ax2)
    if halves:
        obs_pred(0, len(r_traj)//2, ':g', 'obs vs pred 1/2', ax)
        obs_pred(len(r_traj)//2, -1, ':b', 'obs vs pred 2/2', ax)
    ax.plot((r_min, r_max), (r_min, r_max), ':k', label='obs = pred')
    ax.set(xlabel='$\\tau$ predicted', ylabel='$\\tau$ observed')
    if log_scale:
        ax.set(xscale='log', yscale='log')
    else:
        ax.set(xlabel='$\\tau$ predicted', ylabel='$\\tau$ observed')
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
    ok = future_boundary.index > -1
    fpr, tpr, thresh = sklearn.metrics.roc_curve(future_boundary.r[ok].get(), r_traj[ok].get())
    ax.plot(fpr[1:], thresh[1:], 'r-', label='threshold')
    auc = sklearn.metrics.roc_auc_score(future_boundary.r[ok].get(), r_traj[ok].get())
    ax.plot(fpr, tpr, 'b-', label='tpr, AUC: %.2f%%' % (auc * 100))
    if log_scale:
        ax.set(xscale='log', xlabel='False positive rate', ylabel='True positive rate')
    else:
        ax.set(xlabel='False positive rate', ylabel='True positive rate')
    ax.legend()
    ax.grid()

def plot_pr_curve(ax, r_traj, future_boundary, log_scale=False):
    import sklearn.metrics
    ok = future_boundary.index > -1
    precision, recall, thresh = sklearn.metrics.precision_recall_curve(future_boundary.r[ok], r_traj[ok])
    ax.plot(recall[1:], thresh, 'r-', label='threshold')
    auc_pr = sklearn.metrics.auc(recall, precision)
    ax.plot(recall, precision, 'b-', label='precision, AUC: %.2f%%' % (auc_pr * 100))
    if log_scale:
        ax.set(xscale='log', xlabel='recall (TPR)', ylabel='precision')
    else:
        ax.set(xlabel='recall (TPR)', ylabel='precision')
    ax.legend()
    ax.grid()
    
def plot_bootstrap_zq(ax, r_traj, b_traj, i_traj=None, future_boundary=None, past_boundary=None, lp=None, t=2):
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    lx=[]
    ly=[]
    if lp is None:
        lp=[0.5**i for i in range(11)]
    for p in lp:
        w_traj = np.random.poisson(lam=p, size=len(r_traj))
        sd_zq=metrics._comp_max_delta_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, w_traj = w_traj)[2]
        lx.append(sum(w_traj))
        ly.append(sd_zq**2)
    
    import statsmodels.api as sm

    x=np.asarray(lx)
    y=np.asarray(ly)

    model = sm.OLS(y, x)
    results = model.fit()
    y_pred = results.predict(x)

    ax.scatter(x, y, label='Data', color='blue', alpha=0.6)
    ax.plot(x, y_pred, 'r-', label='~ size, F-stat = %g' %results.fvalue, alpha=0.6)

    if t==2:
        model = sm.OLS(y, x*x)
        results = model.fit()
        y_pred = results.predict(x*x)
        ax.plot(x, y_pred, 'g-', label='~ size$^2$, F-stat = %g' %results.fvalue, alpha=0.6)
    if t==3:
        X = np.column_stack((x, x*x))
        model = sm.OLS(y, X)
        results = model.fit()
        y_pred = results.predict(X)
        ax.plot(x, y_pred, 'g-', label='~ size + size$^2$, F-stat = %g' %results.fvalue, alpha=0.6)
    
    ax.set(title='Bootstrap analysis of sd. of $Z_q$', xlabel='data size', ylabel='sd. of $Z_q$', xscale='log', yscale='log')
    ax.legend()
    ax.grid()

def plot_bootstrap_sd_zq(ax, r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None, w_traj=None, ldt=None, mseed=10):
    if ldt is None:
        ldt = ldt0
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    ldti=[]
    lsd=[]
    for seed in range(mseed):
        np.random.seed(seed+100)
        if seed==0:
            w_traj = cp.ones_like(r_traj)
        else:
            w_traj = np.random.poisson(lam=1, size=len(r_traj))
            w_traj = cp.array(w_traj, dtype=r_traj.dtype)
        for dt in ldt:
            lx, lz = cut_profiles.comp_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, w_traj=w_traj, dt=cp.array(dt))
            lz = lz[:-1].get()
            m2 = np.mean((lz - np.mean(lz)) ** 2)**0.5
            ldti.append(dt)
            lsd.append(m2)
        if seed==0:
            ax.plot(ldti,lsd,'ro')
            ldti=[]
            lsd=[]
            
    ax.plot(ldti,lsd,'bx')
    ax.set(title='Bootstrap analysis of sd. of $Z_q$', xlabel='$\\Delta t$', ylabel='sd. of $Z_q$', xscale='log')

def plot_bootstrap_zq_dt(ax, dt, r_traj, b_traj=None, i_traj=None, future_boundary=None, past_boundary=None, w_traj=None, mseed=10):
    if future_boundary is None:
        future_boundary = bd.FutureBoundary(r_traj, b_traj, i_traj=i_traj)
    if past_boundary is None:
        past_boundary = bd.PastBoundary(r_traj, b_traj, i_traj=i_traj)

    for seed in range(mseed):
        np.random.seed(seed)
        if seed==0:
            w_traj = cp.ones_like(r_traj)
        else:
            w_traj = np.random.poisson(lam=1, size=len(r_traj))
            w_traj = cp.array(w_traj,dtype=r_traj.dtype)
        lx, lz = cut_profiles.comp_zq(r_traj, b_traj, i_traj, future_boundary, past_boundary, w_traj=w_traj, dt=cp.array(dt))
        lz = lz[:-1].get()
        lx = lx[:-1].get()
        if seed==0:
            ax.plot(lx,lz,'r-')
        else:
            ax.plot(lx,lz,'b:')
            
    ax.set(title='Bootstrap analysis of $Z_q$', xlabel='q', ylabel='$Z_q$')

