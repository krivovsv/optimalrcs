import numpy as np
import cupy as cp
from . import boundaries
from . import metrics
from . import nonparametrics
from . import plots
import time
import matplotlib.pyplot as plt


envelope_scale = 0.01

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def envelope_sigmoid(r, iter, max_iter):
    r0 = r[np.random.randint(r.shape[0])]
    delta_r = cp.max(cp.abs(r)) - cp.min(cp.abs(r))
    if delta_r < 1e-5:
        delta_r = 1e-5
    if np.random.random() < 0.5:
        return sigmoid((r - r0) / envelope_scale / delta_r)
    return sigmoid(-(r - r0) / envelope_scale / delta_r)


def basis_poly_ry(r, y, n, fenv=None):
    """computes basis functions as terms of polynomial of variables r and y

    r is the putative RC time-series
    y is a randomly chosen collective variable or coordinate to improve r
    n is the degree of the polynomial
    fenv is common envelope to focus optimization on a particular region
    """
    r = r / cp.max(cp.abs(r))
    y = y / cp.max(cp.abs(y))

    if fenv is None:
        f = cp.ones_like(r)
    else:
        f = cp.array(fenv, copy=True)

    fk = []
    for iy in range(n + 1):
        fr = cp.array(f, copy=True)
        for ir in range(n + 1 - iy):
            fk.append(fr)
            fr = fr * r
        f = f * y
    return cp.stack(fk)


class CommittorNE:
    def __init__(self, boundary0, boundary1, i_traj=None, t_traj=None, seed_r=None, prec=cp.float64):
        self.boundary0 = cp.asarray(boundary0)
        self.boundary1 = cp.asarray(boundary1)
        self.b_traj = cp.asarray(boundary0 | boundary1, dtype=prec)
        self.i_traj=None
        if i_traj is not None: self.i_traj = cp.asarray(i_traj)
        self.t_traj=None
        if t_traj is not None: self.t_traj = cp.asarray(t_traj)
        if seed_r is not None:
            self.r_traj = cp.asarray(seed_r, dtype=prec)
        else:
            self.r_traj = cp.ones_like(self.boundary0, dtype=prec) / 2
            self.r_traj[self.boundary0] = 0
            self.r_traj[self.boundary1] = 1
        self.prec = prec
        self.len = len(self.boundary0)
        self.future_boundary = boundaries.FutureBoundary(self.r_traj, self.b_traj, self.t_traj, self.i_traj)
        self.past_boundary = boundaries.PastBoundary(self.r_traj, self.b_traj, self.t_traj, self.i_traj)
        self.metrics_history = {}
        self.iter = 0
        self.p2i0 = None
        self.w_traj = None
        
    def set_fixed_traj_length_trap(self, trap_boundary, traj_length):
        self.future_boundary.set_distance_to_end_fixed_traj_length_trap(self.i_traj, trap_boundary, traj_length)
        
    def set_poisson_traj_length_trap(self, trap_boundary, traj_length=None):
        self.future_boundary.set_distance_to_end_poisson_traj_length_trap(self.i_traj, trap_boundary, traj_length)

    def print_metrics(self, metrics_print):
        s = ''
        for metric in metrics_print:
            s += '%s=%g, ' % (metrics.metrics_short_name[metric], self.metrics_history[metric][-1])
        print(s[:-2])

    def compute_metrics(self, metrics_print):
        for metric in metrics_print:
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            self.metrics_history[metric].append(metrics.metric2function[metric](self))

    def fit_transform(self, comp_y,
                      envelope=envelope_sigmoid, gamma=0, basis_functions=basis_poly_ry, ny=6,
                      max_iter=100000, min_delta_x=None, min_delta_r2=None,
                      print_step=1000, metrics_print=None, stable=False,
                      history_delta_t=None, history_type=None, history_shift_type=None,
                      save_min_delta_zq=True, train_mask=None, delta_r2_max_change_allowed=1e3, cupy_type=1):
        self.r_traj_old = self.r_traj
        self.time_start = time.time()
        if metrics_print is None:
            metrics_print = ('iter', 'cross_entropy', 'mse', 'max_sd_zq', 'max_grad_zq', 'delta_r2', 'auc', 'delta_x', 'time_elapsed')
        self.min_delta_zq = 10000
        _envelope = (1 - self.b_traj)
        if not callable(gamma):
            _gamma = gamma
        if self.i_traj is None:
            It=cp.ones_like(self.r_traj[:-1])
        else:
            It=cp.asarray(self.i_traj[1:] == self.i_traj[:-1],dtype=self.r_traj.dtype)
        delta_r2=cp.sum(It*cp.square(self.r_traj[1:] - self.r_traj[:-1]))
        
        for iter in range(max_iter + 1):
            self.iter+=1

            # compute next CV y, and cast it to the required accuracy
            y = cp.asarray(comp_y(), dtype=self.prec)

            

            # compute envelope, modulating the basis functions
            if iter % 10 == 0 and callable(envelope):
                _envelope = envelope(self.r_traj, iter, max_iter) * (1 - self.b_traj)

            # compute the basis functions
            if history_delta_t is None:
                y1, y2 = self.r_traj, y
            else:
                y1, y2 = self.history_select_y1y2(y, history_delta_t, history_type, history_shift_type)
            fk = basis_functions(y1, y2, ny, _envelope)


            # compute the gamma parameter
            if callable(gamma):
                _gamma = gamma(self.iter, max_iter)

            # compute next update of the RC
            if cupy_type==1:
                r_traj = nonparametrics.npneq(self.r_traj, fk, self.i_traj, _gamma, stable)
            if cupy_type==2:
                r_traj = nonparametrics.npneq(self.r_traj[:-1], fk[:,:-1], self.i_traj[:-1], _gamma, stable)
                r_traj = cp.concatenate([r_traj, cp.array([0])])

            delta_r2_new = cp.sum(It*cp.square(r_traj[1:] - r_traj[:-1]))
            if delta_r2_new-delta_r2<delta_r2_max_change_allowed:
                self.r_traj=r_traj
                delta_r2=delta_r2_new
            

            # compute and print various metrics
            if self.iter % print_step == 0:
                self.compute_metrics(metrics_print)
                self.print_metrics(metrics_print)
                self.r_traj_old = self.r_traj
                if iter > 0:
                    if save_min_delta_zq:
                        if self.metrics_history['max_sd_zq'][-1] < self.min_delta_zq:
                            self.min_delta_zq = self.metrics_history['max_sd_zq'][-1]
                            self.r_traj_min_sd_zq = self.r_traj
                    if min_delta_x is not None and self.metrics_history['delta_x'][-1] < min_delta_x:
                        break
                    if min_delta_r2 is not None and self.metrics_history['delta_r2'][-1] < min_delta_r2:
                        break

    def history_select_y1y2(self, y, history_delta_t, history_type, history_shift_type):
        d = np.random.choice(history_delta_t)
        if d > 0:
            if history_type is None:
                history_type = 'y(t-d),r(t-d)'
            if history_type == 'y(t-d),r(t-d)' and history_shift_type is None:
                if self.i_traj is not None: ### prepend d zeros to self.i_traj[d:] == self.i_traj[:-d]]
                    it = cp.concatenate([cp.zeros([d]), (self.i_traj[d:] == self.i_traj[:-d]).astype(self.r_traj.dtype)], 0)
                else:                       ### prepend d zeros to n-d ones 
                    it = cp.concatenate([cp.zeros([d]), cp.ones([self.len - d])], 0)
                y1 = cp.where(it, cp.roll(y, d, 0), 0)
                y2 = cp.where(it, cp.roll(self.r_traj, d, 0), 0)
            elif history_type == 'y(t-d),y(t)' and history_shift_type is None:
                if self.i_traj is not None:
                    it = cp.concatenate([cp.zeros([d]), (self.i_traj[d:] == self.i_traj[:-d]).astype(self.r_traj.dtype)], 0)
                else:                       ### prepend d zeros to n-d ones 
                    it = cp.concatenate([cp.zeros([d]), cp.ones([self.len - d])], 0)
                y1 = cp.where(it, cp.roll(y, d, 0), 0)
                y2 = y
            else:
                y1, y2 = self._history_select_y(y, d, history_type, history_shift_type)
        else:
            y1, y2 = self.r_traj, y
        return y1, y2

    def _history_select_y(self, y, d, history_type, history_shift_type):
        if history_shift_type is None:
            history_shift_type = 'r(t0)'
        if history_type is None:
            history_type = 'y(t-d),r(t-d)'
        if history_shift_type == 'r(t0)' and self.p2i0 is None:
            # pointer to the first frame of trajectory defined by i_traj
            changes = cp.diff(self.i_traj, prepend=self.i_traj[0]-1) != 0
            first_indices = cp.where(changes)[0]
            self.p2i0 = cp.asarray(np.repeat(first_indices.get(), cp.diff(cp.append(first_indices, len(self.i_traj))).get()))
            #self.p2i0 = np.repeat(first_indices.get(), cp.diff(cp.append(first_indices, len(self.i_traj))).get())

        def shift_y(d, i_traj, y, shift_type, p2i0): # which point to select when previous point at (t-d) belongs to other trajectory
            if shift_type == 'r(t-d)':  # do nothing, take previous values y(t-d) disregarding trajectory info i_traj
                return cp.roll(y, d, 0)
            elif shift_type == 'r(t)':  # take y(t) instead of y(t-d)
                return cp.where(cp.roll(i_traj, d, 0) == i_traj, cp.roll(y, d, 0), y)
            elif shift_type == 'r(t0)':  # take first frame of trajectory, y(t0) instead of y(t-d)
                return cp.where(cp.roll(i_traj, d, 0) == i_traj, cp.roll(y, d, 0), y[p2i0])
            else:  # take 0 instead of y(t-d)
                return cp.where(cp.roll(i_traj, d, 0) == i_traj, cp.roll(y, d, 0), 0)

        if len(history_type) > 1:
            d1, d2 = history_type[np.random.randint(len(history_type))].split(',')
        else:
            d1, d2 = history_type[0].split(',')
        if d1 == 'r(t)':
            y1 = self.r_traj
        if d1 == 'y(t)':
            y1 = y
        if d1 == 'r(t-d)':
            y1 = shift_y(d, self.i_traj, self.r_traj, history_shift_type, self.p2i0)
        if d1 == 'y(t-d)':
            y1 = shift_y(d, self.i_traj, y, history_shift_type, self.p2i0)
        if d1 == 'lndt':
            y1 = shift_y(d, self.i_traj, self.t_traj, history_shift_type, self.p2i0)
            y1 = cp.where(y1 > 0, cp.log(self.t_traj-y1+1e-5), 0)
        if d1 == 'dt':
            y1 = shift_y(d, self.i_traj, self.t_traj, history_shift_type, self.p2i0)
            y1 = cp.where(y1 > 0, self.t_traj-y1, 0)
        if d2 == 'r(t)':
            y2 = self.r_traj
        if d2 == 'y(t)':
            y2 = y
        if d2 == 'r(t-d)':
            y2 = shift_y(d, self.i_traj, self.r_traj, history_shift_type, self.p2i0)
        if d2 == 'y(t-d)':
            y2 = shift_y(d, self.i_traj, y, history_shift_type, self.p2i0)
        if d2 == 'lndt':
            y2 = shift_y(d, self.i_traj, self.t_traj, history_shift_type, self.p2i0)
            y2 = cp.where(y2 > 0,  cp.log(self.t_traj-y2+1e-5), 0)
        if d2 == 'dt':
            y2 = shift_y(d, self.i_traj, self.t_traj, history_shift_type, self.p2i0)
            y2 = cp.where(y2 > 0,  self.t_traj-y2, 0)
        return y1, y2

    def plots_metrics(self, metrics=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 4))
        if metrics is None:
            metrics=[]
            for m in self.metrics_history:
                if m != 'iter':
                    metrics.append(m)
                if len(m) == 3: break
        
        for m, ax in zip(metrics, (ax1,ax2,ax3)):
            n=len(self.metrics_history['iter'])//2
            ax.plot(self.metrics_history['iter'][1:],self.metrics_history[m][1:],':b')
            axt=ax.twinx()
            axt.plot(self.metrics_history['iter'][n:],self.metrics_history[m][n:],'-r')
            axt.grid()
            ax.set(xlabel='iteration',ylabel=m)

    def plots_feps(self, r_traj=None, delta_t_sim=1, ldt=None, reweight=False, dtmin=1, zq_force0=False):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if r_traj is None:
            r_traj = self.r_traj
        if reweight:
            if self.w_traj is None:
                self.comp_eq_weights()
            plots.plot_fep(ax1, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, w_traj=self.w_traj)
            plots.plot_fep(ax2, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, w_traj=self.w_traj,
                           natural=True, dt_sim=delta_t_sim)
            plots.plot_zc1(ax3, r_traj, self.b_traj, self.i_traj, self.future_boundary, self.past_boundary, ldt=ldt,
                           w_traj=self.w_traj, dtmin=dtmin)
        else:
            plots.plot_fep(ax1, r_traj, i_traj=self.i_traj, t_traj=self.t_traj)
            plots.plot_fep(ax2, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, natural=True, dt_sim=delta_t_sim)
            plots.plot_zq(ax3, r_traj, self.b_traj, self.i_traj, self.future_boundary, self.past_boundary, ldt=ldt, force0=zq_force0)
        fig.tight_layout()
        plt.show()

    def plots_obs_pred(self, r_traj=None, log_scale=False, log_scale_pmin=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if r_traj is None:
            r_traj = self.r_traj
        plots.plot_obs_pred_q(ax1, r_traj, self.future_boundary, ax2=ax2, log_scale=log_scale, log_scale_pmin=log_scale_pmin)
        plots.plot_roc_curve(ax3, r_traj, self.future_boundary, log_scale=log_scale)
        fig.tight_layout()
        plt.show()

    def comp_eq_weights(self, ny=6, max_iter=1000, min_delta_x=1e-5, print_step=10, verbose=1, cupy_type=1):
        self.w_traj = cp.ones_like(self.r_traj, dtype=self.prec)

        start = time.time()
        wo = self.w_traj
        if self.i_traj is None:
            it=cp.ones_like(self.r_traj[:-1])
        else:
            it=cp.asarray(self.i_traj[1:] == self.i_traj[:-1], dtype=self.r_traj.dtype)
        for i in range(max_iter):
            self.w_traj = nonparametrics.npnew(self.w_traj, basis_poly_ry(self.w_traj, self.r_traj, ny), it)
            if cupy_type==1:
                self.w_traj = nonparametrics.npnew(self.w_traj, basis_poly_ry(self.w_traj, self.r_traj, ny), it)
            if cupy_type==2:
                self.w_traj = nonparametrics.npnew(self.w_traj[:-1], basis_poly_ry(self.w_traj, self.r_traj, ny)[:,:-1], it[:-1])
                self.w_traj = cp.concatenate([self.w_traj, cp.array([0])])

            if i % print_step == 0:
                dx = self.w_traj - wo
                wo = self.w_traj
                dx = (cp.mean(dx**2)**0.5)
                max_w = cp.max(self.w_traj)
                min_w = cp.min(self.w_traj)
                if verbose > 1:
                    print('iteration %i, max(w)=%g, min(w)=%g, |dx|=%g, time=%g'
                           % (i, max_w, min_w, dx, time.time() - start))
                if min_delta_x != None and dx < min_delta_x: break
        if verbose == 1:
            print('iteration %i, max(w)=%g, min(w)=%g, |dx|=%g, time=%g' %
                  (i, max_w, min_w, dx, time.time() - start))


class MFPTNE(CommittorNE):
    def __init__(self, boundary0, i_traj=None, t_traj=None, seed_r=None, prec=np.float64):
        self.boundary0 = boundary0
        self.b_traj = cp.asarray(boundary0, dtype=prec)
        self.i_traj=None
        if i_traj is not None: self.i_traj = cp.asarray(i_traj)
        if t_traj is not None:
            self.t_traj = cp.asarray(t_traj, dtype=prec)
        else:
            self.t_traj = cp.arange(len(self.b_traj), dtype=prec)
        if seed_r is not None:
            self.r_traj = cp.asarray(seed_r, dtype=prec)
        else:
            self.r_traj = cp.ones_like(self.boundary0, dtype=prec)
            self.r_traj[self.boundary0] = 0

        self.prec = prec
        self.len = len(self.boundary0)
        self.future_boundary = boundaries.FutureBoundary(self.r_traj, self.b_traj, self.t_traj, self.i_traj)
        self.past_boundary = boundaries.PastBoundary(self.r_traj, self.b_traj, self.t_traj, self.i_traj)
        self.metrics_history = {}
        self.iter = 0
        self.p2i0 = None
        self.w_traj = None

    def fit_transform(self, comp_y,
                      envelope=envelope_sigmoid, gamma=0, basis_functions=basis_poly_ry, ny=6,
                      max_iter=100000, min_delta_x=None,
                      print_step=1000, metrics_print=None, stable=False,
                      history_delta_t=None, history_type=None, history_shift_type=None,
                      save_min_delta_zt=True, train_mask=None, cupy_type=1, mfpt_max=None):
        self.r_traj_old = self.r_traj
        self.time_start = time.time()
        if metrics_print is None:
            metrics_print = ('iter', 'imfpt', 'max_sd_zt', 'max_grad_zt', 'delta_x', 'time_elapsed')
        min_delta_zt = 1000
        _envelope = (1 - self.b_traj)
        if not callable(gamma):
            _gamma = gamma
        for self.iter in range(max_iter + 1):

            # compute next CV y, and cast it to the required accuracy
            y = cp.asarray(comp_y(), dtype=self.prec)

            # compute envelope, modulating the basis functions
            if self.iter % 10 == 0 and callable(envelope):
                _envelope = envelope(self.r_traj, self.iter, max_iter) * (1 - self.b_traj)

            # compute the basis functions
            if history_delta_t is None:
                y1, y2 = self.r_traj, y
            else:
                y1, y2 = self.history_select_y1y2(y, history_delta_t, history_type, history_shift_type)
            fk = basis_functions(y1, y2, ny, _envelope)

            # compute the gamma parameter
            if callable(gamma):
                _gamma = gamma(self.iter, max_iter)

            # compute next update of the RC
            if cupy_type==1:
                self.r_traj = nonparametrics.npnet(self.r_traj, fk, self.t_traj, self.i_traj, _gamma, )
            if cupy_type==2:
                self.r_traj = nonparametrics.npnet(self.r_traj[:-1], fk[:,:-1], self.t_traj[:-1], self.i_traj[:-1], _gamma, )
                self.r_traj = cp.concatenate([self.r_traj, cp.array([0])])

            if mfpt_max is not None:
                self.r_traj=cp.clip(self.r_traj,0,mfpt_max)

            # compute and print various metrics
            if self.iter % print_step == 0:
                self.compute_metrics(metrics_print)
                self.print_metrics(metrics_print)
                self.r_traj_old = self.r_traj
                if self.iter > 0:
                    if save_min_delta_zt:
                        if self.metrics_history['max_sd_zt'][-1] < min_delta_zt:
                            min_delta_t = self.metrics_history['max_sd_zt'][-1]
                            self.r_traj_min_sd_zt = self.r_traj
                    if min_delta_x is not None and self.metrics_history['delta_x'][-1] < min_delta_x:
                        break
                    
    def plots_feps(self, r_traj=None, delta_t_sim=1, ldt=None, reweight=False, xlabel='$\\tau$', zt_force0=False):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if r_traj is None:
            r_traj = self.r_traj
        if reweight:
            if self.w_traj is None:
                self.comp_eq_weights()
            plots.plot_fep(ax1, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, w_traj=self.w_traj, xlabel=xlabel)
            plots.plot_fep(ax2, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, w_traj=self.w_traj,
                           natural=True, dt_sim=delta_t_sim, xlabel=xlabel)
            plots.plot_zc1(ax3, r_traj, self.b_traj, self.i_traj, self.future_boundary, self.past_boundary, ldt=ldt,
                           w_traj=self.w_traj, xlabel=xlabel)
        else:
            plots.plot_fep(ax1, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, xlabel=xlabel)
            plots.plot_fep(ax2, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, natural=True, dt_sim=delta_t_sim, xlabel=xlabel)
            plots.plot_zt(ax3, r_traj, self.b_traj, self.t_traj, self.i_traj, self.future_boundary, self.past_boundary, ldt=ldt, xlabel=xlabel, force0=zt_force0)
        fig.tight_layout()
        plt.show()

    def plots_obs_pred(self, r_traj=None, log_scale=False, log_scale_tmin=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if r_traj is None:
            r_traj = self.r_traj
        plots.plot_obs_pred_t(ax1, r_traj, self.future_boundary, ax2=ax2, log_scale=log_scale, log_scale_tmin=log_scale_tmin)
        #r_traj=r_traj/cp.max(r_traj)
        #plots.plot_roc_curve(ax3, r_traj, self.future_boundary, log_scale=log_scale)
        fig.tight_layout()
        plt.show()

class Committor(CommittorNE):
    def fit_transform(self, comp_y,
                      envelope=envelope_sigmoid, gamma=0, basis_functions=basis_poly_ry, ny=6,
                      max_iter=100000, min_delta_x=None, min_delta_r2=None,
                      print_step=1000, metrics_print=None, stable=False,
                      save_min_delta_zq=True, train_mask=None, delta2_r2_min=10):
        self.r_traj_old = self.r_traj
        self.time_start = time.time()
        if metrics_print is None:
            metrics_print = ('iter', 'cross_entropy', 'mse', 'max_sd_zq', 'max_grad_zq', 'delta_r2', 'auc', 'delta_x', 'time_elapsed')
        self.min_delta_zq = 10000
        _envelope = (1 - self.b_traj)
        if not callable(gamma):
            _gamma = gamma
        if self.i_traj is None:
            It=cp.ones_like(self.r_traj[:-1])
        else:
            It=cp.asarray(self.i_traj[1:] == self.i_traj[:-1],dtype=self.r_traj.dtype)
        delta_r2=cp.sum(It*cp.square(self.r_traj[1:] - self.r_traj[:-1]))
        
        for self.iter in range(max_iter + 1):

            # compute next CV y, and cast it to the required accuracy
            y = cp.asarray(comp_y(), dtype=self.prec)

            # compute envelope, modulating the basis functions
            if self.iter % 10 == 0 and callable(envelope):
                _envelope = envelope(self.r_traj, self.iter, max_iter) * (1 - self.b_traj)

            # compute the basis functions
            fk = basis_functions(self.r_traj, y, ny, _envelope)

            # compute the gamma parameter
            if callable(gamma):
                _gamma = gamma(self.iter, max_iter)

            # compute next update of the RC
            r_traj = nonparametrics.npq(self.r_traj, fk, self.i_traj)
            delta_r2_new = cp.sum(It*cp.square(r_traj[1:] - r_traj[:-1]))
            if delta_r2_new-delta_r2<delta2_r2_min:
                self.r_traj=r_traj
                delta_r2=delta_r2_new
            

            # compute and print various metrics
            if self.iter % print_step == 0:
                self.compute_metrics(metrics_print)
                self.print_metrics(metrics_print)
                self.r_traj_old = self.r_traj
                if self.iter > 0:
                    if save_min_delta_zq:
                        if self.metrics_history['max_sd_zq'][-1] < self.min_delta_zq:
                            self.min_delta_zq = self.metrics_history['max_sd_zq'][-1]
                            self.r_traj_min_sd_zq = self.r_traj
                    if min_delta_x is not None and self.metrics_history['delta_x'][-1] < min_delta_x:
                        break
                    if min_delta_r2 is not None and self.metrics_history['delta_r2'][-1] < min_delta_r2:
                        break

                        
    def plots_feps(self, r_traj=None, delta_t_sim=1, ldt=None, dtmin=1):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if r_traj is None:
            r_traj = self.r_traj
        plots.plot_fep(ax1, r_traj, i_traj=self.i_traj, t_traj=self.t_traj)
        plots.plot_fep(ax2, r_traj, i_traj=self.i_traj, t_traj=self.t_traj, natural=True, dt_sim=delta_t_sim)
        plots.plot_zq(ax3, r_traj, self.b_traj, self.i_traj, self.future_boundary, self.past_boundary, ldt=ldt)
        fig.tight_layout()
        plt.show()
