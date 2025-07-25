import numpy as np
import tensorflow as tf
from . import boundaries, metrics, nonparametrics, plots
import time
import matplotlib.pyplot as plt


envelope_scale = 0.01


def envelope_sigmoid(r, iter, max_iter):
    r0 = r[np.random.randint(r.shape[0])]
    delta_r = tf.math.reduce_max(tf.math.abs(r)) - tf.math.reduce_min(tf.math.abs(r))
    if delta_r < 1e-5:
        delta_r = 1e-5
    if np.random.random() < 0.5:
        return tf.math.sigmoid((r - r0) / envelope_scale / delta_r)
    return tf.math.sigmoid(-(r - r0) / envelope_scale / delta_r)


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


class CommittorNE:
    def __init__(self, boundary0, boundary1, i_traj=None, t_traj=None, seed_r=None, prec=np.float64):
        self.boundary0 = boundary0
        self.boundary1 = boundary1
        self.b_traj = np.array(boundary0 | boundary1, prec)
        self.i_traj = i_traj
        self.t_traj = t_traj
        if seed_r is not None:
            self.r_traj = np.array(seed_r, prec)
        else:
            self.r_traj = np.ones_like(self.boundary0, prec) / 2
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
                      save_min_delta_zq=True, train_mask=None, delta2_r2_min=1e3):
        self.r_traj_old = self.r_traj
        self.time_start = time.time()
        if metrics_print is None:
            metrics_print = ('iter', 'cross_entropy', 'mse', 'max_sd_zq', 'max_grad_zq', 'delta_r2', 'auc', 'delta_x', 'time_elapsed')
        self.min_delta_zq = 10000
        _envelope = (1 - self.b_traj)
        if not callable(gamma):
            _gamma = tf.constant(gamma, dtype=self.prec)
        if self.i_traj is None:
            It=tf.ones_like(self.r_traj[:-1])
        else:
            It=tf.cast(self.i_traj[1:] == self.i_traj[:-1],self.r_traj.dtype)
        delta_r2=tf.reduce_sum(It*tf.square(self.r_traj[1:] - self.r_traj[:-1]))
        
        for self.iter in range(max_iter + 1):

           # compute the basis functions
            if history_delta_t is None:
                delta_t = 0
            else:
                delta_t = np.random.choice(history_delta_t)
            if delta_t == 0:
                y1, y2 = self.r_traj, tf.cast(comp_y(), self.prec)
            else:
                y = tf.cast(comp_y(), self.prec)
                y1, y2 = self.history_select_y1y2(y, delta_t, history_type, history_shift_type)

            
            # compute envelope, modulating the basis functions
            if self.iter % 10 == 0 and callable(envelope):
                _envelope = envelope(self.r_traj, self.iter, max_iter) * (1 - self.b_traj)


            fk = basis_functions(y1, y2, ny, _envelope)

            # compute the gamma parameter
            if callable(gamma):
                _gamma = tf.constant(gamma(self.iter, max_iter), dtype=self.prec)

            # compute next update of the RC
            r_traj = nonparametrics.npneq(self.r_traj, fk, self.i_traj, _gamma, stable)
            delta_r2_new = tf.reduce_sum(It*tf.square(r_traj[1:] - r_traj[:-1]))
            if delta_r2_new-delta_r2<delta2_r2_min:
                #print (delta_r2_new.numpy(),end=' ')
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

    def history_select_y1y2(self, y, d, history_type, history_shift_type):
        if d > 0:
            if history_type is None:
                history_type = 'y(t-d),r(t-d)'
            if history_type == 'y(t-d),r(t-d)' and history_shift_type is None:
                if self.i_traj is not None:
                    it = tf.concat([tf.zeros([d], dtype=tf.bool), self.i_traj[d:] == self.i_traj[:-d]], 0)
                else:
                    it = tf.concat([tf.zeros([d], dtype=tf.bool), tf.ones([self.len - d], dtype=tf.bool)], 0)
                y1 = tf.where(it, tf.roll(y, d, 0), 0)
                y2 = tf.where(it, tf.roll(self.r_traj, d, 0), 0)
            elif history_type == 'y(t-d),y(t)' and history_shift_type is None:
                if self.i_traj is not None:
                    it = tf.concat([tf.zeros([d], dtype=tf.bool), self.i_traj[d:] == self.i_traj[:-d]], 0)
                else:
                    it = tf.concat([tf.zeros([d], dtype=tf.bool), tf.ones([self.len - d], dtype=tf.bool)], 0)
                y1 = tf.where(it, tf.roll(y, d, 0), 0)
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
            changes = np.diff(self.i_traj, prepend=self.i_traj[0]-1) != 0
            first_indices = np.where(changes)[0]
            self.p2i0 = np.repeat(first_indices, np.diff(np.append(first_indices, len(self.i_traj))))
                        # pointer to the first frame of trajectory defined by i_traj
            self.p2i0 = tf.convert_to_tensor(self.p2i0)

        def shift_y(d, i_traj, y, shift_type, p2i0): # which point to select when previous point at (t-d) belongs to other trajectory
            if shift_type == 'r(t-d)':  # do nothing, take previous values y(t-d) disregarding trajectory info i_traj
                return tf.roll(y, d, 0)
            elif shift_type == 'r(t)':  # take y(t) instead of y(t-d)
                return tf.where(tf.roll(i_traj, d, 0) == i_traj, tf.roll(y, d, 0), y)
            elif shift_type == 'r(t0)':  # take first frame of trajectory, y(t0) instead of y(t-d)
                return tf.where(tf.roll(i_traj, d, 0) == i_traj, tf.roll(y, d, 0), tf.gather(y,p2i0))
            else:  # take 0 instead of y(t-d)
                return tf.where(tf.roll(i_traj, d, 0) == i_traj, tf.roll(y, d, 0), 0)

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
            y1 = tf.where(y1 > 0, tf.math.log(self.t_traj-y1+1e-5), 0)
        if d1 == 'dt':
            y1 = shift_y(d, self.i_traj, self.t_traj, history_shift_type, self.p2i0)
            y1 = tf.where(y1 > 0, self.t_traj-y1, 0)
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
            y2 = tf.where(y2 > 0,  tf.math.log(self.t_traj-y2+1e-5), 0)
        if d2 == 'dt':
            y2 = shift_y(d, self.i_traj, self.t_traj, history_shift_type, self.p2i0)
            y2 = tf.where(y2 > 0,  self.t_traj-y2, 0)
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

    def plots_feps(self, r_traj=None, delta_t_sim=1, ldt=None, reweight=False, dtmin=1):
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
            plots.plot_zq(ax3, r_traj, self.b_traj, self.i_traj, self.future_boundary, self.past_boundary, ldt=ldt)
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

    def comp_eq_weights(self, ny=6, max_iter=1000, min_delta_x=1e-5, print_step=10, verbose=1):
        self.w_traj = np.ones_like(self.r_traj, self.prec)

        start = time.time()
        wo = self.w_traj
        if self.i_traj is None:
            it=np.ones_like(self.r_traj[:-1])
        else:
            it=tf.cast(self.i_traj[1:] == self.i_traj[:-1], dtype=self.r_traj.dtype)
        for i in range(max_iter):
            self.w_traj = nonparametrics.npnew(self.w_traj, basis_poly_ry(self.w_traj, self.r_traj, ny), it)

            if i % print_step == 0:
                dx = self.w_traj - wo
                wo = self.w_traj
                dx = (tf.math.reduce_mean(dx**2)**0.5).numpy()
                max_w = tf.math.reduce_max(self.w_traj).numpy()
                min_w = tf.math.reduce_min(self.w_traj).numpy()
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
        self.b_traj = np.array(boundary0, prec)
        self.i_traj = i_traj
        if t_traj is not None:
            self.t_traj = np.array(t_traj, prec)
        else:
            self.t_traj = np.arange(len(self.b_traj), dtype=prec)
        if seed_r is not None:
            self.r_traj = np.array(seed_r, prec)
        else:
            self.r_traj = np.ones_like(self.boundary0, prec)
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
                      save_min_delta_zt=True, train_mask=None):
        self.r_traj_old = self.r_traj
        self.time_start = time.time()
        if metrics_print is None:
            metrics_print = ('iter', 'imfpt', 'max_sd_zt', 'max_grad_zt', 'delta_x', 'time_elapsed')
        min_delta_zt = 1000
        _envelope = (1 - self.b_traj)
        if not callable(gamma):
            _gamma = tf.constant(gamma, dtype=self.prec)
        for self.iter in range(max_iter + 1):

            # compute next CV y, and cast it to the required accuracy
            y = tf.cast(comp_y(), self.prec)

            # compute envelope, modulating the basis functions
            if self.iter % 10 == 0 and callable(envelope):
                _envelope = envelope(self.r_traj, self.iter, max_iter) * (1 - self.b_traj)

           # compute the basis functions
            if history_delta_t is None:
                delta_t = 0
            else:
                delta_t = np.random.choice(history_delta_t)
            if delta_t == 0:
                y1, y2 = self.r_traj, tf.cast(comp_y(), self.prec)
            else:
                y = tf.cast(comp_y(), self.prec)
                y1, y2 = self.history_select_y1y2(y, delta_t, history_type, history_shift_type)

            fk = basis_functions(y1, y2, ny, _envelope)

            # compute the gamma parameter
            if callable(gamma):
                _gamma = tf.constant(gamma(self.iter, max_iter), dtype=self.prec)

            # compute next update of the RC
            self.r_traj = nonparametrics.npnet(self.r_traj, fk, self.t_traj, self.i_traj, _gamma, )

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
                    
    def plots_feps(self, r_traj=None, delta_t_sim=1, ldt=None, reweight=False, xlabel='$\\tau$', force0=False):
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
            plots.plot_zt(ax3, r_traj, self.b_traj, self.t_traj, self.i_traj, self.future_boundary, self.past_boundary, ldt=ldt, xlabel=xlabel, force0=force0)
        fig.tight_layout()
        plt.show()

    def plots_obs_pred(self, r_traj=None, log_scale=False, log_scale_tmin=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if r_traj is None:
            r_traj = self.r_traj
        plots.plot_obs_pred_t(ax1, r_traj, self.future_boundary, ax2=ax2, log_scale=log_scale, log_scale_tmin=log_scale_tmin)
        #r_traj=r_traj/tf.reduce_max(r_traj)
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
            _gamma = tf.constant(gamma, dtype=self.prec)
        if self.i_traj is None:
            It=tf.ones_like(self.r_traj[:-1])
        else:
            It=tf.cast(self.i_traj[1:] == self.i_traj[:-1],self.r_traj.dtype)
        delta_r2=tf.reduce_sum(It*tf.square(self.r_traj[1:] - self.r_traj[:-1]))
        
        for self.iter in range(max_iter + 1):

            # compute next CV y, and cast it to the required accuracy
            y = tf.cast(comp_y(), self.prec)

            # compute envelope, modulating the basis functions
            if self.iter % 10 == 0 and callable(envelope):
                _envelope = envelope(self.r_traj, self.iter, max_iter) * (1 - self.b_traj)

            # compute the basis functions
            fk = basis_functions(self.r_traj, y, ny, _envelope)

            # compute the gamma parameter
            if callable(gamma):
                _gamma = tf.constant(gamma(self.iter, max_iter), dtype=self.prec)

            # compute next update of the RC
            r_traj = nonparametrics.npq(self.r_traj, fk, self.i_traj)
            delta_r2_new = tf.reduce_sum(It*tf.square(r_traj[1:] - r_traj[:-1]))
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

        