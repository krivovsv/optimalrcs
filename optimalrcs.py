import numpy as np
import tensorflow as tf
import boundaries
import metrics
import nonparametrics
import time
import plots
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
        self.metric2function = {'delta_r2': self.metric_delta_r2, 'max_delta_zq': self.metric_max_delta_zq,
                                'mse': self.metric_mse, 'mse_eq': self.metric_mse_eq, 'iter': self.metric_iter,
                                'cross_entropy': self.metric_cross_entropy, 'time_elapsed': self.metric_time_elapsed,
                                'delta_x': self.metric_delta_x, 'auc': self.metric_auc,
                                'max_sd_zq': self.metric_max_sd_zq}
        self.metrics_short_name = {'delta_r2': 'dr2', 'max_delta_zq': 'maxdzq', 'mse': 'mse', 'mse_eq': 'mseeq',
                                   'cross_entropy': 'xent', 'time_elapsed': 'time', 'delta_x': 'dx',
                                   'iter': '#', 'auc': 'auc', 'max_sd_zq': 'sdzq'}
        self.iter = 0
        self.p2i0 = None
        self.w_traj = None

    def metric_delta_r2(self):
        return metrics.delta_r2(self.r_traj, self.b_traj, self.i_traj, self.future_boundary, self.past_boundary, dt=1)

    def metric_max_delta_zq(self):
        return metrics.comp_max_delta_zq(self.r_traj, self.b_traj, self.i_traj, self.future_boundary,
                                         self.past_boundary)[0]

    def metric_max_sd_zq(self):
        return metrics.comp_max_delta_zq(self.r_traj, self.b_traj, self.i_traj, self.future_boundary,
                                         self.past_boundary)[2]

    def metric_mse(self):
        return metrics.mse(self.r_traj, self.b_traj, self.i_traj, self.future_boundary)

    def metric_mse_eq(self):
        return metrics.mse_eq(self.r_traj, self.b_traj, self.i_traj, self.future_boundary, self.past_boundary)

    def metric_cross_entropy(self):
        return metrics.cross_entropy(self.r_traj, self.b_traj, self.i_traj, self.future_boundary)

    def metric_auc(self):
        return metrics.auc(self.r_traj, future_boundary=self.future_boundary)

    def metric_low_bound_delta_r2_eq(self):
        return metrics.low_bound_delta_r2_eq(self.r_traj, self.b_traj, self.i_traj, self.future_boundary)

    def metric_time_elapsed(self):
        return time.time()-self.time_start

    def metric_delta_x(self):
        return metrics.delta_x(self.r_traj, self.r_traj_old)

    def metric_iter(self):
        return self.iter

    def print_metrics(self, metrics_print):
        s = ''
        for metric in metrics_print:
            s += '%s=%g, ' % (self.metrics_short_name[metric], self.metrics_history[metric][-1])
        print(s[:-2])

    def compute_metrics(self, metrics_print):
        for metric in metrics_print:
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            self.metrics_history[metric].append(self.metric2function[metric]())

    def fit_transform(self, comp_y,
                      envelope=envelope_sigmoid, gamma=0, basis_functions=basis_poly_ry, ny=6,
                      max_iter=100000, min_delta_x=None, min_delta_r2=None,
                      print_step=1000, metrics_print=None, stable=False,
                      history_delta_t=None, history_type=None, history_shift_type=None,
                      save_min_delta_zq=True, train_mask=None):
        self.r_traj_old = self.r_traj
        self.time_start = time.time()
        if metrics_print is None:
            metrics_print = ('iter', 'cross_entropy', 'mse', 'max_sd_zq', 'delta_r2', 'auc', 'delta_x', 'time_elapsed')
        min_delta_zq = 1000
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
                fk = basis_functions(self.r_traj, y, ny, _envelope)  # compute basis functions
            else:
                d = np.random.choice(history_delta_t)
                if d > 0:
                    if history_type is None:
                        history_type = 'y(t-d),r(t-d)'
                    if history_type == 'y(t-d),r(t-d)' and history_shift_type is None:
                        if self.i_traj is not None:
                            it = tf.concat([tf.zeros([d], dtype=tf.bool), self.i_traj[d:] == self.i_traj[:-d]], 0)
                        else:
                            it = tf.concat([tf.zeros([d], dtype=tf.bool), tf.ones([self.len-d], dtype=tf.bool)], 0)
                        y1 = tf.where(it, tf.roll(y, d, 0), 0)
                        y2 = tf.where(it, tf.roll(self.r_traj, d, 0), 0)
                    elif history_type == 'y(t-d),y(t)' and history_shift_type is None:
                        if self.i_traj is not None:
                            it = tf.concat([tf.zeros([d], dtype=tf.bool), self.i_traj[d:] == self.i_traj[:-d]], 0)
                        else:
                            it = tf.concat([tf.zeros([d], dtype=tf.bool), tf.ones([self.len-d], dtype=tf.bool)], 0)
                        y1 = tf.where(it, tf.roll(y, d, 0), 0)
                        y2 = y
                    else:
                        y1, y2 = self._history_select_y(y, d, history_type, history_shift_type)
                    fk = basis_functions(y1, y2, ny, _envelope)
                else:
                    fk = basis_functions(self.r_traj, y, ny, _envelope)

            # compute the gamma parameter
            if callable(gamma):
                _gamma = tf.constant(gamma(self.iter, max_iter), dtype=self.prec)

            # compute next update of the RC
            self.r_traj = nonparametrics.npneq(self.r_traj, fk, self.i_traj, _gamma, stable)

            # compute and print various metrics
            if self.iter % print_step == 0:
                self.compute_metrics(metrics_print)
                self.print_metrics(metrics_print)
                self.r_traj_old = self.r_traj
                if self.iter > 0:
                    if save_min_delta_zq:
                        if self.metrics_history['max_sd_zq'][-1] < min_delta_zq:
                            min_delta_zq = self.metrics_history['max_sd_zq'][-1]
                            self.r_traj_min_sd_zq = self.r_traj
                    if min_delta_x is not None and self.metrics_history['delta_x'][-1] < min_delta_x:
                        break
                    if min_delta_r2 is not None and self.metrics_history['delta_r2'][-1] < min_delta_r2:
                        break

    def _history_select_y(self, y, d, history_type, history_shift_type):
        if history_shift_type is None:
            history_shift_type = '0'
        if history_type is None:
            history_type = 'y(t-d),r(t-d)'
        if history_shift_type == 'r(t0)' and self.p2i0 is None:
            #self.p2i0 = np.zeros_like(self.i_traj)  # compute pointer to the first frame in the trajectoriy
            #ilast = -1
            #for i in range(len(self.i_traj)):
            #    if self.i_traj[i] != ilast:
            #        p = i
            #        ilast = self.i_traj[i]
            #    self.p2i0[i] = p
            i0=np.where(np.roll(self.i_traj,1)!=self.i_traj)[0]
            self.p2i0=i0[self.i_traj]

        def shift_y(d, i_traj, y, shift_type, p2i0):
            if shift_type == 'r(t-d)':  # do nothing, take values from previous trajectories r(t-d)
                return tf.roll(y, d, 0)
            elif shift_type == 'r(t)':  # take r(t)
                return tf.where(tf.roll(i_traj, d, 0) == i_traj, tf.roll(y, d, 0), y)
            elif shift_type == 'r(t0)':  # take r(t0)
                if tf.is_tensor(y):
                    yn = y.numpy()
                else:
                    yn = y
                return tf.where(tf.roll(i_traj, d, 0) == i_traj, tf.roll(y, d, 0), yn[p2i0])
            else:  # take 0
                return tf.where(tf.roll(i_traj, d, 0) == i_traj, tf.roll(y, d, 0), 0)

        if len(history_type) > 1:
            d1, d2 = history_type[np.random.randint(len(history_type))]
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
        if d2 == 'r(t)':
            y2 = self.r_traj
        if d2 == 'y(t)':
            y2 = y
        if d2 == 'r(t-d)':
            y2 = shift_y(d, self.i_traj, self.r_traj, history_shift_type, self.p2i0)
        if d2 == 'y(t-d)':
            y2 = shift_y(d, self.i_traj, y, history_shift_type, self.p2i0)
        return y1, y2

    def plots_feps(self, r_traj=None, delta_t_sim=1, ldt=None, reweight=False):
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
                           w_traj=self.w_traj)
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
        plots.plot_obs_pred(ax1, r_traj, self.future_boundary, ax2=ax2, log_scale=log_scale, log_scale_pmin=log_scale_pmin)
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

