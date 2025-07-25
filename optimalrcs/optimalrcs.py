import numpy as np
import tensorflow as tf
from . import boundaries, metrics, nonparametrics, plots
import time
import matplotlib.pyplot as plt


envelope_scale = 0.01


def envelope_sigmoid(r, iter, max_iter):
    """
    Generate a sigmoid envelope function to modulate basis functions during RC optimization.

    This function selects a random reference point `r0` from the current reaction coordinate
    time-series `r` and constructs a sigmoid envelope centered at `r0`. The envelope is used
    to focus optimization on specific regions of the RC, typically around minima. The 
    direction of the sigmoid (increasing or decreasing) is chosen randomly.

    Parameters
    ----------
    r : tf.Tensor
        Current reaction coordinate time-series, shape (N,).
    iter : int
        Current iteration number of the optimization loop.
    max_iter : int
        Maximum number of iterations allowed in the optimization.

    Returns
    -------
    tf.Tensor
        A tensor of shape (N,) representing the envelope values applied to each point in `r`.

    Notes
    -----
    The envelope is defined as:
        sigmoid(±(r - r0) / (scale * Δr))
    where `r0` is a randomly selected point from `r`, and Δr is the range of `|r|`.
    The sign is chosen randomly to allow focusing on either side of the transition region.
    """
    r0 = r[np.random.randint(r.shape[0])]
    delta_r = tf.math.reduce_max(tf.math.abs(r)) - tf.math.reduce_min(tf.math.abs(r))
    if delta_r < 1e-5:
        delta_r = 1e-5
    if np.random.random() < 0.5:
        return tf.math.sigmoid((r - r0) / envelope_scale / delta_r)
    return tf.math.sigmoid(-(r - r0) / envelope_scale / delta_r)


@tf.function
def basis_poly_ry(r, y, n, fenv=None):
    """
    Construct a set of polynomial basis functions in variables r and y.

    This function generates a complete basis of monomials up to total degree `n`
    in the variables `r` (reaction coordinate time-series) and `y` (a collective variable),
    optionally modulated by an envelope function `fenv`. The basis is used to construct
    variations of the RC during optimization.

    Parameters
    ----------
    r : tf.Tensor
        The current reaction coordinate time-series, shape (N,).
    y : tf.Tensor
        A collective variable time-series, shape (N,), used to improve the RC.
    n : int
        Maximum total degree of the polynomial basis.
    fenv : tf.Tensor or None, optional
        Envelope function of shape (N,) used to localize optimization in RC space.
        If None, a uniform envelope is used.

    Returns
    -------
    tf.Tensor
        A tensor of shape (M, N), where M = (n+1)(n+2)/2, containing the evaluated
        basis functions at each time point. Each row corresponds to a monomial term
        of the form r^i * y^j with i + j ≤ n.
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
        """
        Initialize the CommittorNE class for non-equilibrium committor estimation.

        This constructor sets up the initial state for nonparametric optimization of a
        reaction coordinate (RC) that approximates the committor function in systems
        with potentially irregular or incomplete trajectory data.

        Parameters
        ----------
        boundary0 : array_like
            Boolean array indicating frames belonging to boundary state A (committor = 0).
        boundary1 : array_like
            Boolean array indicating frames belonging to boundary state B (committor = 1).
        i_traj : array_like of int, optional
            Trajectory index for each frame, used to distinguish between multiple trajectories.
        t_traj : array_like of float, optional
            Time associated with each frame. Required for time-dependent metrics and history-based optimization.
        seed_r : array_like, optional
            Initial guess for the reaction coordinate time-series. If None, initialized to 0.5 in the interior and 0/1 at boundaries.
        prec : dtype, optional
            Precision used for internal computations (default: np.float64).

        Notes
        -----
        The initial RC is set to 0.5 for all non-boundary frames, and to 0 or 1 for frames
        in boundary0 and boundary1, respectively. Boundary masks and trajectory metadata
        are used to construct helper objects for forward and backward analysis.
        """
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
                      print_step=1000, metrics_print=None,
                      history_delta_t=None, history_type=None, history_shift_type=None,
                      save_min_delta_zq=True, train_mask=None, delta2_r2_max_change_allowed=1e3):
        """
        Perform nonparametric optimization of the reaction coordinate (RC) to approximate the committor.

        This method iteratively updates the RC time-series using a basis expansion in collective variables (CVs),
        optionally incorporating trajectory history. The optimization minimizes non-Markovian effects and
        improves kinetic consistency, aiming to converge toward the committor function.

        Parameters
        ----------
        comp_y : callable
            Function that returns a randomly selected collective variable (CV) time-series y(t).
        envelope : callable, optional
            Envelope function used to localize optimization in RC space (default: `envelope_sigmoid`).
        gamma : float or callable, optional
            Regularization strength or a function returning gamma per iteration (default: 0).
        basis_functions : callable, optional
            Function to generate basis functions from r(t) and y(t) (default: `basis_poly_ry`).
        ny : int, optional
            Maximum polynomial degree for basis functions (default: 6).
        max_iter : int, optional
            Maximum number of optimization iterations (default: 100000).
        min_delta_x : float, optional
            Stop criterion: minimum change in RC between print_step iterations.
        min_delta_r2 : float, optional
            Stop criterion: minimum value of delta_r^2 functional.
        print_step : int, optional
            Frequency of metric printing and logging (default: 1000).
        metrics_print : tuple of str, optional
            Names of metrics to compute and print during optimization.
        history_delta_t : list of int, optional
            List of time delays to sample from when incorporating history.
        history_type : str or list of str, optional
            Type(s) of history-based variation to apply (e.g., 'y(t-d),r(t-d)').
        history_shift_type : str, optional
            Strategy for handling history across trajectory boundaries (e.g., 'r(t0)', 'r(t)', or '0').
        save_min_delta_zq : bool, optional
            If True, save the RC with the smallest observed Z_q deviation (default: True).
        train_mask : array_like, optional
            Boolean mask indicating which frames to include in training. # not implemented
        delta2_r2_max_change_allowed : float, optional
            Maximum change for accepting a new RC update based on change in delta_r^2 (default: 1e3).

        Returns
        -------
        None
            Updates `self.r_traj` in place and logs metrics in `self.metrics_history`.

        Notes
        -----
        The optimization proceeds by constructing variations of the RC using basis functions
        of the form f(r(t), y(t)) or f(r(t-d), y(t-d)), depending on the history settings.
        Metrics such as cross-entropy, AUC, and Z_q are tracked to assess convergence and quality.
        """
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
            r_traj = nonparametrics.npneq(self.r_traj, fk, self.i_traj, _gamma)
            delta_r2_new = tf.reduce_sum(It*tf.square(r_traj[1:] - r_traj[:-1]))
            if delta_r2_new-delta_r2<delta2_r2_max_change_allowed:
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
        """
        Plot free energy profiles and the validation criterion Z_q for the current RC.

        This method generates three side-by-side plots:
        1. Free energy profile (FEP) as a function of the RC.
        2. FEP along the natural coordinate where the diffusion coefficient is constant.
        3. Validation criterion Z_q across lag times, used to assess RC optimality.

        Parameters
        ----------
        r_traj : array_like, optional
            Reaction coordinate time-series to plot. If None, uses `self.r_traj`.
        delta_t_sim : int, optional
            Simulation time step used for computing the natural coordinate (default: 1).
        ldt : array_like, optional
            List of lag times used to compute Z_q profiles.
        reweight : bool, optional
            If True, apply equilibrium reweighting using `self.w_traj`.
        dtmin : int, optional
            Minimum lag time for Z_q computation (default: 1).
        zq_force0 : bool, optional
            If True, force Z_q to be zero at the boundaries, useful for visual clarity or specific validation setups.

        Returns
        -------
        None
            Displays the plots using matplotlib.

        Notes
        -----
        The natural coordinate transformation rescales the RC such that the diffusion
        coefficient becomes constant, improving interpretability of the FEP.
        The Z_q plot is used to validate whether the RC approximates the committor,
        with constancy across lag times indicating optimality.
        """
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
        """
        Plot predicted vs. observed committor values and ROC curve for RC validation.

        This method visualizes the quality of the current reaction coordinate (RC)
        by comparing predicted committor values against observed outcomes and
        computing the receiver operating characteristic (ROC) curve.

        Parameters
        ----------
        r_traj : array_like, optional
            Reaction coordinate time-series to evaluate. If None, uses `self.r_traj`.
        log_scale : bool, optional
            If True, apply logarithmic scaling to the probability axis in the predicted vs. observed plot.
        log_scale_pmin : float, optional
            Minimum probability threshold for log scaling. Values below this are clipped.

        Returns
        -------
        None
            Displays the plots using matplotlib.

        Notes
        -----
        
        The three subplots are:
        1. Predicted vs. observed committor values, binned along the RC.
        2. Histogram showing the number of trajectory segments from each RC bin that reach either boundary.
        3. ROC curve evaluating the discriminative power of the RC in separating boundary states.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        if r_traj is None:
            r_traj = self.r_traj
        plots.plot_obs_pred_q(ax1, r_traj, self.future_boundary, ax2=ax2, log_scale=log_scale, log_scale_pmin=log_scale_pmin)
        plots.plot_roc_curve(ax3, r_traj, self.future_boundary, log_scale=log_scale)
        fig.tight_layout()
        plt.show()

    def comp_eq_weights(self, ny=6, max_iter=1000, min_delta_x=1e-5, print_step=10, verbose=1):
        """
        Compute equilibrium reweighting factors for non-equilibrium trajectory data.

        This method estimates weights for each frame in the trajectory such that the
        weighted ensemble approximates equilibrium sampling. The weights are iteratively
        updated using a nonparametric basis expansion in the RC and optimized to minimize
        changes between iterations.

        Parameters
        ----------
        ny : int, optional
            Degree of the polynomial basis used for weight updates (default: 6).
        max_iter : int, optional
            Maximum number of iterations for weight optimization (default: 1000).
        min_delta_x : float, optional
            Stop criterion: minimum change in weights between iterations (default: 1e-5).
        print_step : int, optional
            Frequency of logging progress during optimization (default: 10).
        verbose : int, optional
            Verbosity level. If 0, no output; 1 prints final status; >1 prints progress at each step.

        Returns
        -------
        None
            Updates `self.w_traj` in place with the computed equilibrium weights.

        Notes
        -----
        The weights are computed using a nonparametric update scheme (`npnew`) based on
        polynomial basis functions of the current weights and RC. These weights are used
        in downstream analysis such as reweighted free energy profiles and validation plots.
        """
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
        """
        Initialize the MFPTNE class for non-equilibrium mean first passage time (MFPT) optimization.

        This class estimates a reaction coordinate (RC) that approximates the mean first passage time
        (MFPT) to a specified absorbing boundary (state A). It is particularly useful when only one
        boundary condition is defined, and is applicable to low-dimensional or irregular datasets.

        Parameters
        ----------
        boundary0 : array_like
            Boolean array indicating frames belonging to the absorbing boundary (MFPT = 0).
        i_traj : array_like of int, optional
            Trajectory index for each frame, used to distinguish between multiple trajectories.
        t_traj : array_like of float, optional
            Time associated with each frame. If None, defaults to a uniform time grid (0,1,2,...).
        seed_r : array_like, optional
            Initial guess for the reaction coordinate time-series. If None, initialized to 1.0 in the interior and 0 at the boundary.
        prec : dtype, optional
            Precision used for internal computations (default: np.float64).

        Attributes
        ----------
        r_traj : ndarray
            Current estimate of the MFPT-based reaction coordinate.
        b_traj : ndarray
            Boolean array indicating the absorbing boundary.
        future_boundary : boundaries.FutureBoundary
            Helper object for computing future-based MFPT metrics and validation.
        past_boundary : boundaries.PastBoundary
            Helper object for computing past-based metrics and validation.
        metrics_history : dict
            Dictionary storing the history of computed metrics during optimization.
        iter : int
            Current iteration count.
        p2i0 : ndarray or None
            Cached pointer to the first frame of each trajectory (used in history-based updates).
        w_traj : ndarray or None
            Optional weights for reweighting non-equilibrium trajectories to equilibrium.
        """
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
                      print_step=1000, metrics_print=None,
                      history_delta_t=None, history_type=None, history_shift_type=None,
                      save_min_delta_zt=True, train_mask=None):
        """
        Perform nonparametric optimization of the reaction coordinate to approximate the MFPT.

        This method iteratively updates the reaction coordinate (RC) to approximate the
        mean first passage time (MFPT) to a specified absorbing boundary. The optimization
        uses a basis expansion in collective variables (CVs), optionally incorporating
        trajectory history, and tracks convergence using MFPT-specific metrics.

        Parameters
        ----------
        comp_y : callable
            Function that returns a randomly selected collective variable (CV) time-series y(t).
        envelope : callable, optional
            Envelope function used to localize optimization in RC space (default: `envelope_sigmoid`).
        gamma : float or callable, optional
            Regularization strength or a function returning gamma per iteration (default: 0).
        basis_functions : callable, optional
            Function to generate basis functions from r(t) and y(t) (default: `basis_poly_ry`).
        ny : int, optional
            Maximum polynomial degree for basis functions (default: 6).
        max_iter : int, optional
            Maximum number of optimization iterations (default: 100000).
        min_delta_x : float, optional
            Stop criterion: minimum change in RC between print_step iterations.
        print_step : int, optional
            Frequency of metric printing and logging (default: 1000).
        metrics_print : tuple of str, optional
            Names of metrics to compute and print during optimization.
        history_delta_t : list of int, optional
            List of time delays to sample from when incorporating history.
        history_type : str or list of str, optional
            Type(s) of history-based variation to apply (e.g., 'y(t-d),r(t-d)').
        history_shift_type : str, optional
            Strategy for handling history across trajectory boundaries (e.g., 'r(t0)', 'r(t)', or '0').
        save_min_delta_zt : bool, optional
            If True, save the RC with the smallest observed Z_tau deviation (default: True).
        train_mask : array_like, optional
            Boolean mask indicating which frames to include in training. not implemented!

        Returns
        -------
        None
            Updates `self.r_traj` in place and logs metrics in `self.metrics_history`.

        Notes
        -----
        
        The optimization proceeds by constructing variations of the RC using basis functions
        of the form f(r(t), y(t)) or f(r(t-d), y(t-d)), depending on the history settings. The
        Z_tau validation criterion is used to assess convergence and optimality.
        """
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
        """
        Plot free energy profiles and the validation criterion Z_tau for the MFPT-based RC.

        This method generates three side-by-side plots:
        1. Free energy profile (FEP) as a function of the MFPT-based reaction coordinate.
        2. FEP along the natural coordinate where the diffusion coefficient is constant.
        3. Validation criterion Z_tau across lag times, used to assess RC optimality.

        Parameters
        ----------
        r_traj : array_like, optional
            Reaction coordinate time-series to plot. If None, uses `self.r_traj`.
        delta_t_sim : int, optional
            Simulation time step used for computing the natural coordinate (default: 1).
        ldt : array_like, optional
            List of lag times used to compute Z_tau profiles.
        reweight : bool, optional
            If True, apply equilibrium reweighting using `self.w_traj`.
        xlabel : str, optional
            Label for the x-axis in all plots (default: '$\\tau$').
        force0 : bool, optional
            If True, force Z_tau to be zero at the start (default: False).

        Returns
        -------
        None
            Displays the plots using matplotlib.

        Notes
        -----
        The natural coordinate transformation rescales the RC such that the diffusion
        coefficient becomes constant, improving interpretability of the FEP.
        The Z_tau plot is used to validate whether the RC closely approximates the MFPT,
        with constancy across lag times indicating optimality.
        """
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
        """
        Plot predicted vs. observed MFPT values and statistics on boundary-reaching events.

        This method visualizes the quality of the MFPT-based reaction coordinate (RC)
        by comparing predicted MFPT values against observed outcomes. It also includes
        a histogram showing how many trajectory segments from each RC bin reach the absorbing boundary.

        Parameters
        ----------
        r_traj : array_like, optional
            Reaction coordinate time-series to evaluate. If None, uses `self.r_traj`.
        log_scale : bool, optional
            If True, apply logarithmic scaling to the MFPT axis in the predicted vs. observed plot.
        log_scale_tmin : float, optional
            Minimum MFPT threshold for log scaling. Values below this are clipped.

        Returns
        -------
        None
            Displays the plots using matplotlib.

        Notes
        -----
        The three subplots are:
        1. Predicted vs. observed MFPT values, binned along the RC.
        2. Histogram showing the number of trajectory segments from each RC bin that reach the absorbing boundary.
        3. (Optional) ROC curve — currently disabled in this implementation.

        The middle plot is particularly useful for assessing sampling quality and identifying
        underrepresented regions in the RC space.
        """
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

        