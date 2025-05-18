import numpy as np
import unittest
import optimalrcs.optimalrcs as optimalrcs
import optimalrcs.metrics as metrics


class TestMFPTNE(unittest.TestCase):

    def test_2f4k_history(self):
        f = open('data/2f4k.CArmsd', 'r')
        r_traj = []
        for line in f:
            r_traj.append(float(line.split()[-1]))
        r_traj = np.asarray(r_traj)
        f.close()
        mfpt = optimalrcs.MFPTNE(boundary0=r_traj < 1)
        print(metrics.low_bound_i_mfpt_eq(mfpt))

        def comp_y():
            return r_traj
        history_delta_t = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        history_delta_t = [0,] + [2**i for i in range(9)]
        mfpt.fit_transform(comp_y=comp_y, gamma=0.01, history_delta_t=history_delta_t, max_iter=2000, min_delta_x=1,
                        print_step=100)
        mfpt.plots_metrics()
        mfpt.plots_feps()
        mfpt.plots_obs_pred()
        
    def test_2f4k(self):
        f = open('data/2f4k.CArmsd', 'r')
        r_traj = []
        for line in f:
            r_traj.append(float(line.split()[-1]))
        r_traj = np.asarray(r_traj)
        f.close()
        
        mfpt = optimalrcs.MFPTNE(boundary0=r_traj < 1.0)
        print(metrics.low_bound_i_mfpt_eq(mfpt))

        def comp_y():
            return r_traj

        def gamma(iter, max_iter):
            return 0.5

        def envelope(r_traj, iter, max_iter):
            return np.ones_like(r_traj)

        np.random.seed(0)
        mfpt.fit_transform(comp_y=comp_y, envelope=envelope, gamma=gamma, max_iter=1, min_delta_x=1e-4, print_step=1)
        mfpt.fit_transform(comp_y=comp_y, max_iter=10, min_delta_x=1e-4*637, print_step=1)
        mfpt.plots_feps()
        #q.plots_feps(delta_t_sim=1, reweight=True)
        mfpt.plots_obs_pred()
        #q.plots_feps(r_traj=q.r_traj_min_sd_zt)
        #q.plots_obs_pred(r_traj=q.r_traj_min_sd_zt)


if __name__ == '__main__':
    unittest.main()
