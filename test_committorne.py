import numpy as np
import unittest
import optimalrcs


class TestCommittorNE(unittest.TestCase):

    def test_2f4k_history(self):
        f = open('2f4k.CArmsd', 'r')
        r_traj = []
        for line in f:
            r_traj.append(float(line.split()[-1]))
        r_traj = np.asarray(r_traj)
        f.close()
        q = optimalrcs.CommittorNE(boundary0=r_traj > 10.5, boundary1=r_traj < 1.0)
        print(q.metric_low_bound_delta_r2_eq().numpy())

        def comp_y():
            return r_traj
        history_delta_t = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        q.fit_transform(comp_y=comp_y, gamma=0.05, history_delta_t=history_delta_t, max_iter=1000, min_delta_x=1e-4,
                        print_step=100)
        q.plots_feps(delta_t_sim=1)
        q.plots_obs_pred()
        q.plots_feps(r_traj=q.r_traj_min_sd_zq)
        q.plots_obs_pred(r_traj=q.r_traj_min_sd_zq)

    def test_2f4k(self):
        f = open('2f4k.CArmsd', 'r')
        r_traj = []
        for line in f:
            r_traj.append(float(line.split()[-1]))
        r_traj = np.asarray(r_traj)
        f.close()
        q = optimalrcs.CommittorNE(boundary0=r_traj > 10.5, boundary1=r_traj < 1.0)
        print(q.metric_low_bound_delta_r2_eq().numpy())

        def comp_y():
            return r_traj

        def gamma(iter, max_iter):
            return 0.5

        def envelope(r_traj, iter, max_iter):
            return np.ones_like(r_traj)

        q.fit_transform(comp_y=comp_y, envelope=envelope, gamma=gamma, max_iter=1, min_delta_x=1e-4, print_step=1)
        q.fit_transform(comp_y=comp_y, max_iter=10, min_delta_x=1e-4, print_step=1)
        q.plots_feps(delta_t_sim=1)
        q.plots_feps(delta_t_sim=1, reweight=True)
        q.plots_obs_pred()
        q.plots_feps(r_traj=q.r_traj_min_sd_zq)
        q.plots_obs_pred(r_traj=q.r_traj_min_sd_zq)


if __name__ == '__main__':
    unittest.main()
