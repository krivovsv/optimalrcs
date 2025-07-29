# Copyright (c) 2025 Sergei Krivov
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import os
import unittest
import numpy as np
import optimalrcs.optimalrcs as optimalrcs
import optimalrcs.metrics as metrics


class TestCommittor(unittest.TestCase):

    def test_2f4k(self):
        file_path = os.path.join(os.path.dirname(__file__), "data", "2f4k.CArmsd")
        f = open(file_path, 'r')
        r_traj = []
        for line in f:
            r_traj.append(float(line.split()[-1]))
        r_traj = np.asarray(r_traj)
        f.close()
        q = optimalrcs.Committor(boundary0=r_traj > 10.5, boundary1=r_traj < 1.0)
        print(metrics.low_bound_delta_r2_eq(q).numpy())

        def comp_y():
            return r_traj

        def gamma(iteration, max_iter):
            return 0.5

        def envelope(r_traj, iteration, max_iter):
            return np.ones_like(r_traj)

        q = optimalrcs.CommittorNE(boundary0=r_traj > 10.5, boundary1=r_traj < 1.0)
        print(metrics.low_bound_delta_r2_eq(q).numpy())
        q.fit_transform(comp_y=comp_y, envelope=envelope, gamma=gamma, max_iter=2, min_delta_x=1e-4, print_step=1)
        q.fit_transform(comp_y=comp_y, max_iter=10, min_delta_x=1e-4, print_step=1)
        q.plots_feps(delta_t_sim=1)
        q.plots_feps(delta_t_sim=1, reweight=True)
        q.plots_obs_pred()
        q.plots_feps(r_traj=q.r_traj_min_sd_zq)
        q.plots_obs_pred(r_traj=q.r_traj_min_sd_zq)


if __name__ == '__main__':
    unittest.main()
