# Copyright (c) 2025 Sergei Krivov
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import unittest
import numpy as np
import matplotlib.pyplot as plt
import optimalrcs.boundaries as bd
import optimalrcs.plots as plots

class TestPlots(unittest.TestCase):
    def test_2f4k(self):
        import pickle, os
        file_path = os.path.join(os.path.dirname(__file__), "data", "q-SOTA.pkl")
        f = open(file_path, 'rb')
        r_traj = pickle.load(f)
        b_traj = np.where(np.logical_or(r_traj == 1, r_traj == 0), 1, 0)
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        plots.plot_fep(axs[0,0], r_traj)
        plots.plot_fep(axs[0,1], r_traj, natural=True)
        plots.plot_zq(axs[0, 2], r_traj, b_traj)
        future_boundary = bd.FutureBoundary(r_traj, b_traj)
        plots.plot_obs_pred_q(axs[1,0], r_traj, future_boundary, ax2=axs[1,1])
        plots.plot_roc_curve(axs[1,2], r_traj, future_boundary)
        plots.plot_obs_pred_q(axs[2,0], r_traj, future_boundary, ax2=axs[2,1], log_scale=True)
        plots.plot_zc1(axs[2,2], r_traj, b_traj, ln=False)

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
