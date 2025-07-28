import unittest
import numpy as np
import os
import optimalrcs.metrics as metrics


class TestDeltaR2(unittest.TestCase):

    def test_single_transitions(self):
        r_traj = np.asarray([0.1, 0.2])
        self.assertAlmostEqual(metrics._delta_r2(r_traj).numpy(), 0.01)

    def test_single_trajectory(self):
        for dt in range(1, 10):
            r_traj = np.asarray([0.1, 0, 0.1, 0.2, 0.3])
            b_traj = np.asarray([0, 1, 0, 0, 0])
            r_traj_tp1 = np.asarray([0., ]  + [0, 0.1, 0.2, 0.3])
            r_traj_tp2 = np.asarray([0.1, 0] + [0, 0, 0])
            val = metrics._delta_r2_eq_nobd(r_traj_tp1, dt=dt).numpy()
            val += metrics._delta_r2_eq_nobd(r_traj_tp2, dt=dt).numpy()

            self.assertAlmostEqual(metrics._delta_r2(r_traj, b_traj, dt=dt).numpy(), val)

            r_traj = np.asarray([0.5, 0.6, 0.7, 1.0, 0.9])
            b_traj = np.asarray([0, 0, 0, 1, 0])
            # TP summation scheme; continue boundary states dt times and compute dr2
            r_traj_tp1 = np.asarray([0.5, 0.6, 0.7, 1.0] + [1.0,])
            r_traj_tp2 = np.asarray([1.0, 1.0,  1.0] + [1.0, 0.9])
            val = metrics._delta_r2_eq_nobd(r_traj_tp1, dt=dt).numpy()
            val += metrics._delta_r2_eq_nobd(r_traj_tp2, dt=dt).numpy()
            self.assertAlmostEqual(metrics._delta_r2(r_traj, b_traj, dt=dt).numpy(), val)


    def test_two_trajectory(self):
        for dt in range(1, 10):
            r_traj = np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.1, 0.4])
            b_traj = np.asarray([1, 0, 0, 0, 0, 0, 0, 0])
            i_traj = np.asarray([1, 1, 1, 1, 1, 2, 2, 2])
            val = metrics._delta_r2_slow_exact(r_traj, b_traj, i_traj, dt=dt).numpy()
            self.assertAlmostEqual(metrics._delta_r2(r_traj, b_traj, i_traj, dt=dt).numpy(), val)

            r_traj_tp_1 = np.asarray([0, 0.1, 0.2, 0.3, 0.4])
            r_traj_tp_2 = np.asarray([0.3, 0.1, 0.4])
            val = metrics._delta_r2_eq_nobd(r_traj_tp_1, dt=dt) + metrics._delta_r2_eq_nobd(r_traj_tp_2, dt=dt)
            self.assertAlmostEqual(metrics._delta_r2(r_traj, b_traj, i_traj, dt=dt).numpy(), val.numpy())

            r_traj = np.asarray([0.3, 0.2, 0.1, 0.5, 0.6, 0.7, 1.0])
            b_traj = np.asarray([0, 0, 0, 0, 0, 0, 1])
            i_traj = np.asarray([1, 1, 1, 2, 2, 2, 2])
            val = metrics._delta_r2_slow_exact(r_traj, b_traj, i_traj, dt=dt).numpy()
            self.assertAlmostEqual(metrics._delta_r2(r_traj, b_traj, i_traj, dt=dt).numpy(), val)
            # TP summation scheme; continue boundary states dt times and compute dr2
            r_traj_tp_1 = np.asarray([0.3, 0.2, 0.1])
            r_traj_tp_2 = np.asarray([0.5, 0.6, 0.7, 1.0] + [1.0,]*dt)
            #val = metrics.delta_r2_eq_nobd(r_traj_tp_1, dt=dt)+metrics.delta_r2_eq_nobd(r_traj_tp_2, dt=dt)
            #self.assertAlmostEqual(metrics.delta_r2(r_traj, b_traj, i_traj, dt=dt).numpy(), val.numpy())

    def _test_random_walks(self):
        for seed in range(10):
            r_traj = []
            b_traj = []
            i_traj = []
            np.random.seed(seed)
            for i in range(10):
                x0 = np.random.random()
                for step in range(200):
                    x0 = x0+0.1*(np.random.random()-0.5)
                    if x0 > 1:
                        x0 = 1
                    if x0 < 0:
                        x0 = 0
                    r_traj.append(x0)
                    i_traj.append(i)
                    if x0 == 0 or x0 == 1:
                        b_traj.append(1)
                    else:
                        b_traj.append(0)
            r_traj = np.asarray(r_traj)
            b_traj = np.asarray(b_traj)
            i_traj = np.asarray(i_traj)
            for dt in range(1, 10):
                val = 0
                for i in range(10):
                    val += metrics._delta_r2(r_traj[i_traj == i], b_traj[i_traj == i], dt=dt)
                self.assertAlmostEqual(metrics._delta_r2(r_traj, b_traj, i_traj, dt=dt).numpy(), val.numpy())


class TestMetrics(unittest.TestCase):
    def test_2f4k(self):
        import pickle
        file_path = os.path.join(os.path.dirname(__file__), "data", "q-SOTA.pkl")
        f = open(file_path, 'rb')
        r_traj = pickle.load(f)
        f.close()
        b_traj = np.where(np.logical_or(r_traj == 1, r_traj == 0), 1, 0)
        print()
        print(metrics._low_bound_delta_r2_eq(r_traj, b_traj).numpy())
        print(metrics._mse(r_traj, b_traj).numpy())
        print(metrics._mse_eq(r_traj, b_traj).numpy())
        print(metrics._cross_entropy(r_traj, b_traj).numpy())
        print(metrics._cross_entropy(r_traj, b_traj, eps=1e-8).numpy())


if __name__ == '__main__':
    unittest.main()
