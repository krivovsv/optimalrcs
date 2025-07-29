# Copyright (c) 2025 Sergei Krivov
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import unittest
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import optimalrcs.boundaries as bd
import optimalrcs.cut_profiles as cut_profiles
import optimalrcs.metrics as metrics


class TestZC1(unittest.TestCase):

    def test_single_trajectory(self):
        r_traj = np.asarray([0.4, 0.5, 0.8])
        b_traj = np.asarray([0, 0, 0])
        fb = bd.FutureBoundary(r_traj, b_traj)
        pb = bd.PastBoundary(r_traj, b_traj)
        lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, fb, pb)
        npt.assert_array_almost_equal(lx, np.linspace(0.4, 0.8, 1001, True))
        lz1 = np.ones(1000)
        lz1[:250] = 0.05
        lz1[250:] = 0.15
        lz1[-1] = 0
        npt.assert_array_almost_equal(lz, lz1)

    def test_boundaries(self):
        r_traj = np.asarray([1, 0, 1], dtype=np.float64)
        b_traj = np.asarray([1, 1, 1])
        fb = bd.FutureBoundary(r_traj, b_traj)
        pb = bd.PastBoundary(r_traj, b_traj)
        lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, fb, pb)
        npt.assert_array_almost_equal(lx, np.linspace(0, 1, 1001, True))
        lz1 = np.ones(1000)
        lz1[-1] = 0
        npt.assert_array_almost_equal(lz, lz1)

        lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, fb, pb, dt=10)
        lz1 = np.ones(1000)
        lz1[-1] = 0
        npt.assert_array_almost_equal(lz, lz1)

    def test_boundaries2(self):
        r_traj = np.asarray([1, 0, 1], dtype=np.float64)
        b_traj = np.asarray([1, 1, 1])
        lx, lz = cut_profiles.comp_zc1(r_traj, b_traj)
        npt.assert_array_almost_equal(lx, np.linspace(0, 1, 1001, True))
        lz1 = np.ones(1000)
        lz1[-1] = 0
        npt.assert_array_almost_equal(lz, lz1)

        lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, dt=10)
        lz1 = np.ones(1000)
        lz1[-1] = 0
        npt.assert_array_almost_equal(lz, lz1)

    def test_boundaries3(self):
        r_traj = np.asarray([0, 0.4, 1], dtype=np.float64)
        b_traj = np.asarray([1, 0, 1])
        fb = bd.FutureBoundary(r_traj, b_traj)
        pb = bd.PastBoundary(r_traj, b_traj)
        lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, fb, pb, nbins=10)
        npt.assert_array_almost_equal(lx, np.linspace(0, 1, 11, True))
        lz1 = np.ones(10)
        lz1[:4] = 0.2
        lz1[4:] = 0.3
        lz1[-1] = 0
        npt.assert_array_almost_equal(lz, lz1)


def integrate(lx, lz):
    return tf.reduce_sum((lx[1:]-lx[:-1])*lz)


def random_walks(mtraj, msteps):
    r_traj = []
    b_traj = []
    i_traj = []
    for i in range(mtraj):
        x0 = np.random.random()
        for _ in range(msteps):
            x0 = x0 + 0.1 * (np.random.random() - 0.5)
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
    return r_traj, b_traj, i_traj

def random_walk(ntraj, nsteps):
    r_traj=np.zeros(ntraj*nsteps,'float32')
    b_traj=np.zeros((ntraj*nsteps),'float32')
    i_traj=np.zeros((ntraj*nsteps),'int32')
    k=0
    for i in range(ntraj):
        x=np.random.choice((0,1,2,3,4,5,6,7,8,9,10))
        r_traj[k]=x/10
        i_traj[k]=i
        k=k+1
        for _ in range(1,nsteps):
            x=x+np.random.choice((1,-1))
            if x<0:x=1
            if x>10:x=9
            r_traj[k]=x/10
            i_traj[k]=i
            k=k+1
    b_traj[r_traj==0]=1
    b_traj[r_traj==1]=1
    return r_traj, b_traj, i_traj


ldt = [2**i for i in range(10)]


class TestDeltaR2(unittest.TestCase):  # compare dr2 computed from the profile and directly

    def test_single_trajectory(self):
        for dt in ldt:
            r_traj = np.asarray([0, 0.1, 0.2, 0.3])
            b_traj = np.asarray([1, 0, 0, 0])
            val = metrics._delta_r2(r_traj, b_traj, dt=dt) / 2
            lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, dt=dt, nbins=100000)
            self.assertAlmostEqual(integrate(lx, lz).numpy(), val.numpy(), 4)

            r_traj = np.asarray([0.5, 0.6, 0.7, 1.0])
            b_traj = np.asarray([0, 0, 0, 1])
            val = metrics._delta_r2(r_traj, b_traj, dt=dt) / 2
            lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, dt=dt, nbins=100000)
            self.assertAlmostEqual(integrate(lx, lz).numpy(), val.numpy(), 4)

            r_traj = np.asarray([0., 0.2, 0.5, 0.7, 1.0])
            b_traj = np.asarray([1, 0, 0, 0, 1])
            val = metrics._delta_r2(r_traj, b_traj, dt=dt) / 2
            lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, dt=dt, nbins=100000)
            self.assertAlmostEqual(integrate(lx, lz).numpy(), val.numpy(), 4)

            r_traj = np.asarray([0., 0.5, 1.0, 0.7, 0.])
            b_traj = np.asarray([1, 0, 1, 0, 1])
            val = metrics._delta_r2(r_traj, b_traj, dt=dt) / 2
            lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, dt=dt, nbins=100000)
            self.assertAlmostEqual(integrate(lx, lz).numpy(), val.numpy(), 4)

    def test_random_walks(self):
        for seed in range(10):
            np.random.seed(seed)
            r_traj, b_traj, i_traj = random_walks(10, 3)
            for dt in ldt:
                val = metrics._delta_r2(r_traj, b_traj, i_traj=i_traj, dt=dt) / 2
                lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, i_traj=i_traj, dt=dt, nbins=100000)
                self.assertAlmostEqual(integrate(lx, lz).numpy(), val.numpy(), 4)


class TestZq(unittest.TestCase):  # compare dr2 computed from the profile and directly

    def test_single_trajectory(self):
        for dt in ldt:
            r_traj = np.asarray([0., 0.2, 0.5, 0.7, 1.0])
            b_traj = np.asarray([1, 0, 0, 0, 1])
            _, lz1 = cut_profiles.comp_zq(r_traj, b_traj, dt=dt)
            _, lz2 = cut_profiles.comp_zq(r_traj[::-1], b_traj[::-1], dt=dt)
            val = (lz1+lz2)/2
            _, lz = cut_profiles.comp_zc1(r_traj, b_traj, dt=dt)
            npt.assert_array_almost_equal(lz.numpy(), val.numpy())

    def test_random_walks(self):
        for seed in range(10):
            np.random.seed(seed)
            r_traj, b_traj, i_traj = random_walks(10, 200)
            for dt in ldt:
                _, lz1 = cut_profiles.comp_zq(r_traj, b_traj, i_traj, dt=dt)
                _, lz2 = cut_profiles.comp_zq(r_traj[::-1], b_traj[::-1], i_traj[::-1], dt=dt)
                val = (lz1 + lz2) / 2
                _, lz = cut_profiles.comp_zc1(r_traj, b_traj, i_traj=i_traj, dt=dt)
                npt.assert_array_almost_equal(lz.numpy(), val.numpy())


class TestZCa(unittest.TestCase):  # compare dr2 computed from the profile and directly

    def test_single_trajectory(self):
        r_traj = np.asarray([0., 0.2, 0.5, 0.7, 1.0])
        b_traj = np.asarray([1, 0, 0, 0, 1])
        lx, lz = cut_profiles.comp_zc1(r_traj, b_traj, dt=1)
        lx1, lz1 = cut_profiles.comp_zca(r_traj, 1, dt=1)
        npt.assert_array_almost_equal(lz.numpy(), lz1.numpy())

    def test_random_walks(self):
        for seed in range(10):
            np.random.seed(seed)
            r_traj, b_traj, i_traj = random_walks(10, 200)
            for dt in [1,]: # no path summation in zca
                _, lz = cut_profiles.comp_zc1(r_traj, b_traj, i_traj=i_traj, dt=dt)
                _, lz1 = cut_profiles.comp_zca(r_traj, 1, i_traj=i_traj, dt=dt)
                npt.assert_array_almost_equal(lz.numpy(), lz1.numpy())


if __name__ == '__main__':
    unittest.main()
