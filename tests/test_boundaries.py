import unittest
import optimalrcs.boundaries as boundaries
import numpy as np
import numpy.testing as npt


class TestFutureBoundary(unittest.TestCase):

    def test_single_trajectory(self):
        r_traj = np.asarray([0, 0.4, 1, 0.6, 0])
        b_traj = np.asarray([1, 0, 1, 0, 1])
        t_traj = np.asarray([1, 3, 4, 7, 10])
        i_traj = np.asarray([1, 1, 1, 1, 1])
        fb = boundaries.FutureBoundary(r_traj, b_traj, t_traj, i_traj)
        npt.assert_array_equal(fb.index, np.asarray([0, 2, 2, 4, 4]))
        npt.assert_array_equal(fb.r, np.asarray([0, 1, 1, 0, 0]))
        npt.assert_array_equal(fb.delta_i, np.asarray([0, 1, 0, 1, 0]))
        npt.assert_array_equal(fb.delta_t, np.asarray([0, 1, 0, 3, 0]))
        npt.assert_array_equal(fb.index2, np.asarray([2, 2, 4, 4, -1]))
        npt.assert_array_equal(fb.r2, np.asarray([1, 1, 0, 0, 0]))
        npt.assert_array_equal(fb.delta_i2, np.asarray([2, 1, 2, 1, 0]))
        npt.assert_array_equal(fb.delta_t2, np.asarray([3, 1, 6, 3, 0]))

        fb = boundaries.FutureBoundary(r_traj, b_traj, t_traj)
        npt.assert_array_equal(fb.index, np.asarray([0, 2, 2, 4, 4]))
        npt.assert_array_equal(fb.r, np.asarray([0, 1, 1, 0, 0]))
        npt.assert_array_equal(fb.delta_i, np.asarray([0, 1, 0, 1, 0]))
        npt.assert_array_equal(fb.delta_t, np.asarray([0, 1, 0, 3, 0]))
        npt.assert_array_equal(fb.index2, np.asarray([2, 2, 4, 4, -1]))
        npt.assert_array_equal(fb.r2, np.asarray([1, 1, 0, 0, 0]))
        npt.assert_array_equal(fb.delta_i2, np.asarray([2, 1, 2, 1, 0]))
        npt.assert_array_equal(fb.delta_t2, np.asarray([3, 1, 6, 3, 0]))

        fb = boundaries.FutureBoundary(r_traj, b_traj )
        npt.assert_array_equal(fb.delta_t, fb.delta_i)
        npt.assert_array_equal(fb.delta_t2, fb.delta_i2)

    def test_two_trajectories(self):
        r_traj = np.asarray([0, 0.4, 1, 0.6, 0])
        b_traj = np.asarray([1, 0, 1, 0, 1])
        t_traj = np.asarray([1, 3, 4, 7, 10])
        i_traj = np.asarray([1, 1, 1, 1, 2])
        fb = boundaries.FutureBoundary(r_traj, b_traj, t_traj, i_traj)
        npt.assert_array_equal(fb.index, np.asarray([0, 2, 2, -1, 4]))
        npt.assert_array_equal(fb.r, np.asarray([0, 1, 1, 0, 0]))
        npt.assert_array_equal(fb.delta_i, np.asarray([0, 1, 0, 0, 0]))
        npt.assert_array_equal(fb.delta_t, np.asarray([0, 1, 0, 0, 0]))
        npt.assert_array_equal(fb.index2, np.asarray([2, 2, -1, -1, -1]))
        npt.assert_array_equal(fb.r2, np.asarray([1, 1, 0, 0, 0]))
        npt.assert_array_equal(fb.delta_i2, np.asarray([2, 1, 0, 0, 0]))
        npt.assert_array_equal(fb.delta_t2, np.asarray([3, 1, 0, 0, 0]))

class TestPastBoundary(unittest.TestCase):

    def test_single_trajectory(self):
        r_traj = np.asarray([0, 0.4, 1, 0.6, 0])
        b_traj = np.asarray([1, 0, 1, 0, 1])
        t_traj = np.asarray([1, 3, 4, 7, 10])
        i_traj = np.asarray([1, 1, 1, 1, 1])
        pb = boundaries.PastBoundary(r_traj, b_traj, t_traj, i_traj)
        npt.assert_array_equal(pb.index, np.asarray([0, 0, 2, 2, 4]))
        npt.assert_array_equal(pb.r, np.asarray([0, 0, 1, 1, 0]))
        npt.assert_array_equal(pb.delta_i, np.asarray([0, -1, 0, -1, 0]))
        npt.assert_array_equal(pb.delta_t, np.asarray([0, -2, 0, -3, 0]))
        npt.assert_array_equal(pb.index2, np.asarray([-1, 0, 0, 2, 2]))
        npt.assert_array_equal(pb.r2, np.asarray([0, 0, 0, 1, 1]))
        npt.assert_array_equal(pb.delta_i2, np.asarray([0, -1, -2, -1, -2]))
        npt.assert_array_equal(pb.delta_t2, np.asarray([0, -2, -3, -3, -6]))

        pb = boundaries.PastBoundary(r_traj, b_traj, t_traj)
        npt.assert_array_equal(pb.index, np.asarray([0, 0, 2, 2, 4]))
        npt.assert_array_equal(pb.r, np.asarray([0, 0, 1, 1, 0]))
        npt.assert_array_equal(pb.delta_i, np.asarray([0, -1, 0, -1, 0]))
        npt.assert_array_equal(pb.delta_t, np.asarray([0, -2, 0, -3, 0]))
        npt.assert_array_equal(pb.index2, np.asarray([-1, 0, 0, 2, 2]))
        npt.assert_array_equal(pb.r2, np.asarray([0, 0, 0, 1, 1]))
        npt.assert_array_equal(pb.delta_i2, np.asarray([0, -1, -2, -1, -2]))
        npt.assert_array_equal(pb.delta_t2, np.asarray([0, -2, -3, -3, -6]))

        pb = boundaries.PastBoundary(r_traj, b_traj)
        npt.assert_array_equal(pb.delta_t, pb.delta_i)
        npt.assert_array_equal(pb.delta_t2, pb.delta_i2)

    def test_two_trajectories(self):
        r_traj = np.asarray([0, 0.4, 1, 0.6, 0])
        b_traj = np.asarray([1, 0, 1, 0, 1])
        t_traj = np.asarray([1, 3, 4, 7, 10])
        i_traj = np.asarray([1, 1, 1, 2, 2])
        pb = boundaries.PastBoundary(r_traj, b_traj, t_traj, i_traj)
        npt.assert_array_equal(pb.index, np.asarray([0, 0, 2, -1, 4]))
        npt.assert_array_equal(pb.r, np.asarray([0, 0, 1, 0, 0]))
        npt.assert_array_equal(pb.delta_i, np.asarray([0, -1, 0, 0, 0]))
        npt.assert_array_equal(pb.delta_t, np.asarray([0, -2, 0, 0, 0]))
        npt.assert_array_equal(pb.index2, np.asarray([-1, 0, 0, -1, -1]))
        npt.assert_array_equal(pb.r2, np.asarray([0, 0, 0, 0, 0]))
        npt.assert_array_equal(pb.delta_i2, np.asarray([0, -1, -2, 0, 0]))
        npt.assert_array_equal(pb.delta_t2, np.asarray([0, -2, -3, 0, 0]))



if __name__ == '__main__':
    unittest.main()