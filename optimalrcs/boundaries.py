# Copyright (c) 2025 Sergei Krivov
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import numpy as np


class FutureBoundary:
    """
    class to contain information about boundaries in the future, to correctly describe martingale at the boundaries
    self.index[i] - index of the next boundary in the future for the current point with index i
            if i is a boundary itself, then index[i]=i
    self.r[i] - r value of the next boundary in the future
    self.delta_i - difference in indices between the current point and the future boundary
    self.index2[i] - index for the next boundary in the future for the current point eith index i, however
            if i is a boundary itself, then index2[i] points to the next boundary
    self.r2[i] - r value of the next boundary in the future, however
            if i is a boundary itself, then r2[i] points to the next boundary
    self.delta_i2 - difference in indices between the current point and the future boundary, however
            if i is a boundary itself, then delta_i2[i]>0 points to the next boundary
    self.delta_i_to_end - distance to the end of trajectory
    """
    def __init__(self, r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray = None, i_traj: np.ndarray = None) -> None:
        n = len(r_traj)
        self.index = np.zeros(n, 'int32')  # index of the boundary in the future
        index_current = -1
        if b_traj[-1] > 0:
            index_current = n - 1
        self.index[-1] = index_current
        for i in range(n - 2, -1, -1):
            if (i_traj is not None) and (i_traj[i] != i_traj[i + 1]):  # new trajectory's end
                index_current = -1
            if b_traj[i] > 0:
                index_current = i
            self.index[i] = index_current

        #if i_traj is None or len(np.unique(i_traj))==1:
        #    self.delta_i_to_end = np.array(range(n-1,-1,-1))
        #else:
        #    self.set_distance_to_end(i_traj)

        self.delta_i_to_end = np.zeros(n, 'int32')  # index of the boundary in the future
        delta_i_to_end_current = 0
        for i in range(n - 2, -1, -1):
            delta_i_to_end_current += 1
            if (i_traj is not None) and (i_traj[i] != i_traj[i + 1]):  # new trajectory's end
                delta_i_to_end_current = 0
            self.delta_i_to_end[i] = delta_i_to_end_current
        
        self.r = np.where(self.index > -1, r_traj[self.index], 0)
        index_frame = range(n)
        self.delta_i = np.where(self.index > -1, self.index - index_frame, 0)
        self.index2=np.roll(self.index, -1)
        self.index2[-1]=-1
        if i_traj is not None:
            self.index2[i_traj!=np.roll(i_traj,-1)]=-1
        self.r2 = np.where(self.index2 > -1, r_traj[self.index2], 0)
        self.delta_i2 = np.where(self.index2 > -1, self.index2 - index_frame, 0)

        if t_traj is None:
            self.delta_t = self.delta_i
            self.delta_t2 = self.delta_i2
        else:
            self.delta_t = np.where(self.index > -1, t_traj[self.index] - t_traj[index_frame], 0)
            self.delta_t2 = np.where(self.index2 > -1, t_traj[self.index2] - t_traj[index_frame], 0)

    def set_distance_to_end(self, i_traj):
        traj_ends=np.where(np.roll(i_traj,-1)!=i_traj)[0]
        traj_starts=np.concatenate(([0],traj_ends[:-1]+1))
        self.delta_i_to_end=np.zeros_like(i_traj)
        for i_start, i_end in zip(traj_starts, traj_ends):
            self.delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
            
    def set_distance_to_end_fixed_traj_length_trap(self, i_traj, trap_boundary, traj_length):
        traj_ends=np.where(np.roll(i_traj,-1)!=i_traj)[0]
        traj_starts=np.concatenate(([0],traj_ends[:-1]+1))
        self.delta_i_to_end=np.zeros_like(i_traj)
        traj_length=traj_length-1
        for i_start, i_end in zip(traj_starts, traj_ends):
            self.delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
            if trap_boundary[i_end] and (traj_length > i_end-i_start):
                self.delta_i_to_end[i_start:i_end+1] += traj_length - (i_end - i_start)

    def set_distance_to_end_poisson_traj_length_trap(self, i_traj, trap_boundary, average_traj_length=None):
            traj_ends=np.where(np.roll(i_traj,-1)!=i_traj)[0]
            traj_starts=np.concatenate(([0],traj_ends[:-1]+1))
            self.delta_i_to_end=np.zeros_like(i_traj)
            tb=0
            nb=0
            if average_traj_length is None:
                for i_start, i_end in zip(traj_starts, traj_ends):
                    if trap_boundary[i_end]:
                        tb+=i_end-i_start
                        nb+=1
                tnb=(len(i_traj)-tb)/(len(traj_ends)-nb)
                tb=tb/nb
                average_traj_length=int(tnb-tb)
                #print (tb,tnb,average_traj_length)
            for i_start, i_end in zip(traj_starts, traj_ends):
                self.delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
                if trap_boundary[i_end]:
                    traj_length=int(-np.log(np.random.random())*average_traj_length -1)
                    self.delta_i_to_end[i_start:i_end+1] += traj_length
            
class PastBoundary:
    def __init__(self, r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray = None, i_traj: np.ndarray = None) -> None:
        n = len(r_traj)
        self.index = np.zeros(n, 'int32')  # index of the boundary in the future
        index_current = -1
        if b_traj[0] > 0:
            index_current = 0
        self.index[0] = index_current
        for i in range(1,n):
            if (i_traj is not None) and (i_traj[i] != i_traj[i - 1]):
                index_current = -1
            if b_traj[i] > 0:
                index_current = i
            self.index[i] = index_current

        self.delta_i_from_start = np.zeros(n, 'int32')  # index of the boundary in the future
        delta_i_from_start_current = 0
        for i in range(1,n):
            delta_i_from_start_current += 1
            if (i_traj is not None) and (i_traj[i] != i_traj[i - 1]):  # new trajectory's end
                delta_i_from_start_current=0
            self.delta_i_from_start[i] = delta_i_from_start_current

        self.r = np.where(self.index > -1, r_traj[self.index], 0)
        index_frame = range(n)
        self.delta_i = np.where(self.index > -1, self.index - index_frame, 0)
        self.index2=np.roll(self.index, 1)
        self.index2[0]=-1
        if i_traj is not None:
            self.index2[i_traj!=np.roll(i_traj,1)]=-1
        self.r2 = np.where(self.index2 > -1, r_traj[self.index2], 0)
        self.delta_i2 = np.where(self.index2 > -1, self.index2 - index_frame, 0)
        if t_traj is None:
            self.delta_t=self.delta_i
            self.delta_t2=self.delta_i2
        else:
            self.delta_t = np.where(self.index > -1, t_traj[self.index] - t_traj[index_frame], 0)
            self.delta_t2 = np.where(self.index2 > -1, t_traj[self.index2] - t_traj[index_frame], 0)
