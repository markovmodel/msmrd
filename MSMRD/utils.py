#helper functions for analysis
import numpy as np


def voronoi(traj, centers):
    assert traj.shape[1] == centers.shape[1]
    ltraj = traj.shape[0]
    clustering = np.zeros(ltraj, dtype=int)
    for i in range(0, ltraj):
        index = np.argmin(np.linalg.norm(centers - traj[i,:], axis=1))
        clustering[i] = index
    return clustering

def correlate(sig1, sig2):
    sig1 = (sig1-np.mean(sig1))/np.std(sig1)
    sig2 = (sig2-np.mean(sig2))/np.std(sig2)
    return np.correlate(sig1, sig2)
