import numpy as np
from .utils import voronoi

class MSM:
    def __init__(self, path, exitR = 0., MSMradius=0., lagtime=1):
        self.T = np.loadtxt(path + '_Tmatrix.txt')
        self.centers = np.loadtxt(path + '_centers.txt')
        self.MSMradius=MSMradius
        self.lagtime = lagtime
        self.states = self.T.shape[0]
        self.exitStates = np.where(np.linalg.norm(self.centers, axis=1) > exitR)[0]
        self.entryStates = np.where(np.linalg.norm(self.centers, axis=1) <= exitR)[0]
        #self.entryStates = np.where(np.logical_and(np.linalg.norm(self.centers, axis=1) < entryRadii[1], np.linalg.norm(self.centers,axis=1) > entryRadii[0]))[0]
        self.state = -1
        self.exit = False

    def propagate(self):
        self.state = np.random.choice(self.states, p=self.T[self.state])
        self.exit = np.in1d(self.state, self.exitStates)

    def allocateStates(self, traj):
        clusters = voronoi(traj, self.centers)
        return clusters