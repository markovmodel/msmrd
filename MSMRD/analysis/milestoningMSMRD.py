import numpy as np
from ..utils import voronoi


class milestoningMSMRD:
    def __init__(self, trajectory, Rbound, Runbound, MSM):
        self.BDtraj = trajectory[:,1:3]
        self.msmTraj = trajectory[:,3]
        self.Rbound = Rbound
        self.Runbound = Runbound
        self.centers = MSM.centers
        self.b = np.zeros(len(self.msmTraj))

    def computeWeights(self, boundary, MSMradius, resolution=0.01):
        #compute weights on cluster centers of MSM
        #create grid
        gridX = np.arange(-boundary, boundary, resolution)
        gridY = np.arange(-boundary, boundary, resolution)
        #allocate gridpoints to cluster centers within MSM range
        count = np.zeros(self.centers.shape[0], dtype='float')
        countA = np.zeros(self.centers.shape[0], dtype='float')
        countB = np.zeros(self.centers.shape[0], dtype='float')
        for i in range(0, len(gridX)):
            for j in range(0, len(gridY)):
                position = np.array([[gridX[i],gridY[j]]])
                if np.linalg.norm(position) <= MSMradius:
                    state = voronoi(position, self.centers)
                    count[state] += 1
                    if np.linalg.norm(position) >= self.Runbound:
                        countB[state] += 1
                    elif np.linalg.norm(position) <= self.Rbound:
                        countA[state] += 1
        self.weightA = countA/count
        self.weightB = countB/count
        self.idcsAw = np.where(countA != 0)[0]
        self.idcsBw = np.where(countB != 0)[0]

    def computeMilestoningTraj(self):
        #b = 1 corresponds to the unbound and b=0 to the bound state
        self.b[0] = 1
        for i in range(1, len(self.msmTraj)):
            if self.b[i-1] == 0:
                if self.msmTraj[i] == -1:
                    if np.linalg.norm(self.BDtraj[i]) >= self.Runbound:
                        self.b[i] = 1
                    else:
                        self.b[i] = 0
                elif np.in1d(self.msmTraj[i], self.idcsBw)[0]:
                    if np.random.rand() < self.weightB[ int(self.msmTraj[i]) ]:
                        self.b[i] = 1
                else:
                    self.b[i] = 0
            elif self.b[i-1] == 1:
                if self.msmTraj[i] == -1:
                    if np.linalg.norm(self.BDtraj[i]) <= self.Rbound:
                        self.b[i] = 0
                    else:
                        self.b[i] = 1
                elif np.in1d(self.msmTraj[i], self.idcsAw)[0]:
                    if np.random.rand() < self.weightA[ int(self.msmTraj[i]) ]:
                        self.b[i] = 0
                else:
                    self.b[i] = 1

    def computeTransitionTimes(self):
        transitions = self.b[1:]-self.b[:-1]
        self.UtoB = np.where(transitions == 1)[0]
        self.BtoU = np.where(transitions == -1)[0]

    def transitionTimes(self):
        if len(self.BtoU) == len(self.UtoB):
            if self.BtoU[0] < self.UtoB[0]:
                print "case1"
                self.tauU = self.UtoB - self.BtoU
                self.tauB = self.BtoU[1:] - self.UtoB[:-1]
            else:
                print "case2"
                self.tauU = self.BtoU - self.UtoB
                self.tauB = self.UtoB[1:] - self.BtoU[:-1]
        elif len(self.BtoU)+1 == len(self.UtoB):
            if self.BtoU[0] < self.UtoB[0]:
                print "case3"
                self.tauU = self.UtoB[:-1] - self.BtoU
                self.tauB = self.BtoU[1:] - self.UtoB[:-2]
            else:
                print "case4"
                self.tauB = self.BtoU - self.UtoB[:-1]
                self.tauU = self.UtoB[1:] - self.BtoU
        elif len(self.BtoU)-1 == len(self.UtoB):
            if self.BtoU[0] < self.UtoB[0]:
                print "case5"
                self.tauU = self.UtoB - self.BtoU[:-1]
                self.tauB = self.BtoU[1:] - self.UtoB
            else:
                print "case6"
                self.tauB = self.BtoU[:-1] - self.UtoB
                self.tauU = self.UtoB[1:-1] - self.BtoU[:-2]
        else:
            print "Trajectories not compatible"