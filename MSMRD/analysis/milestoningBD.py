import numpy as np

class milestoning_BD:
    def __init__(self, trajectories, Rbound, Runbound):
        self.trajs = trajectories #list of trajectories
        self.Rbound = Rbound
        self.Runbound = Runbound
        self.bs = [] #empty list for milestoning trajectories

    def compute_milestoning_trajectory(self):
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
