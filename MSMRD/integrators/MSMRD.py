import numpy as np
from ..integrator import integrator


class MSMRD(integrator):
    def __init__(self,  MSM, box, p1, p2, p3, timestep, Re):
        self.MSM = MSM
        self.box = box
        self.dim = p1.position.size
        self.pa = p1
        self.pb = p2
        self.pc = p3
        self.timestep = timestep
        self.Re = Re #entry radius
        self.sampleSize = 8 #sample consists of (time, p1, p2, p3, MSMstate)
        self.MSMactive = False

    def above_threshold(self, threshold):
        #assume that threshold is larger than the MSM radius
        if self.MSMactive:
            return np.linalg.norm(self.MSM.centers[self.MSM.state]) > threshold
        else:
            return True

    def propagateDiffusion(self, particle):
        sigma = np.sqrt(2. * self.timestep * particle.D)
        dr = np.random.normal(0., sigma, self.dim)
        particle.position += dr

    def enterMSM(self):
        self.pc.position = self.pa.position
        R = self.box.periodicDistanceVector(self.pa.position, self.pb.position)
        self.MSM.state = (np.linalg.norm(self.MSM.centers[self.MSM.entryStates] - R, axis=1)).argmin()
        self.MSM.exit = False
        self.MSMactive = True

    def exitMSM(self):
        R = self.MSM.centers[self.MSM.state,:]
        self.pa.position = self.pc.position
        self.pb.position = self.pc.position + R
        self.box.reduce(self.pb)
        self.MSMactive = False

    def integrate(self):
        if self.MSMactive:
            self.MSM.propagate()
            self.propagateDiffusion(self.pc)
            self.box.reduce(self.pc)
            if self.MSM.exit:
                self.exitMSM()
        elif not self.MSMactive:
            self.propagateDiffusion(self.pa)
            self.box.reduce(self.pa)
            self.propagateDiffusion(self.pb)
            self.box.reduce(self.pb)
            if self.box.particleDistance(self.pa, self.pb) < self.Re:
                self.enterMSM()

    def sample(self, step):
        if self.MSMactive:
            return [self.timestep*step, 0., 0., 0., 0., self.pc.position[0], self.pc.position[1], self.MSM.state]
        else:
            return [self.timestep*step, self.pa.position[0], self.pa.position[1], self.pb.position[0], self.pb.position[1], 0., 0., -1]

    def compute_stationary_distribution(self, traj):
        #cluster data in transition area
        #extract BD part of trajectory
        BDidcs = np.where(traj[:,7] == -1)[0]
        BDtraj = traj[BDidcs, ...]
        r1 = BDtraj[:, 1:3]
        r2 = BDtraj[:, 3:5]
        #compute periodically reduces distance and find points in transition region
        distances = np.zeros(BDidcs[0].size)
        distances = self.box.periodicDistance(r1, r2)
        transitionRegion = np.where(distances < self.MSM.MSMradius)[0]
        #allocate transition trajectories to states
        dr = self.box.periodicDistanceVector(r1, r2)
        clusters = np.array([])
        if transitionRegion != []:
            clusters = self.MSM.allocateStates(dr[transitionRegion, ...])
        #count observations
        counts = np.zeros(self.MSM.states)
        for i in range(0, self.MSM.states):
            counts[i] += (np.where(traj[:,7] == i)[0].size)*self.MSM.lagtime
            counts[i] += np.where(clusters == i)[0].size
        counts /= float(counts.sum())
        return counts
