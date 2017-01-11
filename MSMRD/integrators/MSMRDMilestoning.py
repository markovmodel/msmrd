import numpy as np
from ..integrator import integrator


class MSMRDMilestoning(integrator):
    def __init__(self,  MSM, radius, p, timestep, Re, regionMap):
        self.MSM = MSM
        self.radius = radius
        self.dim = p.position.size
        self.p = p
        self.timestep = timestep
        self.Re = Re #entry radius
        self.regionMap = regionMap
        self.sampleSize = 4 #sample consists of (time, p, MSMstate)
        self.MSMactive = False

    def above_threshold(self, threshold):
        #assume that threshold is larger than the MSM radius
        #this function is only used for the MFTP computation
        if self.MSMactive:
            return np.linalg.norm(self.MSM.centers[self.MSM.state]) > threshold
        else:
            return True

    def propagateDiffusion(self, particle):
        sigma = np.sqrt(2. * self.timestep * particle.D)
        rNew = self.radius + 1.
        while rNew > self.radius:
            dr = np.random.normal(0., sigma, self.dim)
            assert len(dr) == self.dim
            newPosition = particle.position + dr
            rNew = np.linalg.norm(newPosition)
        particle.position = newPosition

    def enterMSM(self):
        #Enter MSM domain: search for state closest to particle postion and replace particle by MSM state.
        R = self.p.position
        entranceState = (np.linalg.norm(self.MSM.centers[self.MSM.entryStates] - R, axis=1)).argmin()
        self.MSM.state = self.MSM.entryStates[entranceState]
        self.MSM.exit = False
        self.MSMactive = True

    def exitMSM(self):
        #Exit MSM domain: pick new position from uniform distirbution on an annulus segment
        rand = np.random.random(2)
        thetaL = self.regionMap[self.MSM.state][0]
        thetaU = self.regionMap[self.MSM.state][1]
        radiusL = self.regionMap['rexit_int'][0]
        radiusU = self.regionMap['rexit_int'][1]
        theta = rand[0]*(thetaU-thetaL) + thetaL
        radius = np.sqrt(rand[1]*(radiusU**2-radiusL**2)+raiusL**2)
        newPosition = np.array(radius*[np.cos(theta),np.sin(theta)])
        self.p.position = newPosition
        self.MSMactive = False

    def integrate(self):
        if self.MSMactive:
            self.MSM.propagate()
            if self.MSM.exit:
                self.exitMSM()
        elif not self.MSMactive:
            self.propagateDiffusion(self.p)
            if np.linalg.norm(self.p.position) < self.Re:
                self.enterMSM()

    def sample(self, step):
        if self.MSMactive:
            return [self.timestep*step, 0., 0., self.MSM.state]
        else:
            return [self.timestep*step, self.p.position[0], self.p.position[1], -1]

    def compute_stationary_distribution(self, traj):
        #cluster data in transition area
        #extract BD part of trajectory
        BDidcs = np.where(traj[:,3] == -1)[0]
        BDtraj = traj[BDidcs, ...]
        dr = BDtraj[:, 1:3]
        distances = np.linalg.norm(dr, axis=1)
        #compute periodically reduces distance and find points in transition region
        transitionRegion = np.where(distances < self.MSM.MSMradius)[0]
        #allocate transition trajectories to states
        clusters = np.array([])
        if transitionRegion != []:
            clusters = self.MSM.allocateStates(dr[transitionRegion, ...])
        #count observations
        counts = np.zeros(self.MSM.states)
        for i in range(0, self.MSM.states):
            counts[i] += (np.where(traj[:,3] == i)[0].size)*self.MSM.lagtime
            counts[i] += np.where(clusters == i)[0].size
        counts /= float(counts.sum())
        return counts
