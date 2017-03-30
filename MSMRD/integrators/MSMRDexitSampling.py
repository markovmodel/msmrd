import numpy as np
from ..integrator import integrator


class MSMRDexitSampling(integrator):
    def __init__(self,  MSM, radius, p, timestep, parameters):
        self.MSM = MSM
        #Radius of the MSM domain
        self.radius = radius
        self.dim = p.position.size
        self.p = p
        self.timestep = timestep
        self.sigma = np.sqrt(2.*self.p.D*self.timestep)
        #get entry and exit radius directly from region map object
        self.entryRadius = parameters['entryRadius']
        self.interactionRadius = parameters['interactionRadius']
        self.bathRadius = parameters['bathRadius']
        self.NangularPartitions = parameters['NangularPartitions']
        self.entryRings = parameters['entryRings']
        self.exitRings = parameters['exitRings']
        self.radialIncrementEntry = (self.entryRadius - self.interactionRadius)/float(self.entryRings)
        self.radialIncrementExit = (self.bathRadius - self.entryRadius)/float(self.exitRings)
        self.angularIncrement = float(2.*np.pi/ self.NangularPartitions)
        self.NCenters = parameters['NCenters']
        self.sampleSize = 4 #sample consists of (time, p, MSMstate)
        self.MSMactive = False
        self.lastState = -1

    def above_threshold(self, threshold):
        #assume that threshold is larger than the MSM radius
        if self.MSMactive:
            return np.linalg.norm(self.MSM.centers[self.MSM.state]) > threshold
        else:
            return True

    def propagateDiffusion(self, particle):
        #use inversion on circle to keep the particle inside of the simulation radius
        #see https://de.wikipedia.org/wiki/Kreisspiegelung for details
        dr = np.random.normal(0., self.sigma, self.dim)
        assert len(dr) == self.dim
        newPosition = particle.position + dr
        rNew = np.linalg.norm(newPosition)
        if rNew >= self.radius:
            newPosition = newPosition*self.radius**2/(rNew**2)
        particle.position = newPosition

    def enterMSM(self):
        #assign closest state as entry state in MSM domain
        #test not allowing for direct transitions to inner MSM states
        R = self.p.position
        if np.linalg.norm(R) < self.interactionRadius:
            entranceState = (np.linalg.norm(self.MSM.centers - R, axis=1)).argmin()
        else:
            entranceRing = int((np.linalg.norm(R)-self.interactionRadius) / self.radialIncrementEntry)
            theta = np.arctan2(R[1], R[0]) + np.pi #add pi for the angle to be in [0, 2pi]
            entranceState = np.floor(theta / (2.*np.pi) * self.NangularPartitions) + self.NCenters + entranceRing * self.NangularPartitions
        self.MSM.state = entranceState
        self.MSM.exit = False
        self.MSMactive = True

    def exitMSM(self):
        #Exit MSM domain: pick new position in last state the particle was in and perform 1 BD step that leaves the MSM domain
        assert(self.lastState >= self.NCenters)
        exitState = self.lastState - self.NCenters
        if exitState < self.NangularPartitions*self.exitRings:
            #exit from entry state
            exitRing = int(exitState / self.NangularPartitions)
            thetaL = (exitState % self.NangularPartitions) * self.angularIncrement
            theta = np.random.random()*(self.angularIncrement) + thetaL
            Rlower = self.interactionRadius + exitRing * self.radialIncrementEntry
            Rupper = Rlower + self.radialIncrementEntry
            R = np.sqrt(np.random.random()*(Rupper**2 - Rlower**2) + Rlower**2)
        else :
            #exit from exit state
            exitRing = int(exitState / self.NangularPartitions)
            thetaL = (exitState%self.NangularPartitions)  * self.angularIncrement
            theta = np.random.random()*(self.angularIncrement) + thetaL
            Rlower = self.interactionRadius + exitRing * self.radialIncrementExit
            Rupper = Rlower + self.radialIncrementExit
            R = np.sqrt(np.random.random()*(Rupper**2 - Rlower**2) + Rlower**2)
        internalPosition = R*np.array([np.cos(theta+np.pi), np.sin(theta+np.pi)])
        newPosition = np.zeros(2)
        exitSigma = self.sigma * np.sqrt(self.MSM.lagtime)
        counter = 0
        while np.linalg.norm(newPosition) < self.bathRadius:
            counter += 1 
            dr = np.random.normal(0., exitSigma, self.dim)
            assert(len(dr) == 2)
            newPosition = internalPosition + dr
            if counter > 10000:
                print 'no exit position found'
                break
        assert newPosition.shape[0] == 2
        self.p.position = newPosition
        self.MSMactive = False

    def integrate(self):
        if self.MSMactive:
            #save last state to sample from in case of an exit event
            self.lastState = self.MSM.state
            self.MSM.propagate()
            if self.MSM.exit:
                self.exitMSM()
        elif not self.MSMactive:
            self.propagateDiffusion(self.p)
            if np.linalg.norm(self.p.position) < self.entryRadius:
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
