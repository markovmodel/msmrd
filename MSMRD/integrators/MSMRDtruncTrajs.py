import numpy as np
from ..integrator import integrator
from ..MSM import MSM


class MSMRDtruncTrajs(integrator):
    def __init__(self, MSM, radius, p, timestep, truncTrajsModel, entryRadius):
        #Radius of the MSM domain
        self.radius = radius
        self.dim = p.position.size
        self.p = p
        self.timestep = timestep
        self.entryTrajsStart = truncTrajsModel.entryTrajsStart
        self.entryTrajsEnd = truncTrajsModel.entryTrajsEnd
        self.entryTimes = truncTrajsModel.entryTimes
        self.exitTrajs = truncTrajsModel.exitTrajs
        self.exitTimes = truncTrajsModel.exitTimes
        self.exitProbs = truncTrajsModel.exitProbs
        self.MSMtimestep = truncTrajsModel.MSMtimestep
        self.NangularBins = truncTrajsModel.NangularBins
        self.entryRadius = entryRadius
        self.NentryTrajs = len(self.entryTrajsStart)
        self.MSM = MSM
        self.clock = 0.
        self.sigma = np.sqrt(2.*self.p.D*self.timestep)
        self.sampleSize = 5 #sample consists of (step, time, p, MSMstate)
        self.MSMactive = False
        self.lastState = -1
        self.transition = False

    def above_threshold(self, threshold):
        #assume that threshold is larger than the MSM radius
        if self.MSMactive:
            return True
            #return np.linalg.norm(self.MSM.centers[self.MSM.state]) < threshold
        else:
            return np.linalg.norm(self.p.position) < threshold

    def propagateDiffusion(self, particle):
        #use inversion on circle to keep the particle inside of the simulation radius
        #see https://de.wikipedia.org/wiki/Kreisspiegelung for details
        dr = np.random.normal(0., self.sigma, self.dim)
        newPosition = particle.position + dr
        rNew = np.linalg.norm(newPosition)
        if rNew >= self.radius:
            newPosition = newPosition*self.radius**2/(rNew**2)
        particle.position = newPosition

    def enterMSMold(self):
        #assign closest state as entry state in MSM domain use Gaussian distance as metric
        #this might be really slow when having a ton of entry states
        # look up bin to search for entry position
        angularBin = (np.arctan2(self.p.position[1], self.p.position[0]) + np.pi)*self.NangularBins/(2*np.pi)
        angularBin = int(angularBin)
        # compute Gaussian weighted distance to determine closest entry position
        dist = np.linalg.norm(self.entryTrajsStart[angularBin] - self.p.position, axis=1)
        weights = np.exp(-dist*dist/(2*self.sigma*self.sigma))
        weights /= np.sum(weights)
        # select entry trajectory
        entryTraj = np.random.choice(len(weights), p=weights)
        state = self.entryTrajsEnd[angularBin][entryTraj]
        # determine whether entry position is a state or position
        if isinstance(state, np.ndarray):
            self.p.position = state
            self.MSMactive = False
        else:
            self.MSM.state = state
            self.MSMactive = True
            self.lastState = self.MSM.state
            self.transition = True
        # propagate clock
        self.clock += self.entryTimes[angularBin][entryTraj]*self.timestep
        self.MSM.exit = False

    def enterMSM(self):
        #assign closest state as entry state in MSM domain use Gaussian distance as metric
        #this might be really slow when having a ton of entry states
        # look up bin to search for entry position
        angularBin = (np.arctan2(self.p.position[1], self.p.position[0]) + np.pi)*self.NangularBins/(2*np.pi)
        angularBin = int(angularBin)
        # compute Gaussian weighted distance to determine closest entry position
        dist = np.linalg.norm(self.entryTrajsStart[angularBin] - self.p.position, axis=1)
        entryTraj = np.argmin(dist)
"""
        weights = np.exp(-dist*dist/(2*self.sigma*self.sigma))
        weights /= np.sum(weights)
        # select entry trajectory
        entryTraj = np.random.choice(len(weights), p=weights)
        """
        state = self.entryTrajsEnd[angularBin][entryTraj]
        # determine whether entry position is a state or position
        if isinstance(state, np.ndarray):
            self.p.position = state
            self.MSMactive = False
        else:
            self.MSM.state = state
            self.MSMactive = True
            self.lastState = self.MSM.state
            self.transition = True
        # propagate clock
        self.clock += self.entryTimes[angularBin][entryTraj]*self.timestep
        self.MSM.exit = False

    def exitMSM(self):
        exitTimeIndex = np.random.choice(len(self.exitTimes[self.MSM.state]))
        exitTime = self.exitTimes[self.MSM.state][exitTimeIndex] * self.timestep
        self.p.position = self.exitTrajs[self.MSM.state][exitTimeIndex]
        self.clock += exitTime
        self.MSMactive = False
        self.MSM.exit = False
        self.transition = False

    def integrate(self):
        if self.MSMactive:
            if self.transition:
                # transition occured, check for exit event
                if np.random.rand() <= self.exitProbs[self.MSM.state]:
                    self.exitMSM()
                else:
                    self.MSM.propagate()
                    if self.MSM.state != self.lastState:
                        self.transition = True
                        self.lastState = self.MSM.state
                    else:
                        self.transition = False
                    self.clock += self.MSMtimestep
            else:
                self.MSM.propagate()
                if self.MSM.state != self.lastState:
                    self.transition = True
                    self.lastState = self.MSM.state
                self.clock += self.MSMtimestep
        else: #MSM not active
            self.lastPosition = self.p.position
            self.propagateDiffusion(self.p)
            if np.linalg.norm(self.p.position) < self.entryRadius:
                self.enterMSM()
            else:
                self.clock += self.timestep

    def sample(self, step):
        if self.MSMactive:
            return [step, self.clock, 0., 0., self.MSM.state]
        else:
            return [step, self.clock, self.p.position[0], self.p.position[1], -1]
