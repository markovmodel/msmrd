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
        self.entryRadius = entryRadius
        self.NentryTrajs = len(self.entryTrajsStart)
        self.MSM = MSM
        self.clock = 0.
        self.transitionTime = 0.
        self.sigma = np.sqrt(2.*self.p.D*self.timestep)
        self.sampleSize = 5 #sample consists of (step, time, p, MSMstate)
        self.MSMactive = False
        self.lastState = -1

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

    def enterMSM(self):
        #assign closest state as entry state in MSM domain use Gaussian distance as metric
        #this might be really slow when having a ton of entry states
        dist = np.linalg.norm(self.entryTrajsStart - self.p.position, axis=1)
        weights = np.exp(-dist*dist/(2*self.sigma*self.sigma))
        weights /= np.sum(weights)
        entryTraj = np.random.choice(len(weights), p=weights)
        state = self.entryTrajsEnd[entryTraj]
        if isinstance(state, np.ndarray):
            self.p.position = state
            self.MSMactive = False
        else:
            self.MSM.state = self.entryTrajsEnd[entryTraj]
            self.MSMactive = True
            self.lastTransition = self.MSM.state
            self.transitionTime = 0
        self.clock += self.entryTimes[entryTraj]
        self.MSM.exit = False

    def exitMSM(self):
        exitTrajs = np.where(self.exitTimes[self.MSM.state] > self.transitionTime * self.MSMtimestep)[0]
        exitTraj = np.random.choice(exitTrajs)
        exitPosition = self.exitTrajs[self.MSM.state][exitTraj]
        self.clock += self.exitTimes[self.MSM.state][exitTraj] - self.transitionTime
        self.p.position = exitPosition
        self.MSMactive = False
        self.MSM.exit = False

    def integrate(self):
        if self.MSMactive:
            if np.random.rand() <= self.exitProbs[self.MSM.state]:
                self.exitMSM()
            else:
                self.MSM.propagate()
                if self.MSM.state == self.lastTransition:
                    self.transitionTime += 1
                else:
                    self.lastTransition = self.MSM.state
                self.clock += self.MSMtimestep
        elif not self.MSMactive:
            self.lastPosition = self.p.position
            self.propagateDiffusion(self.p)
            if np.linalg.norm(self.p.position) < self.entryRadius:
                self.enterMSM()

    def sample(self, step):
        if self.MSMactive:
            return [self.clock, 0., 0., self.MSM.state]
        else:
            return [self.clock, self.p.position[0], self.p.position[1], -1]
