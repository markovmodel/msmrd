import numpy as np
from ..integrator import integrator
from ..MSM import MSM
from ..discretization import  getSectionNumber, getAngles


class MSMRDtruncTrajs3D(integrator):
    def __init__(self, MSM, radius, particle, timestep, entryRadius, boundary, truncTrajsModel3D):
        # Imported class variables
        self.MSM = MSM
        self.radius = radius
        self.particle = particle
        self.timestep = timestep
        self.entryRadius = entryRadius
        self.boundary =  boundary
        self.entryTrajsStart = truncTrajsModel3D.entryTrajsStart
        self.entryTrajsEnd = truncTrajsModel3D.entryTrajsEnd
        self.entryTimes = truncTrajsModel3D.entryTimes
        self.exitTrajs = truncTrajsModel3D.exitTrajs
        self.exitTimes = truncTrajsModel3D.exitTimes
        self.exitProbs = truncTrajsModel3D.exitProbs
        self.MSMtimestep = truncTrajsModel3D.MSMtimestep
        self.numPartitions = truncTrajsModel3D.numPartitions
        # Derived class variables
        self.dim = self.particle.position.size
        self.NentryTrajs = len(self.entryTrajsStart)
        self.clock = 0.
        self.sigma = np.sqrt(2.*self.particle.D*self.timestep)
        self.sampleSize = 6 #sample consists of (step, time, p, MSMstate)
        self.MSMactive = False
        self.lastState = -1
        self.transition = False
        self.entryCalls = 0
        self.exitCalls = 0

    def above_threshold(self, threshold):
        #assume that threshold is larger than the MSM radius
        if self.MSMactive:
            return True
            #return np.linalg.norm(self.MSM.centers[self.MSM.state]) < threshold
        else:
            return np.linalg.norm(self.particle.position) < threshold

    def propagateDiffusion(self):
        #use boundary conditions from boundary
        dr = np.random.normal(0., self.sigma, self.dim)
        oldpos = self.particle.position
        newPosition = self.particle.position + dr
        rNew = np.linalg.norm(newPosition)
	self.particle.position = newPosition
        if rNew >= self.radius:
            self.boundary.reduce(self.particle, oldpos)

    #assign closest state as entry state in MSM domain
    def enterMSM(self):
        self.entryCalls += 1
        # find partition bin where particle is entering to 
        sectionNum = getSectionNumber(self.particle.position, self.numPartitions)
        # choose closest entry position from trajectories in bin
        dist = np.linalg.norm(self.entryTrajsStart[sectionNum-1] - self.particle.position, axis=1)
        entryTraj = np.argmin(dist)
        state = self.entryTrajsEnd[sectionNum-1][entryTraj]
        # determine whether entry position is a state or position
        if isinstance(state, np.ndarray):
            self.particle.position = state
            self.MSMactive = False
        else:
            self.particle.position = np.zeros(3)
            self.MSM.state = state
            self.MSMactive = True
            self.lastState = self.MSM.state
            self.transition = True
        # propagate clock
        self.clock += self.entryTimes[sectionNum-1][entryTraj] * self.timestep
        self.MSM.exit = False

    #assign closest state as entry state in MSM domain use Gaussian distance as metric
    def enterMSMGauss(self):
        self.entryCalls += 1
        # look up bin to search for entry position
        sectionNum = getSectionNumber(self.particle.position, self.numPartitions)
        # compute Gaussian weighted distance to determine closest entry position
        dist = np.linalg.norm(self.entryTrajsStart[sectionNum-1] - self.particle.position, axis=1)
        weights = np.exp(-dist*dist/(2.0*self.sigma*self.sigma))
        weights /= np.sum(weights)
        # select entry trajectory
        entryTraj = np.random.choice(len(weights), p=weights)
        state = self.entryTrajsEnd[sectionNum-1][entryTraj]
        # determine whether entry position is a state or position
        if isinstance(state, np.ndarray):
            self.particle.position = state
            self.MSMactive = False
        else:
            self.MSM.state = state
            self.MSMactive = True
            self.lastState = self.MSM.state
            self.transition = True
        # propagate clock
        self.clock += self.entryTimes[sectionNum-1][entryTraj]*self.timestep
        self.MSM.exit = False

    # Assign position to particle when exiting MSM using trajectory statistics
    def exitMSM(self):
        self.exitCalls += 1
        exitTimeIndex = np.random.choice(len(self.exitTimes[self.MSM.state]))
        exitTime = self.exitTimes[self.MSM.state][exitTimeIndex] * self.timestep
        self.particle.position = self.exitTrajs[self.MSM.state][exitTimeIndex]
        self.clock += exitTime
        self.MSMactive = False
        self.MSM.exit = False
        self.transition = False

    # Integrate hybrid model over time
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
            self.lastPosition = self.particle.position
            self.propagateDiffusion()
            if np.linalg.norm(self.particle.position) < self.entryRadius:
                self.enterMSM()
            else:
                self.clock += self.timestep

    # test sample
    def sample(self, step):
        if self.MSMactive:
            return [step, self.clock, 0., 0., 0.,  self.MSM.state]
        else:
            return [step, self.clock, self.particle.position[0], self.particle.position[1],  self.particle.position[2], -1]
