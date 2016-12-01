import numpy as np
from MSMRD.integrator import integrator as MSMRD_integrator

class simulation:
    def __init__(self, integrator):
        assert isinstance(integrator, MSMRD_integrator)
        self.integrator=integrator

    def run(self, steps, sample = False, samplingInterval = 1):
        samples = int(steps / samplingInterval)
        self.traj = np.zeros((samples, self.integrator.sampleSize))
        for i in range(0,steps):
            self.integrator.integrate()
            if sample:
                if not i % samplingInterval:
                    j = i / samplingInterval
                    self.traj[j,:] = self.integrator.sample(i)

    def run_mfpt(self, threshold):
        i = 0
        while self.integrator.above_threshold(threshold):
            self.integrator.integrate()
            i+=1
        return i*self.integrator.timestep


"""
    def histogramTransition(self, bins):
        if self.traj is None:
            raise ValueError('No simulation data. Run simulation first')
        else:
            #cluster data in transition area
            #extract BD part of trajectory
            BDidcs = np.where(self.traj[:,7] == -1)[0]
            BDtraj = self.traj[BDidcs, ...]
            r1 = BDtraj[:, 1:3]
            r2 = BDtraj[:, 1:3]
            #compute periodically reduces distance and find points in transition region
            distances = np.zeros(BDidcs[0].size)
            distances = self.box.periodicDistance(r1, r2)
            transitionRegion = np.where(distances < self.MSM.MSMradius)[0]
            dr = self.box.periodicDistanceVector(r1, r2)
            self.histogram, self.xedges, self.yedges = np.histogram2d(dr[transitionRegion, 0], dr[transitionRegion,1], bins=bins)
            """