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

    def run_n_buffer(self, steps, sample = False, samplingInterval = 1, \
	             filename="../data/trajs_data.txt", buffersize = int(1e5)):
	numbuffer = 0
	bufcounter = 0
        samples = int(steps / samplingInterval) + 1
        self.traj = np.zeros((buffersize, self.integrator.sampleSize))
	# open blank file and allow appending for buffer mode	
	open(filename, "w")
	f = open(filename, "a")
	# loop over all steps and write file when buffer is filled
        for i in range(steps):
            self.integrator.integrate()
            if sample:
                if not i % samplingInterval:
		    bufcounter = bufcounter + 1 
                    j = i / samplingInterval - numbuffer*buffersize
                    self.traj[j,:] = self.integrator.sample(i)
	    # empty to file when buffer is full and reinitialize traj array
	    if (bufcounter == buffersize):
	    	bufcounter = 0
		numbuffer = numbuffer + 1
		np.savetxt(f, self.traj)
                a = [1,2,3]
		np.savetxt(f, a)
		ransteps = numbuffer*buffersize*samplingInterval
		if (steps - ransteps) < (buffersize*samplingInterval):
			#lastarray = int((steps - ransteps)/samplingInterval) + 1
			lastarray = samples%buffersize
			self.traj = np.zeros((lastarray, self.integrator.sampleSize))
		else:
			self.traj = np.zeros((buffersize, self.integrator.sampleSize))
	np.savetxt(f,self.traj)

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
