import numpy as np
import h5py
from MSMRD.integrator import integrator as MSMRD_integrator

class simulation:
    def __init__(self, integrator):
        assert isinstance(integrator, MSMRD_integrator)
        self.integrator=integrator

    def run(self, steps, sample = False, samplingInterval = 1):
        samples = np.ceil(1.0*steps / samplingInterval)
        self.traj = np.zeros((samples, self.integrator.sampleSize))
        for i in range(0,steps):
            self.integrator.integrate()
            if sample:
                if not i % samplingInterval:
                    j = i / samplingInterval
                    self.traj[j,:] = self.integrator.sample(i)
    
    # run code using a buffer to avoid computer overload while writing into file filename
    def run_n_buffer(self, steps, sample = False, samplingInterval = 1, filename="../data/trajs_data.txt", buffersize = int(1e5)):
        # init number of current buffer and index of current buffer
        numbuffer = 0
        bufindex = 0
        # obtain total number of samples and number of buffers
        samples = np.ceil(1.0*steps / samplingInterval)
        if samples < buffersize:
            buffersize = int(samples)
        lastbuffer = np.ceil(1.0*samples / buffersize)
        # init traj array with maxsize as buffersize
        self.traj = np.zeros((buffersize, self.integrator.sampleSize))
        # open blank file and allow appending for buffer mode
        open(filename, "w")
        #create and open file, overwrite if it exists
        f = h5py.File(filename, 'a')
        f.create_dataset('traj', (samples, self.integrator.sampleSize))
        dset = f['traj']
        # loop over all steps and and write array to file when buffer is filled
        for i in range(steps):
            self.integrator.integrate()
            if sample:
                if not i % samplingInterval:
                    bufindex = bufindex + 1
                    j = i / samplingInterval - numbuffer*buffersize
                    self.traj[j,:] = self.integrator.sample(i)
        # empty to file when buffer is full and reinitialize traj array
            if bufindex == buffersize:
                bufindex = 0
                dset[numbuffer*buffersize:(numbuffer+1)*buffersize] = self.traj
                numbuffer = numbuffer + 1
                if numbuffer == lastbuffer - 1:
                    lastarray = int(samples - buffersize*numbuffer)
                    self.traj = np.zeros((lastarray, self.integrator.sampleSize))
                elif numbuffer != lastbuffer:
                    self.traj = np.zeros((buffersize, self.integrator.sampleSize))
        # if last buffer size < buffersize, empty left over data to file
        if (bufindex != 0):
            dset[-len(self.traj):] = self.traj
        f.close()

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
