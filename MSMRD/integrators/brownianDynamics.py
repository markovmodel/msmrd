import numpy as np
from ..integrator import integrator


class brownianDynamics(integrator):
    def __init__(self,  potential, box, p1, p2, timestep, temp):
        self.potential = potential
        self.box = box
        self.dim = p1.position.size
        self.pa = p1
        self.pb = p2
        self.timestep = timestep
        self.temp = temp
        self.sigmaA = np.sqrt(2 * self.timestep * self.pa.D)
        self.sigmaB = np.sqrt(2 * self.timestep * self.pb.D)
        self.traj = None
        self.sampleSize = 3 #sample has shape (time, reducedDistanceVector, energy)

    def above_threshold(self, threshold):
        return self.box.particleDistance(self.pa, self.pb) > threshold

    def integrate(self):
        #reduced vector from particle b to particle a
        r = self.box.periodicDistanceVector(self.pb.position, self.pa.position)
        #compute force from particle b on particle a
        force = self.potential.force(r)
        dr1 = np.random.normal(0., self.sigmaA, self.dim) + force * self.timestep * self.pa.D / self.temp
        dr2 = np.random.normal(0., self.sigmaB, self.dim) - force * self.timestep * self.pb.D / self.temp
        self.pa.position += dr1
        self.box.reduce(self.pa)
        self.pb.position += dr2
        self.box.reduce(self.pb)

    def sample(self, step):
        dr = self.box.periodicDistanceVector(self.pb.position, self.pa.position)
        return [step*self.timestep, dr[0], dr[1]]

    def compute_stationary_distribution(self, traj, MSM):
        #computes the stationary distribution clustered onto cluster centers of the MSM
        distance_vectors = traj[:, 1:3]
        distances = np.linalg.norm(distance_vectors, axis=1)
        msm_region = np.where(distances < MSM.MSMradius)[0]
        clusters = MSM.allocateStates(distance_vectors[msm_region, ...])
        counts = np.zeros(MSM.states)
        for i in range(0, MSM.states):
            counts[i] += np.where(clusters == i)[0].size
        #normalize
        counts /= float(counts.sum())
        return counts