import numpy as np
from ..integrator import integrator


class brownianDynamicsSp(integrator):
    def __init__(self,  potential, boundary, p1, timestep, temp):
        self.potential = potential
        self.boundary = boundary
        self.dim = p1.position.size
        self.pa = p1
        self.timestep = timestep
        self.time = 0.
        self.temp = temp
        self.sigmaA = np.sqrt(2 * self.timestep * self.pa.D)
        self.traj = None
        self.sampleSize = 3 #sample has shape (time, reducedDistanceVector, energy)
        print 'new version'

    def integrate(self):
        #compute force from particle b on particle a
        force = self.potential.force(self.pa.position)
        dr1 = np.random.normal(0., self.sigmaA, self.dim) + force * self.timestep * self.pa.D / self.temp
        self.pa.position += dr1
        self.boundary.reduce(self.pa)
        self.time += self.timestep

    def sample(self, step):
        return [step, self.pa.position[0], self.pa.position[1]]

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