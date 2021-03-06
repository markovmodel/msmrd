import numpy as np
from ..integrator import integrator


class brownianDynamicsSp(integrator):
    def __init__(self,  potential, boundary, p1, timestep, temp):
        self.potential = potential
        self.boundary = boundary
        self.dim = p1.position.size
        self.pa = p1
        self.timestep = timestep
        self.clock = 0.
        self.temp = temp
        self.sigmaA = np.sqrt(2 * self.timestep * self.pa.D)
        self.traj = None
        self.sampleSize = self.dim + 1 #sample has shape (time, position vector)

    def integrate(self):
        #compute force from particle b on particle a
        force = self.potential.force(self.pa.position)
        dr1 = np.random.normal(0., self.sigmaA, self.dim) + force * self.timestep * self.pa.D / self.temp
	oldpos = 1.0*self.pa.position
        self.pa.position += dr1
        self.boundary.reduce(self.pa,oldpos)
        self.clock += self.timestep

    def sample(self, step):
	if self.dim == 1:
        	return [step, self.pa.position[0]]
	if self.dim == 2:
        	return [step, self.pa.position[0], self.pa.position[1]]
	if self.dim == 3:
        	return [step, self.pa.position[0], self.pa.position[1], self.pa.position[2]]

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

    def above_threshold(self, threshold):
        return np.linalg.norm(self.pa.position) < threshold

    def outside_radius(self, point, radius):
        return np.linalg.norm(self.pa.position - point) > radius
