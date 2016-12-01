import numpy as np


class box:
    def __init__(self, boxsize):
        if not boxsize > 0:
            raise ValueError('invalid box size')
        self.boxsize = boxsize
        self.image = np.zeros(3)

    def _reducePeriodic(self, r):
        assert isinstance(r, np.ndarray)
        assert len(r.shape) == 1
        dim = r.shape[0]
        for i in range(0, dim):
            if r[i] >= self.boxsize / 2.:
                r[i] -= self.boxsize
            if r[i] < - self.boxsize / 2.:
                r[i] += self.boxsize
        return r

    def reducePeriodic(self, particle):
        r = self._reducePeriodic(particle.position)
        particle.position = r

    def _periodicDistance(self, dr):
        assert(dr.shape == (2,))
        dr = self._reducePeriodic(dr)
        return np.linalg.norm(dr)

    def periodicDistance(self, r1, r2):
        assert isinstance(r1, np.ndarray)
        assert isinstance(r2, np.ndarray)
        dr = r1 - r2
        if len(dr.shape) == 2:
            ltraj = dr.shape[0]
            distances = np.zeros(ltraj)
            for i in range(0, ltraj):
                distances[i] = self._periodicDistance(dr[i,:])
            return distances
        if len(dr.shape) == 1:
            return self._periodicDistance(dr)

    def particleDistance(self, particle1, particle2):
        dr = particle1.position - particle2.position
        return self._periodicDistance(dr)

    def periodicDistanceVector(self, r1, r2):
        dr = r2-r1
        if len(dr.shape) == 2:
            ltraj = dr.shape[0]
            dim = dr.shape[1]
            distances = np.zeros([ltraj, dim])
            for i in range(0,ltraj):
                distances[i,:] = self._reducePeriodic(dr[i,:])
            return distances
        elif len(dr.shape) == 1:
            return self._reducePeriodic(dr)