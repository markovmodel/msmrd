import numpy as np


class box:
    def __init__(self, boxsize, dim):
        if not boxsize > 0:
            raise ValueError('invalid box size')
        self.boxsize = boxsize
        self.image = np.zeros(3)
        self.dim = dim

    def _reducePeriodic(self, r):
        assert isinstance(r, np.ndarray)
        assert len(r.shape) == 1
        for i in range(0, self.dim):
            if r[i] >= self.boxsize / 2.:
                r[i] -= self.boxsize
            if r[i] < - self.boxsize / 2.:
                r[i] += self.boxsize
        return r

    def reduce(self, particle, oldpos):
        r = self._reducePeriodic(particle.position)
        particle.position = r

    def _periodicDistance(self, dr):
        assert(dr.shape == (self.dim,))
        dr = self._reducePeriodic(dr)
        return np.linalg.norm(dr)

    def periodicDistance(self, r1, r2):
        #this function returns the reduced distance between r1 and r2
        #r1 and r2 might be coordinates or arrays of coordinates
        assert isinstance(r1, np.ndarray)
        assert isinstance(r2, np.ndarray)
        dr = r1 - r2
        if len(dr.shape) == 2:
            #in this case dr is an array of distances
            ltraj = dr.shape[0]
            distances = np.zeros(ltraj)
            for i in range(0, ltraj):
                distances[i] = self._periodicDistance(dr[i,:])
            return distances
        if len(dr.shape) == 1:
            #dr is one distance
            return self._periodicDistance(dr)

    def particleDistance(self, particle1, particle2):
        dr = particle1.position - particle2.position
        return self._periodicDistance(dr)

    def periodicDistanceVector(self, r1, r2):
        #returns the distance vector between r1 and r2
        #r1 and r2 may be arrays of coordinates
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

class reflectiveRing:
    def __init__(self, radius):
        self.radius2 = radius**2

    #perform inversion on a circle to map the particle's position back into the domain
    def reduce(self, particle, oldpos):
        r2 = np.linalg.norm(particle.position)**2
        if r2 > self.radius2:
            particle.position = particle.position * self.radius2/r2

class reflectiveSphere:
    def __init__(self, radius):
        self.radius2 = radius**2

    #do reflective boundary condition
    def reduce(self, particle, oldpos):
        r2 = np.linalg.norm(particle.position)**2
        if r2 > self.radius2:
            # Create vector between old and new position
            r0 = oldpos
            dr = particle.position - r0
            if np.linalg.norm(dr) == 0:
                return
            # Calculate intersection point (r0 + al*dr intersection with sphere)
            A = np.dot(dr,dr)
            B = 2.0*np.dot(r0,dr)
            C = np.dot(r0,r0) - self.radius2
            discriminant = B**2 - 4.0*A*C
            if  discriminant <= 0: 	    
                print A, B, C, r0, B**2 - 4.0*A*C
                discriminant = 0
            al = (-B + np.sqrt(discriminant))/(2.0*A) # take only the not always negative root
            intpt = r0 + al*dr
            # Calculate normal to sphere at intersection point
            nvec =  intpt/np.linalg.norm(intpt)
            # Norm of reflected vector
            # Calculate reflected vector (subtract normal projection twice and rescale)
            vref = dr - 2*np.dot(dr,nvec)*nvec
            refnorm = np.linalg.norm(r0 + dr - intpt)
            vref = refnorm*vref/np.linalg.norm(vref)
            # Translate reflected vector to intersection point and update
            ptref = intpt + vref
            particle.position = ptref

























