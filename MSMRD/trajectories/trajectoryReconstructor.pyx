import numpy as np
cimport numpy as np

cdef class trajReconstructor:
    cdef public double innerRadius, outerRadius, outerRadius2, timeStep, diff, sigma
    cdef public int dim
    def __init__(self, double innerRadius, double outerRadius, double timeStep, double diff, int dim):
        self.innerRadius = innerRadius
        self.outerRadius = outerRadius
        self.outerRadius2 = self.outerRadius*self.outerRadius
        self.timeStep = timeStep
        self.diff = diff
        self.sigma = np.sqrt(2*self.timeStep*self.diff)
        self.dim = dim

      
    cpdef reconstructTraj(self, np.ndarray[np.float64_t, ndim=1] initialPosition):
        cdef np.ndarray[np.float64_t, ndim=1] position = initialPosition
        cdef list outerTraj = [position]
        cdef int inBDRegion = 1
        cdef int refOutBoundary = 1
        cdef double rad
        while (inBDRegion == 1):
            position = self.integrate(position)
            rad = np.linalg.norm(position)
            outerTraj.append(position)
            if (rad <= self.innerRadius):
                inBDRegion = 0
                return outerTraj
            else:
                if (refOutBoundary == 1):
                    position = self.reduce(position)
                elif (rad >= self.outerRadius):
                    inBdRegion = 0
                    return outerTraj
                
        
    cdef integrate(self, np.ndarray[np.float64_t, ndim=1] position):
        cdef np.ndarray[np.float64_t, ndim=1] dr = np.random.normal(0., self.sigma, self.dim)
        return position + dr
    
    cdef reduce(self, np.ndarray[np.float64_t, ndim=1] position):
        cdef double r2 = np.linalg.norm(position)**2
        if r2 >= self.outerRadius2:
            position = position * self.outerRadius2/r2
        return position