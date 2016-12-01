import numpy as np

class modifiedLJ:
    def __init__(self, a=1., b=3., c=2., epsilon=2., sigma= 1.):
        self.a = a
        self.b = b
        self.c = c
        self.epsilon = epsilon
        self.sigma = sigma
        self.sig3 = pow(self.sigma, 3)

    def potential(self, r):
        R = np.linalg.norm(r)
        Rinv = 1./R
        Rinv3 = Rinv**3
        Rinv6 = Rinv3**2
        return self.epsilon * self.sig3**2 * Rinv6 * (self.a * self.sig3**2 * Rinv6 - self.b * self.sig3 * Rinv3 + self.c )

    def force(self, r):
        R = np.linalg.norm(r)
        Rinv = 1./R
        rhat = r * Rinv
        Rinv3 = Rinv**3
        Rinv7 = Rinv3**2 * Rinv
        return -rhat * self.epsilon * self.sig3**2 * Rinv7 * (-12 * self.a * self.sig3**2 * Rinv3**2 + 9 * self.b * self.sig3 * Rinv3 - 6 * self.c )