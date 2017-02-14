import numpy as np

class particle:
    def __init__(self, r0, D):
        self.position = r0
        self.D = D
        self.image = np.zeros(r0.size)