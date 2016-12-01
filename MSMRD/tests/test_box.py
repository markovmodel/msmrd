import numpy as np
from unittest import TestCase
import MSMreaddy as mrd

class BoxTest(TestCase):
    def test_periodic_distance_vector(self):
        box = mrd.box(10.)
        x = np.arange(-20., 20., 0.2)
        distances = np.zeros([2, x.size*x.size])
        distances0 = np.zeros([2, x.size*x.size])
        reference = np.zeros([2, x.size*x.size])
        for i in range(0, x.size):
            for j in range(0, x.size):
                distances[0, i+j*x.size] = x[i]
                distances[1, i+j*x.size] = x[j]
                reference[0, i+j*x.size] = x[i]
                if x[i] >= box.boxsize/2.:
                    reference[0, i+j*x.size] = x[i] - box.boxsize
                elif x[i] < -box.boxsize/2.:
                    reference[0, i+j*x.size] = x[i] + box.boxsize
                reference[1, i+j*x.size] = x[j]
                if x[j] >= box.boxsize/2.:
                    reference[1, i+j*x.size] = x[j] - box.boxsize
                if x[j] < -box.boxsize/2.:
                    reference[1, i+j*x.size] = x[j] + box.boxsize
        computed = box.periodicDistanceVector(distances0, distances)
        print computed[computed != reference] - reference[computed != reference]
        self.assertTrue(np.all(computed == reference))