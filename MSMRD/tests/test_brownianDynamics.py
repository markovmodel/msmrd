import numpy as np
from unittest import TestCase
import MSMreaddy as mrd
import MSMreaddy.integrators as integrators


class MSMreaddyTest(TestCase):
    def test_initialization(self):
        testT = np.array([[0.3, 0.7], [0.1, 0.9]])
        testcc = np.array([[2., 1.], [1.,2.]])
        np.savetxt('testmsm_Tmatrix.txt', testT)
        np.savetxt('testmsm_centers.txt', testcc)
        pos1 = np.array([0.1, 1.])
        pos2 = np.array([1., 0.5])
        p1 = mrd.particle(pos1, 1.0)
        p2 = mrd.particle(pos2, 2.0)
        p3 = mrd.particle(pos2, 3.0)
        box = mrd.box(10.)
        msm = mrd.MSM('testmsm')
        integrator = integrators.MSMreaddy(msm, box, p1, p2, p3, 0.1, 4.0)
        sim = mrd.simulation(integrator)
        sim.run(100)
        self.assertTrue(True)