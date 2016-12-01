import numpy as np
from unittest import TestCase
import MSMreaddy as mrd
import MSMreaddy.integrators as integrators
import MSMreaddy.potentials as potentials


class BrownianDynamicsTest(TestCase):
    def test_initialization(self):
        pos1 = np.array([0.1, 1.])
        pos2 = np.array([1., 0.5])
        p1 = mrd.particle(pos1, 1.0)
        p2 = mrd.particle(pos2, 2.0)
        box = mrd.box(10.)
        potential = potentials.modifiedLJ()
        integrator = integrators.brownianDynamics(potential, box, p1, p2, 0.1, 1.0)
        sim = mrd.simulation(integrator)
        sim.run(100)
        self.assertTrue(True)