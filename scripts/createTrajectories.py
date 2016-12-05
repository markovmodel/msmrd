import numpy as np
import matplotlib.pyplot as plt
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from MSMRD.utils import voronoi
from MSMRD.utils import correlate
from multiprocessing import Process

def run_simulation(runNumber, R):
    global correlation, runtimes
    runtime = 100000000
    LJpot = potentials.modifiedLJ()
    r1 = np.array([0., 0.])
    r2 = R*np.array([np.cos(np.pi*(float(runNumber)/16.+1./32.)), np.sin(np.pi*(float(runNumber)/16.+1./32.))])
    p1 = mrd.particle(r1, 0.5)
    p2 = mrd.particle(r2, 0.5)
    box = mrd.box(8.)
    integrator = integrators.brownianDynamics(LJpot, box, p1, p2, 0.0001, 1.0)
    sim = mrd.simulation(integrator)
    sim.run(runtime, sample=True, samplingInterval=100)
    np.savetxt('/srv/data/dibakma/potentialMSM/2DmodifiedLJtraj/traj_eps2_box6_init_'+str(R)+'_'+str(runNumber)+'.txt', sim.traj)

processes = []
R = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
for Rinit in R:
    for j in range(0, 4):
        for i in range(0, 4):
            p = Process(target = run_simulation, args=(i+4*j, Rinit))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
