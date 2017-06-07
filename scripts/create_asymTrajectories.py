import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process

def run_simulation(runNumber):
    global correlation, runtimes
    np.random.seed()
    runtime = int(1e7)
    asympot = potentials.asym2Dpotential(scalefactor = 2.0)
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0])
    p1 = mrd.particle(r1, 1.0)
    box = mrd.box(6., 2)
    integrator = integrators.brownianDynamicsSp(asympot, box, p1, 0.0001, 1.0)
    sim = mrd.simulation(integrator)
    outfile = '/srv/data/dibakma/MSMRD/asym2D/BDdata/box/2Dasym_B6_RT1E7_dt1E-4_SI10_SF2_'+ str(runNumber)+ '.h5'
    sim.run_n_buffer(runtime, sample=True, samplingInterval=10, \
                     filename = outfile, buffersize = int(1e3))

#run_simulation(2)
processes = []
print("Simulation started")
for j in range(0, 20):
    for i in range(0, 4):
	print("Process " + str(i+4*j) + " running")
        p = Process(target = run_simulation, args=(i+4*j,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
numruns = 51 
for runs in range(1):
    run_simulation(runs)
    print runs
    """
