import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process

def run_simulation(runNumber):
    global correlation, runtimes
    runtime = int(1e7)
    asympot = potentials.asym2Dpotential()
    r1 = np.array([2., 2.])
    p1 = mrd.particle(r1, 1.0)
    box = mrd.box(8.)
    np.random.seed()
    integrator = integrators.brownianDynamicsSp(asympot, box, p1, 0.0001, 1.0)
    sim = mrd.simulation(integrator)
    outfile = '../data/asym2D/2DasymTrajsLong_'+ str(runNumber)+ '.txt'
    sim.run_n_buffer(runtime, sample=True, samplingInterval=100, \
                     filename = outfile, buffersize = int(1e5))

#run_simulation(2)

processes = []
print("Simulation started")
for j in range(0, 4):
    for i in range(0, 4):
	print("Process " + str(i+4*j) + " running")
        p = Process(target = run_simulation, args=(i+4*j,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
