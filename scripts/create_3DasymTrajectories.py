import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process

def run_simulation(runNumber):
    global correlation, runtimes
    np.random.seed()
    runtime = int(1e7)
    dt = 0.001
    scalef = 2.0
    sampInterval = 10
    asympot = potentials.asym3Dpotential(scalefactor = scalef)
    x0 = 4.0*np.random.rand() - 2.0
    y0 = 4.0*np.random.rand() - 2.0
    z0 = 4.0*np.random.rand() - 2.0
    r1 = np.array([x0, y0, z0])
    p1 = mrd.particle(r1, 1.0)
    boundary = mrd.box(5.0,3) #mrd.reflectiveSphere(4.)
    integrator = integrators.brownianDynamicsSp(asympot, boundary, p1, dt, 1.0)
    sim = mrd.simulation(integrator)
    outfile = '../data/asym3D/3DasymTrajs_pbox5_RT1e7_'+ dt + 'dt' + sampInterval '_si_' + str(runNumber)+ '.h5'
    sim.run_n_buffer(runtime, sample=True, samplingInterval=sampInterval, \
                     filename = outfile, buffersize = int(1e3))

#run_simulation(2)

processes = []
print("Simulation started")
for j in range(0, 15):
    for i in range(0, 4):
	print("Process " + str(i+4*j) + " running")
        p = Process(target = run_simulation, args=(i+4*j,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
