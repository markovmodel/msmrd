import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial
import pickle

MFPTS = np.zeros([9,9])
minima = [[-0.9,0.7,0.3] ,  [-0.1,0.9,0.7],  [0.8,0.8,-0.8],  \
          [-1.0,-0.3,-0.4], [0.0,0.0,0.0],   [0.9,-0.1,-0.9], \
          [-0.7,-1.0,-0.3], [0.0,-0.9,0.1],  [0.8,-0.2,0.8]]

def run_mfpts(statePair, runs, scalef, dt):
    if statePair[0] == statePair[1]:
        return 0.
    global MFPTS, minima
    np.random.seed()
    asympot3D = potentials.asym3Dpotential(scalefactor = scalef)
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    z0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0, z0])
    p1 = mrd.particle(r1, 1.0)
    sphereboundary = mrd.reflectiveSphere(4.)
    integrator = integrators.brownianDynamicsSp(asympot3D, sphereboundary, p1, dt, 1.0)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.pa.position = np.array(minima[statePair[0]])
        fpts.append(sim.run_mfpt_point(np.array(minima[statePair[1]]), 0.2))
    pickle.dump(np.array(fpts), open('../data/asym3D/MFPTS/'+str(statePair[0])+'to'+str(statePair[1])+'_'+str(runs)+'runs_' + str(dt) + 'dt_' + str(scalef) 'sf.p', 'wa'))
    return np.mean(fpts)

statePairs = []
for i in range(9):
    for j in range(9):
        statePairs.append((i,j))

pool = Pool(processes=8)
MFPT_list = pool.map(partial(run_mfpts, runs=10000, scalef=2, dt=0.001), statePairs)

for i in range(9):
    for j in range(9):
        MFPTS[i,j] = MFPT_list[i*9+j]

'''
#run_simulation(2)
processes = []
print("Simulation started")
for j in range(0, 2):
    for i in range(0, 4):
	print("Process " + str(i+4*j) + " running")
        p = Process(target = run_mfpts, args=(i+4*j,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
'''

#pickle.dump(MFPTS, open('mfpts_10runs_01dt.p', 'wa'))
