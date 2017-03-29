import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial
import pickle

MFPTS = np.zeros([9,9])
minima = [[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]]

def run_mfpts(statePair, runs):
    if statePair[0] == statePair[1]:
        return 0.
    global MFPTS, minima
    np.random.seed()
    asympot = potentials.asym2Dpotential(scalefactor = 0.7)
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0])
    p1 = mrd.particle(r1, 1.0)
    ringboundary = mrd.reflectiveRing(4.)
    integrator = integrators.brownianDynamicsSp(asympot, ringboundary, p1, 0.01, 1.0)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.pa.position = np.array(minima[statePair[0]])
        fpts.append(sim.run_mfpt_point(np.array(minima[statePair[1]]), 0.2))
    pickle.dump(np.array(fpts), open('../data/asym2D/MFPTS/'+str(statePair[0])+'to'+str(statePair[1])+'_'+str(runs)+'runs.p', 'wa'))
    return np.mean(fpts)

statePairs = []
for i in range(9):
    for j in range(9):
        statePairs.append((i,j))

pool = Pool(processes=8)
MFPT_list = pool.map(partial(run_mfpts, runs=10000), statePairs)

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
