import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

minima = [[-0.9,0.7,0.3] ,  [-0.1,0.9,0.7],  [0.8,0.8,-0.8],  \
          [-1.0,-0.3,-0.4], [0.0,0.0,0.0],   [0.9,-0.1,-0.9], \
          [-0.7,-1.0,-0.3], [0.0,-0.9,0.1],  [0.8,-0.2,0.8]]

R = 2.5
suffix = str(R)+'R_3D_internalDynamics_benchmark_dt0.001_SF2.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/asym3D/MFPTS/internal/' 


def run_exit_times(statePair, runs):
    np.random.seed()
    asympot = potentials.asym3Dpotential(scalefactor = 2.0)
    r1 = np.array(np.zeros(3))
    p1 = mrd.particle(r1, 1.0)
    sphereboundary = mrd.reflectiveSphere(4.)
    integrator = integrators.brownianDynamicsSp(asympot, sphereboundary, p1, 0.001, 1.0)
    sim = mrd.simulation(integrator)
    exitTimes = []
    exitPositions = []
    fpts = []
    exits = 0
    run = 0
    totalRuns = 0
    while run < runs:
        integrator.pa.position = np.array(minima[statePair[0]])
        integrator.clock = 0.
        totalRuns += 1
        while True:
            sim.integrator.integrate()
            if np.linalg.norm(minima[statePair[1]] - integrator.pa.position) < 0.2:
                fpts.append(integrator.clock)
                run += 1
                break
            elif np.linalg.norm(integrator.pa.position) > R:
                exits += 1
                exitTimes.append(integrator.clock)
                exitPositions.append(integrator.pa.position)
                break
    print str(statePair[0])+' to ' + str(statePair[1]) 
    return [np.array(fpts), totalRuns,  exitTimes, exitPositions]

statePairs = []
for i in range(9):
    for j in range(9):
        statePairs.append((i,j))
pool = Pool(processes=4)
runs = 10000
# RERUN for transition 2 to 5!
FPT_list = []
results = pool.map(partial(run_exit_times, runs=runs), statePairs)
FPTs = []
for i in range(9):
    fromi = []
    for j in range(9):
        fromi.append([])
    FPTs.append(fromi)
for i in range(len(statePairs)):
    FPTs[statePairs[i][0]][statePairs[i][1]].append(results[9*statePairs[i][0]+statePairs[i][1]][0])

MFPTs = np.zeros([9,9])
for i in range(9):
    for j in range(9):
        if i!=j:
            MFPTs[i,j] = np.mean(FPTs[i][j])

dill.dump(FPTs, open(path+'fpts_'+str(runs)+'_runs_fullRerun_'+suffix, 'wa' ))
dill.dump(MFPTs, open(path+'mfpts_'+str(runs)+'_runs_fullRerun_'+suffix, 'wa' ))
dill.dump(results,  open(path+'results_'+str(runs)+'_runs_fullRerun_'+suffix, 'wa' ))
