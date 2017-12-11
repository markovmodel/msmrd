import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

minima = np.array([[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]])

R = 2.5
suffix = str(R)+'R_3D_internalDynamics_hybrid_lag0.05_240partitions.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/asym3D/MFPTS/internal/' 

class truncTrajsModel3D(object):    
    def __init__(self, entryTrajsStart, entryTrajsEnd, entryTimes, exitTrajs, exitTimes, exitProbs, MSMtime, tmatrix, numPartitions):
        self.entryTrajsStart = entryTrajsStart
        self.entryTrajsEnd = entryTrajsEnd
        self.entryTimes = entryTimes
        self.exitTrajs = exitTrajs
        self.exitTimes = exitTimes
        self.exitProbs = exitProbs
        self.MSMtimestep = MSMtime
        self.tmatrix = tmatrix
        self.numPartitions = numPartitions

model = dill.load(open('/group/ag_cmb/scratch/dibakma/MSMRD/models/asym3D/periodicModel_lag0.05_R2.5_120files_240partitions_recovered.p', 'r'))
T = np.copy(model.tmatrix)

def run_exit_times(statePair, runs):
    np.random.seed()
    r1 = np.array(np.zeros(3))
    p1 = mrd.particle(r1, 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(4.)
    integrator = integrators.MSMRDtruncTrajs3D(msm, 4.0, p1, 0.001, 2.5, boundary, model)
    sim = mrd.simulation(integrator)
    exitTimes = []
    exitPositions = []
    fpts = []
    exits = 0
    run = 0
    totalRuns = 0
    while run < runs:
        integrator.particle.position = np.array(np.zeros(3))
        integrator.clock = 0.
        integrator.MSM.state = statePair[0]
        integrator.lastState = statePair[0]
        integrator.transition = True
        integrator.MSMactive = True
        totalRuns += 1
        while True:
            sim.integrator.integrate()
            if integrator.MSM.state == statePair[1]:
                fpts.append(integrator.clock)
                run += 1
                break
            elif not integrator.MSMactive:
                exits += 1
                exitTimes.append(integrator.clock)
                exitPositions.append(integrator.particle.position)
                break
    print str(statePair[0])+' to ' + str(statePair[1]) 
    return [np.array(fpts), totalRuns,  exitTimes, exitPositions]
'''
statePairs = []
for i in range(9):
    for j in range(9):
        statePairs.append((i,j))
'''
statePairs = [(2,5)]
pool = Pool(processes=4)
runs = 10000
FPT_list = []
results = pool.map(partial(run_exit_times, runs=runs), statePairs)
'''
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
dill.dump(FPTs, open(path+'fpts_'+str(runs)+'_runs_'+suffix, 'wa' ))
dill.dump(MFPTs, open(path+'mfpts_'+str(runs)+'_runs_'+suffix, 'wa' ))
'''
dill.dump(results,  open(path+'results_'+str(runs)+'_runs_rerun_'+suffix, 'wa' ))
