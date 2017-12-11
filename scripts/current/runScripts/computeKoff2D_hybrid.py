import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

# Hybrid simulation code to obtain MFPTs between MSM states and to and from the bath state

MFPTS = np.zeros([9,9])
minima = [[-0.9,0.7,0.3] ,  [-0.1,0.9,0.7],  [0.8,0.8,-0.8],  \
          [-1.0,-0.3,-0.4], [0.0,0.0,0.0],   [0.9,-0.1,-0.9], \
          [-0.7,-1.0,-0.3], [0.0,-0.9,0.1],  [0.8,-0.2,0.8]]

class truncTrajsModel(object):    
    def __init__(self, entryTrajsStart, entryTrajsEnd, entryTimes, exitTrajs, exitTimes, exitProbs, MSMtime, tmatrix):
        self.entryTrajsStart = entryTrajsStart
        self.entryTrajsEnd = entryTrajsEnd
        self.entryTimes = entryTimes
        self.exitTrajs = exitTrajs
        self.exitTimes = exitTimes
        self.exitProbs = exitProbs
        self.tmatrix = tmatrix
        self.MSMtimestep = MSMtime

model = dill.load(open('/group/ag_cmb/scratch/dibakma/MSMRD/models/asym2D/periodicModel_lag50_60partitions_SF2_exitCompensation.p'))
T = np.copy(model.tmatrix)

suffix = '_lag0.05_R2.5_60partitions.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/asym2D/rates/' 

# Calculate MFPTs from a given state to the bath
def run_mfpts_to_bath(state, runs, dt = 0.001):
    p1 = mrd.particle(np.zeros(2), 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(4.)
    integrator = integrators.MSMRDtruncTrajs(msm, 4.0, p1, 0.001, model, 2.5)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.p.position = np.zeros(2)
        integrator.clock = 0.
        integrator.MSM.state = state
        integrator.lastState = state
        integrator.lastStateTime = 0
        integrator.transition = True
        integrator.MSMactive = True
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt(2.7))
    print str(state)+' to bath'
    return np.array(fpts)
states = range(9)
pool = Pool(processes=2)
runs = 10000

FPT_list = pool.map(partial(run_mfpts_to_bath, runs=runs), states)
dill.dump(FPT_list, open(path+'fpts_off'+str(runs)+'_runs'+suffix, 'wa' ))
