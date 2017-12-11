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


class truncTrajsModelMyoglobin(object):    
    def __init__(self, entryTrajsStart, entryTrajsEnd, entryTimes, exitTrajs, exitTimes, exitProbs, MSMtime, tmatrix, numPartitions, interactionRadius, description = None):
        self.entryTrajsStart = entryTrajsStart
        self.entryTrajsEnd = entryTrajsEnd
        self.entryTimes = entryTimes
        self.exitTrajs = exitTrajs
        self.exitTimes = exitTimes
        self.exitProbs = exitProbs
        self.MSMtimestep = MSMtime
        self.tmatrix = tmatrix
        self.numPartitions = numPartitions
        self.interactionRadius = interactionRadius
        self.description = description


model = dill.load(open('/group/ag_cmb/scratch/dibakma/MSMRD/models/myoglobin/myoglobin_lag150_240partitions_min200_eps0.33_lcs_rad25_fixed.p'))
T = np.copy(model.tmatrix)
interactionRadius = 25.#np.copy(model.interactionRadius)
# define state which is considered to be the bound state
boundState = 10#12
# diffusion constant of Myoglobin
D_CO = 0.25417 # A^2/ps

suffix = '_lag150_eps0.33_240partitions_lcs_rad25_fixed.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/myoglobin/rates/' 

# Calculate MFPTs from a given state to the bath
# Note that we use a much larger time step here this might be problematic numerically but should be fine, since we have larger lengthscales
def run_mfpts_to_bath(state, runs, dt = 0.1):
    p1 = mrd.particle(np.zeros(3), D_CO)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(2.*interactionRadius)
    integrator = integrators.MSMRDtruncTrajs3D(msm, 2.*interactionRadius, p1, dt, interactionRadius, boundary, model)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.particle.position = np.zeros(3)
        integrator.clock = 0.
        integrator.MSM.state = state
        integrator.lastState = state
        integrator.lastStateTime = 0
        integrator.transition = True
        integrator.MSMactive = True
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt(interactionRadius*1.2))
    print str(state)+' to bath'
    return np.array(fpts)
states = [10]
pool = Pool(processes=4)
runs = 10000

FPT_list = pool.map(partial(run_mfpts_to_bath, runs=runs), states)
dill.dump(FPT_list, open(path+'fpts_off'+str(runs)+'_runs'+suffix, 'wa' ))
