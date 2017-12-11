import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

# Hybrid simulation code to obtain MFPTs between MSM states and to and from the bath state

radii = [2.75, 3.0, 3.25, 3.5, 3.75, 4., 4.25, 4.5]
radiusThreshold = 0.05
runs = 10000
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

suffix = '_lag0.05_R2.5_80files_60partitions.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/asym2D/rates/' 

def get_startpoints(numpoints, radius):
    theta = np.linspace(0., 2*np.pi, num=numpoints, endpoint=False)
    startpoints = []
    for i in range(numpoints):
        x = np.cos(theta[i])
        y = np.sin(theta[i])
        startpoints.append(radius*np.array([x,y])) 
    return startpoints

# Calculate MFPTs from the bath to a given state
def run_mfpts_from_bath(bathRad, numpoints, dt):
    np.random.seed()
    p1 = mrd.particle(np.zeros(2), 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(bathRad)
    integrator = integrators.MSMRDtruncTrajs(msm, bathRad, p1, 0.001, model, 2.5)
    sim = mrd.simulation(integrator)
    fpts = []
    startpoints = get_startpoints(numpoints, bathRad - radiusThreshold)
    for startpoint in startpoints:
        integrator.p.position = startpoint 
        integrator.clock = 0.
        integrator.transition = False
        integrator.MSM.state = -1
        integrator.MSMactive = False
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt_states())
    print 'bath to ' + str(bathRad)
    return np.array(fpts)

pool = Pool(processes=2)
FPT_list = pool.map(partial(run_mfpts_from_bath, numpoints=runs, dt=0.001), radii)
dill.dump(FPT_list, open(path+'fpts_on_'+str(runs)+'_runs'+suffix, 'wa' ))
