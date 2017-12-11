import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

# Hybrid simulation code to obtain MFPTs between MSM states and to and from the bath state

radii = [2.75, 3.0, 3.25, 3.5, 3.75, 4., 4.25, 4.5]
radiusThreshold = 0.05
runs = 100
minima = [[-0.9,0.7,0.3] ,  [-0.1,0.9,0.7],  [0.8,0.8,-0.8],  \
          [-1.0,-0.3,-0.4], [0.0,0.0,0.0],   [0.9,-0.1,-0.9], \
          [-0.7,-1.0,-0.3], [0.0,-0.9,0.1],  [0.8,-0.2,0.8]]
 
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


model = dill.load(open('/group/ag_cmb/scratch/dibakma/MSMRD/models/asym3D/periodicModel_lag0.01_R2.0_125files_240partitions.p'))
T = np.copy(model.tmatrix)

suffix = '_lag0.01_R2.0_125files_240partitions.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/asym3D/rates/' 

def get_startpoints(numpoints, radius):
    theta = np.linspace(0., 2*np.pi, num=numpoints, endpoint=False)
    u = np.linspace(0., 1, num=numpoints, endpoint=False)
    startpoints = []
    for i in range(numpoints):
        for j in range(numpoints):
            x = np.sqrt(1.-u[i])*np.cos(theta[j])
            y = np.sqrt(1.-u[i])*np.sin(theta[j])
            z = u[i]
            startpoints.append(radius*np.array([x,y,z])) 
    return startpoints

# Calculate MFPTs from the bath to a given state
def run_mfpts_from_bath(bathRad, numpoints, dt):
    np.random.seed()
    p1 = mrd.particle(np.zeros(3), 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(bathRad)
    integrator = integrators.MSMRDtruncTrajs3D(msm, bathRad, p1, dt, 2.0, boundary, model)
    sim = mrd.simulation(integrator)
    fpts = []
    startpoints = get_startpoints(numpoints, bathRad - radiusThreshold)
    for startpoint in startpoints:
        integrator.particle.position = startpoint 
        integrator.clock = 0.
        integrator.transition = False
        integrator.MSM.state = -1
        integrator.MSMactive = False
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt_states())
    print 'bath to ' + str(bathRad)
    return np.array(fpts)

pool = Pool(processes=4)
FPT_list = pool.map(partial(run_mfpts_from_bath, numpoints=runs, dt=0.001), radii)
dill.dump(FPT_list, open(path+'fpts_on_'+str(runs*runs)+'_runs'+suffix, 'wa' ))
