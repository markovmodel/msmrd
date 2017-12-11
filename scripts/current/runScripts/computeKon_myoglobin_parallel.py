import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

# Hybrid simulation code to obtain MFPTs between MSM states and to and from the bath state

minima = [np.array([ 28.12223244,  14.19789028,  35.48400116]), np.array([ 20.64152145,  32.80093765,  30.45864677]), np.array([ 37.6779213 ,  21.53466225,  14.31337452]), np.array([ 26.51408577,  35.27264023,  13.38927937]), np.array([ 28.60249901,  26.23224068,  34.70554352]), np.array([ 28.35201073,  15.2212162 ,  19.03871536]), np.array([ 11.4475584 ,  28.21699715,  15.27291393]), np.array([ 19.63159943,  16.71857643,  23.72273064]), np.array([ 15.11069298,  25.30273247,  30.79670715]), np.array([ 19.70653343,  24.95970917,  24.99034119]), np.array([ 20.28194046,  30.83546066,  22.95305252]), np.array([ 25.48045921,  21.23278427,  31.31694412]), np.array([ 29.18637085,  20.05953598,  29.45631981]), np.array([ 29.92654419,  34.40747452,  24.24386978]), np.array([ 26.37868881,  26.09814835,  26.27168083]), np.array([ 28.63999557,  28.84711266,  21.15133667]), np.array([ 21.23306656,  14.4724474 ,  34.73643494]), np.array([ 20.51725197,  21.56830215,  31.60126305]), np.array([ 18.09608459,  31.30299568,  18.05716515])]

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
interactionRadius = 25. #np.copy(model.interactionRadius)
# define state which is considered to be the bound state
boundState = 10#12
# diffusion constant of Myoglobin
D_CO = 0.25417 # A^2/ps
radiusThreshold = 0.1 #A

suffix = '_lag150_240partitions_rad25_fixed5.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/myoglobin/rates/'

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

def get_uniform_startpoints(rmin, rmax, N):
    # sample random points in cube and reject the ones that are in the forbidden region
    n = 0
    startpoints = []
    while n<N:
        r = rmax*(2*np.random.rand(3) - 1)
        norm = np.linalg.norm(r)
        if norm > rmin and norm < rmax:
            startpoints.append(r)
            n+=1
    return startpoints

# Calculate MFPTs from the bath to a given state
def run_mfpts_from_bath(startPosition, bathRad, dt):
    np.random.seed()
    p1 = mrd.particle(np.zeros(3), D_CO)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(bathRad)
    integrator = integrators.MSMRDtruncTrajs3D(msm, bathRad, p1, dt, interactionRadius, boundary, model)
    sim = mrd.simulation(integrator)
    integrator.particle.position = startPosition 
    integrator.clock = 0.
    integrator.transition = False
    integrator.MSM.state = -1
    integrator.MSMactive = False
    integrator.MSM.exit = False
    return sim.run_mfpt_state(boundState)

#maxBathRadius = 158.266943
#maxBathRadius = 75.00
#maxBathRadius1 = 50.00
#minBathRadius1 = 31.0175245
#minBathRadius2 = 75.
#maxBathRadius2 = 110.
#minBathRadius = 50.
#maxBathRadius = 75.
minBathRadius = 110
maxBathRadius = 158.266943

radii = np.linspace(minBathRadius, maxBathRadius, 5, endpoint=True)
midpoints = (radii[1:] + radii[:-1])/2.
runs = 25
#midpoints = np.array([])
print 'running for radii' + str(midpoints)
pool = Pool(processes=4)
for radius in radii[1:]:#midpoints:
    print 'running radius ' + str(radius)
    startpoints = get_uniform_startpoints(interactionRadius, radius, runs)
    print 'startpoints created ', startpoints
    FPT_list = pool.map(partial(run_mfpts_from_bath, bathRad=radius, dt=0.5), startpoints)
    dill.dump(FPT_list, open(path+'fpts_on_parallel_'+str(runs)+'runs_uniformStart_R%0.2f' % radius + '_eps0.33_lcs'+suffix, 'wa' ))
