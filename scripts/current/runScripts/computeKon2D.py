import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

# Benchmark code of MFPTs between MSM states

radii = [2.75, 3.0, 3.25, 3.5, 3.75, 4., 4.25, 4.5]
radiusThreshold = 0.05
runs = 10000
MFPTS = np.zeros([9,9])
minima = np.array([[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]])
suffix = '_benchmark_dt0.001.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/asym2D/rates/' 

def get_startpoints(numpoints, radius):
    theta = np.linspace(0., 2*np.pi, num=numpoints, endpoint=False)
    startpoints = []
    for i in range(numpoints):
        x = np.cos(theta[i])
        y = np.sin(theta[i])
        startpoints.append(radius*np.array([x,y])) 
    return startpoints

def run_mfpts_from_bath(bathRad, numpoints, scalefactor, dt):
    np.random.seed()
    asympot2D = potentials.asym2Dpotential(scalefactor = scalefactor)
    p1 = mrd.particle(np.zeros(2), 1.0)
    sphereboundary = mrd.reflectiveRing(bathRad)
    integrator = integrators.brownianDynamicsSp(asympot2D, sphereboundary, p1, dt, 1.0)
    sim = mrd.simulation(integrator)
    startpoints = get_startpoints(numpoints, bathRad - radiusThreshold)
    fpts = []
    for startpoint in startpoints:
        integrator.pa.position = startpoint
        integrator.clock = 0.
        fpts.append(sim.run_mfpt_points(np.array(minima), 0.2))
    print 'Radius ' + str(bathRad)
    return np.array(fpts)

pool = Pool(processes=2)
FPT_list = pool.map(partial(run_mfpts_from_bath, scalefactor=2.0, numpoints=runs, dt=0.001), radii)
dill.dump(FPT_list, open(path+'fpts_on_'+str(runs)+'_runs'+suffix, 'wa' ))
