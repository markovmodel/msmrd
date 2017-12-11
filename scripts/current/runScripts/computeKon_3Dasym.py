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
runs = 1000
MFPTS = np.zeros([9,9])
minima = [[-0.9,0.7,0.3] ,  [-0.1,0.9,0.7],  [0.8,0.8,-0.8],  \
         [-1.0,-0.3,-0.4], [0.0,0.0,0.0],   [0.9,-0.1,-0.9], \
          [-0.7,-1.0,-0.3], [0.0,-0.9,0.1],  [0.8,-0.2,0.8]]
suffix = '_benchmark_dt0.001.p'
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

def run_mfpts_from_bath(bathRad, numpoints, scalefactor, dt):
    np.random.seed()
    asympot3D = potentials.asym3Dpotential(scalefactor = scalefactor)
    p1 = mrd.particle(np.zeros(3), 1.0)
    sphereboundary = mrd.reflectiveSphere(bathRad)
    integrator = integrators.brownianDynamicsSp(asympot3D, sphereboundary, p1, dt, 1.0)
    sim = mrd.simulation(integrator)
    startpoints = get_startpoints(numpoints, bathRad - radiusThreshold)
    fpts = []
    for startpoint in startpoints:
        integrator.pa.position = startpoint
        integrator.clock = 0.
        fpts.append(sim.run_mfpt_points(np.array(minima), 0.2))
    print 'bath to ' + str(bathRad)
    return np.array(fpts)

pool = Pool(processes=8)
FPT_list = pool.map(partial(run_mfpts_from_bath, scalefactor=2.0, numpoints=runs, dt=0.001), radii)
dill.dump(FPT_list, open(path+'fpts_on_'+str(runs*runs)+'_runs'+suffix, 'wa' ))
