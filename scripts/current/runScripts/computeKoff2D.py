import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import dill

# Benchmark code of MFPTs between bath and MSM states

MFPTS = np.zeros([9,9])
minima = [[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]]
runbound = 2.7
suffix = '_benchmark_dt0.001.p'
path = '/group/ag_cmb/scratch/dibakma/MSMRD/asym2D/rates/' 
runs = 10000

# Calculate MFPT from a given the state to the bath
def run_mfpts_to_bath(state, runs, scalef, dt, runbound):
    global MFPTS, minima
    np.random.seed()
    asympot2D = potentials.asym2Dpotential(scalefactor = scalef)
    p1 = mrd.particle(np.zeros(2), 1.0)
    ringboundary = mrd.reflectiveRing(4.0)
    integrator = integrators.brownianDynamicsSp(asympot2D, ringboundary, p1, dt, 1.0)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.pa.position = np.array(minima[state]) 
        fpts.append(sim.run_mfpt(runbound))
    print 'state '+str(state)+' done'
    return np.array(fpts)


# Calculate the MFPTs to and from the bath
states = [i for i in range(9)]
pool = Pool(processes=2)
FPT_to_bath = pool.map(partial(run_mfpts_to_bath, runs=runs, scalef = 2.0, dt = 0.001, runbound=runbound), states)
dill.dump(FPT_to_bath, open(path+'fpts_off_'+str(runs)+'_runs'+suffix, 'wa' ))
