import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial
import pickle

MFPTS = np.zeros([9,9])
minima = [[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]]

def sampleBathPosition():
    theta = 2*np.pi * np.random.rand()
    r = np.sqrt(np.random.rand()*(4.0**2 - 3.0**2) + 3.0**2)
    return r*np.array([np.sin(theta), np.cos(theta)])

def run_mfpts_from_bath(state, runs):
    global MFPTS, minima
    np.random.seed()
    asympot = potentials.asym2Dpotential(scalefactor = 2.0)
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0])
    p1 = mrd.particle(r1, 1.0)
    ringboundary = mrd.reflectiveRing(4.)
    integrator = integrators.brownianDynamicsSp(asympot, ringboundary, p1, 0.001, 1.0)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.pa.position = sampleBathPosition()
        fpts.append(sim.run_mfpt_point(np.array(minima[state]), 0.2))
    pickle.dump(np.array(fpts), open('../data/asym2D/MFPTS/bath_'+str(state)+'_'+str(runs)+'runs_dt1E-3_SF2.0.p', 'wa'))
    print 'state '+str(state)+' done'
    return np.mean(fpts)


def run_mfpts_to_bath(state, runs):
    global MFPTS, minima
    np.random.seed()
    asympot = potentials.asym2Dpotential(scalefactor = 2.0)
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0])
    p1 = mrd.particle(r1, 1.0)
    ringboundary = mrd.reflectiveRing(4.)
    integrator = integrators.brownianDynamicsSp(asympot, ringboundary, p1, 0.001, 1.0)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.pa.position = np.array(minima[state]) 
        fpts.append(sim.run_mfpt(3.0))
    pickle.dump(np.array(fpts), open('../data/asym2D/MFPTS/'+str(state)+'_bath_'+str(runs)+'runs_dt1E-3_SF2.0_R1.0.p', 'wa'))
    print 'state '+str(state)+' done'
    return np.mean(fpts)


states = [i for i in range(9)]
pool = Pool(processes=5)
MFPT_from_bath = pool.map(partial(run_mfpts_from_bath, runs=10000), states)
MFPT_to_bath = pool.map(partial(run_mfpts_to_bath, runs=10000), states)
