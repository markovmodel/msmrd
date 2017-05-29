import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial
import pickle

MFPTS = np.zeros([9,9])
minima = [[-0.9,0.7,0.3] ,  [-0.1,0.9,0.7],  [0.8,0.8,-0.8],  \
          [-1.0,-0.3,-0.4], [0.0,0.0,0.0],   [0.9,-0.1,-0.9], \
          [-0.7,-1.0,-0.3], [0.0,-0.9,0.1],  [0.8,-0.2,0.8]]

def sampleBathPosition(rmin,rmax):
    outshell = True
    while outshell:
        r = np.random.rand()
        r = rmax * r**(1./3.);
        if (r>=rmin) and (r<=rmax):
            outshell = False
    randcosth = 2.0*np.random.rand() - 1.0;
    randph = 2.0*np.pi*np.random.rand();
    theta = np.arccos(randcosth);
    posx = r*np.sin(th)*np.cos(randph);
    posy = r*np.sin(th)*np.sin(randph);
    posz = r*np.cos(th);
    return np.array([posx, posy, posz])

def run_mfpts_from_bath(state, runs, scalef, dt, rmin, rmax):
    global MFPTS, minima
    np.random.seed()
    asympot3D = potentials.asym3Dpotential(scalefactor = scalef)
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    z0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0, z0])
    p1 = mrd.particle(r1, 1.0)
    ringboundary = mrd.reflectiveRing(4.)
    integrator = integrators.brownianDynamicsSp(asympot3D, ringboundary, p1, dt, 1.0)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.pa.position = sampleBathPosition(rmin, rmax)
        fpts.append(sim.run_mfpt_point(np.array(minima[state]), 0.2))
    pickle.dump(np.array(fpts), open('../data/asym3D/MFPTS/bath_'+str(state)+'_'+str(runs)+'runs_' + str(dt) + 'dt_' + str(scalef) 'sf.p', 'wa'))
    print 'state '+str(state)+' done'
    return np.mean(fpts)


def run_mfpts_to_bath(state, runs, scalef, dt):
    global MFPTS, minima
    np.random.seed()
    asympot3D = potentials.asym3Dpotential(scalefactor = scalef)
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    z0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0, z0])
    p1 = mrd.particle(r1, 1.0)
    ringboundary = mrd.reflectiveRing(4.)
    integrator = integrators.brownianDynamicsSp(asympot3D, ringboundary, p1, dt, 1.0)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.pa.position = np.array(minima[state]) 
        fpts.append(sim.run_mfpt(3.0))
    pickle.dump(np.array(fpts), open('../data/asym2D/MFPTS/'+str(state)+'_bath_'+str(runs)+'runs_' + str(dt) + 'dt_' + str(scalef) 'sf.p', 'wa'))
    print 'state '+str(state)+' done'
    return np.mean(fpts)


states = [i for i in range(9)]
pool = Pool(processes=4)
#MFPT_from_bath = pool.map(partial(run_mfpts_from_bath, runs=100000, scalef = 2.0, dt = 0.01, rmin = 3.0, rmax = 4.0), states)
MFPT_to_bath = pool.map(partial(run_mfpts_to_bath, runs=10000, scalef = 2.0, dt = 0.01), states)