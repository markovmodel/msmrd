import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial
import pickle
import os

MFPTS = np.zeros([9,9])
minima = np.array([[0.0,0.0], [1.0,0.0] , [1.1, 1.0], [-0.1,0.9], [-1.3,0.8], [-1.0,-0.2], [-0.6,-1.0], [0.9,-0.8], [0.2,-1.5]])


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

def sampleBathPosition():
    theta = 2*np.pi * np.random.rand()
    r = np.sqrt(np.random.rand()*(4.0**2 - 3.0**2) + 3.0**2)
    return r*np.array([np.sin(theta), np.cos(theta)])

model = pickle.load(open('../data/models/asym2D/periodicModel_lag50_60partitions_SF2_exitCompensation_2files.p'))
T = np.copy(model.tmatrix)

suffix = '_hybrid_box_lag50_SF2_exitCompensation_2files.p'
path = '../data/asym2D/MFPTS/hybrid/'

def run_mfpts(statePair, runs):
    if statePair[0] == statePair[1]:
        return 0.
    np.random.seed()
    x0 = 2.0*np.random.rand() - 1.0
    y0 = 2.0*np.random.rand() - 1.0
    r1 = np.array([x0, y0])
    p1 = mrd.particle(r1, 1.0)
    msm = mrd.MSM(model.tmatrix, minima)
    msm.exitStates  = []
    integrator = integrators.MSMRDtruncTrajs(msm, 4.0, p1, 0.001, model, 2.5)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.clock = 0.
        integrator.MSM.state = statePair[0]
        integrator.lastState = statePair[0]
        integrator.transition = True
        integrator.MSMactive = True
        integrator.lastStateTime = 0
        fpts.append(sim.run_mfpt_state(statePair[1]))
    pickle.dump(np.array(fpts), open(path+'/data/'+str(statePair[0])+'_to_'+str(statePair[1])+'_'+str(runs)+'runs'+suffix, 'wa'))
    print str(statePair[0])+' to '+str(statePair[1])
    return np.mean(fpts)

def run_mfpts_to_bath(state, runs):
    np.random.seed()
    r1 = np.array([0., 0.])
    p1 = mrd.particle(r1, 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    integrator = integrators.MSMRDtruncTrajs(msm, 4.0, p1, 0.001, model, 2.5)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.p.position = np.array([0.,0.])
        integrator.clock = 0.
        integrator.MSM.state = state
        integrator.lastState = state
        integrator.lastStateTime = 0
        integrator.transition = True
        integrator.MSMactive = True
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt(3.0))
    pickle.dump(np.array(fpts), open(path+'/data/'+str(state)+'_to_bath_'+str(runs)+'runs'+suffix, 'wa'))
    print str(state)+' to bath'
    return np.mean(fpts)

def run_mfpts_to_bath_old(state, runs):
    np.random.seed()
    r1 = np.array([0., 0.])
    p1 = mrd.particle(r1, 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    integrator = integrators.MSMRDtruncTrajs(msm, 4.0, p1, 0.01, model, 2.5)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.p.position = np.array([0.,0.])
        integrator.clock = 0.
        integrator.MSM.state = state
        integrator.lastState = state
        integrator.transition = True
        integrator.MSMactive = True
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt(3.0))
    pickle.dump(np.array(fpts), open(path+'/data/'+str(state)+'_to_bath_'+str(runs)+'runs'+suffix, 'wa'))

def run_mfpts_from_bath(state, runs):
    np.random.seed()
    r1 = np.array([0., 0.])
    p1 = mrd.particle(r1, 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    integrator = integrators.MSMRDtruncTrajs(msm, 4.0, p1, 0.001, model, 2.5)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.p.position = sampleBathPosition()
        integrator.clock = 0.
        integrator.transition = False
        integrator.MSM.state = -1
        integrator.MSMactive = False
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt_state(state))
    pickle.dump(np.array(fpts), open(path+'/data/'+'bath_to_'+str(state)+'_'+str(runs)+'runs' + suffix, 'wa'))
    print 'bath to ' + str(state)
    return np.mean(fpts)

statePairs = []
for i in range(9):
    for j in range(9):
        statePairs.append((i,j))
states = range(9)
pool = Pool(processes=2)
runs = 10000
MFPT_list = pool.map(partial(run_mfpts, runs=runs), statePairs)
MFPT_list = pool.map(partial(run_mfpts_to_bath, runs=runs), states)
MFPT_list = pool.map(partial(run_mfpts_from_bath, runs=runs), states)

# load data and compute mean first passage times. Write first passage times to one file as nested list.
fpts = []
for i in range(9):
    fromi = []
    for j in range(9):
        if i != j:
            toj = pickle.load(open(path+'/data/'+str(i)+'_to_'+str(j)+'_'+str(runs)+'runs'+suffix, 'r'))
            fromi.append(toj)
        else:
            fromi.append([0])
    tobath = pickle.load(open(path+'/data/'+str(i)+'_to_bath_'+str(runs)+'runs'+suffix, 'r'))
    fromi.append(tobath)
    fpts.append(fromi)
frombath = []
for i in range(9):
    frombath.append(pickle.load(open(path+'/data/bath_to_'+str(i)+'_'+str(runs)+'runs' + suffix, 'r')))
frombath.append([0])
fpts.append(frombath)
# compute mean first passage times
mfpts = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        mfpts[i,j] = np.mean(fpts[i][j])
# write mfpts and fpts to file
pickle.dump(fpts, open(path+'fpts_'+str(runs)+'runs'+suffix,'wa'))
pickle.dump(mfpts, open(path+'mfpts_'+str(runs)+'runs'+suffix,'wa'))
# remove data
for i in range(9):
    for j in range(9):
        if i != j:
            os.remove(path+'/data/'+str(i)+'_to_'+str(j)+'_'+str(runs)+'runs'+suffix)
        else:
            fromi.append([0])
    os.remove(path+'/data/'+str(i)+'_to_bath_'+str(runs)+'runs'+suffix)
for i in range(9):
    os.remove(path+'/data/bath_to_'+str(i)+'_'+str(runs)+'runs' + suffix)

