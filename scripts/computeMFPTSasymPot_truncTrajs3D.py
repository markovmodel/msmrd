import numpy as np
import MSMRD as mrd
import MSMRD.integrators as integrators
import MSMRD.potentials as potentials
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial
import pickle

# Hybrid simulation code to obtain MFPTs between MSM states and to and from the bath state

MFPTS = np.zeros([9,9])
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

# Sample random position i sphericall shell between rmin and rmax 
def sampleBathPosition(rmin,rmax):
    outshell = True
    while (outshell):
        rr = np.random.rand()
        rr = rmax*np.cbrt(rr)
        if (rr >= rmin and rr <= rmax):
            outshell = False
    randcosth = 2.0*np.random.rand() - 1.0
    randph = 2.0*np.pi*np.random.rand()
    th = np.arccos(randcosth)
    posx = rr*np.sin(th)*np.cos(randph)
    posy = rr*np.sin(th)*np.sin(randph)
    posz = rr*np.cos(th)
    pos = np.array([posx, posy, posz])
    return pos

model = pickle.load(open('../data/models/asym3D/periodicModel_lag10_6files_177partitions.p'))
T = np.copy(model.tmatrix)

# Calculate MFPTs between a pair of states
def run_mfpts(statePair, runs, dt=0.01):
    if statePair[0] == statePair[1]:
        return 0.
    p1 = mrd.particle(np.zeros(3), 1.0)
    msm = mrd.MSM(model.tmatrix, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(4.)
    integrator = integrators.MSMRDtruncTrajs3D(msm, 4.0, p1, dt, 2.5, boundary, model)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.particle.position = np.zeros(3)
        integrator.clock = 0.
        integrator.MSM.state = statePair[0]
        integrator.lastState = statePair[0]
        integrator.transition = True
        integrator.MSMactive = True
        integrator.lastStateTime = 0
        fpts.append(sim.run_mfpt_state(statePair[1]))
    pickle.dump(np.array(fpts), open('../data/asym3D/MFPTS/hybrid/'+str(statePair[0])+'to'+str(statePair[1])+'_'+str(runs)+'runs_hybrid_box_dt' + str(dt) + '_exitCompensation.p', 'wa'))
    return np.mean(fpts)

# Calculate MFPTs from a given state to the bath
def run_mfpts_to_bath(state, runs, dt = 0.01):
    p1 = mrd.particle(np.zeros(3), 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(4.)
    integrator = integrators.MSMRDtruncTrajs3D(msm, 4.0, p1, dt, 2.5, boundary, model)
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
        fpts.append(sim.run_mfpt(3.0))
    pickle.dump(np.array(fpts), open('../data/asym3D/MFPTS/hybrid/'+str(state)+'_to_bath_'+str(runs)+'runs_hybrid_box_dt' + dt + 'exitCompensation.p', 'wa'))
    return np.mean(fpts)

# Calculate MFPTs from the bath to a given state
def run_mfpts_from_bath(state, runs, dt = 0.01):
    np.random.seed()
    p1 = mrd.particle(np.zeros(3), 1.0)
    msm = mrd.MSM(T, minima)
    msm.exitStates  = []
    boundary = mrd.reflectiveSphere(4.)
    integrator = integrators.MSMRDtruncTrajs3D(msm, 4.0, p1, dt, 2.5, boundary, model)
    sim = mrd.simulation(integrator)
    fpts = []
    for run in range(runs):
        integrator.particle.position = sampleBathPosition(2.5,4.0)
        integrator.clock = 0.
        integrator.transition = False
        integrator.MSM.state = -1
        integrator.MSMactive = False
        integrator.MSM.exit = False
        fpts.append(sim.run_mfpt_state(state))
    pickle.dump(np.array(fpts), open('../data/asym3D/MFPTS/hybrid/bath_to_'+str(state)+'_'+str(runs)+'runs_hybrid_box_dt' + dt + 'exitCompensation.p', 'wa'))
    return np.mean(fpts)


# Calculate the MFPTs between all the states
statePairs = []
for i in range(9):
    for j in range(9):
        statePairs.append((i,j))
states = range(9)
pool = Pool(processes=4)
runs = 10000
dt = 0.01
#for s in statePairs:
#	run_mfpts(s,1,dt)
MFPT_list = pool.map(partial(run_mfpts, runs=runs, dt=dt), statePairs)
#MFPT_list = pool.map(partial(run_mfpts_to_bath, runs=runs, dt=0.01), states)
#MFPT_list = pool.map(partial(run_mfpts_from_bath, runs=runs, dt=0.01), states)
'''
for i in range(9):
    for j in range(9):
        MFPTS[i,j] = MFPT_list[i*9+j]
for i in range(9):
    pool.mapt

#run_simulation(2)
processes = []
print("Simulation started")
for j in range(0, 2):
    for i in range(0, 4):
	print("Process " + str(i+4*j) + " running")
        p = Process(target = run_mfpts, args=(i+4*j,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
'''

#pickle.dump(MFPTS, open('mfpts_10runs_01dt.p', 'wa'))
