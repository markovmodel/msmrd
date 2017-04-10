import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess 
import ConfigParser


#os.system("g++ const_c0.cpp")
#os.system("g++ const_c0.cpp -fopenmp")
#os.system("./a.out")

# Set runid
runid = "0"

# Read parameters file
config = ConfigParser.ConfigParser()
paramfile = "data/parameters_" + runid + ".dat"
config.read(paramfile)
# Get parameters
reactR = float(config.get("myparams", "Reaction radius"))
maxR = float(config.get("myparams", "Maximum radius"))
dt = float(config.get("myparams", "Time interval dt"))

#rad, conc = np.loadtxt("histogram.dat", delimiter=' ', usecols=(0,1) , unpack=True)
state, MFPT, rates = np.loadtxt("data/MFPTs_0.dat", delimiter=' ', usecols=(0,1,2) , unpack=True)
# Make sure rates are all positive
rates[rates<0] = 0


# Code a SSA (Gillespie style algorithm) using the rates
np.random.seed()
lam0 = sum(rates)
probs = rates/lam0

r1 = np.random.rand()
r2 = np.random.rand()

tau = np.log(1/r1)/lam0
for j in range(len(probs)-1):
    probj = sum(probs[0:j+1])
    if r2 < probj:
        reaction = j
        break
print tau, j

