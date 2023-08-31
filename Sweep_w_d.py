########### Importing ###########
import numpy as np
from numpy import linalg as LA
from numpy import pi as pi

import qutip as qp

import scipy as sp

import copy

import pickle 

import time

from Transmon_Ionization import TR

#Units
GHz = 1
MHz = 1e-3

ns = 1*2*pi
us = 1e3*2*pi

params = {  'dim_t' : 16, #Number of transmon states
            'EC' : 280*MHz,
            'EJ' : 50*280*MHz,
            'dim_r' : 160, #Number of resonator states, i.e. N_r-1 is total max number of photons
            'w_r' : 7.5*GHz,
            'kappa' : 20*MHz,
            'g' :250*MHz
                    }
times = np.arange(0, pi*1/params['kappa']*(1.001), 1/params['kappa']*0.001) 

System = TR(params)

epsilon = 280*MHz

Create = open("Update.txt", 'x')
Create.close()

for Delta_w_d in np.array([-20*MHz, -10*MHz, 0*MHz, 10*MHz, 20*MHz]):
    
    start_time = time.time()
    
    System.time_evolve_e(times, {'w_d' : Delta_w_d+System.w_r, 'epsilon':epsilon} ,options = qp.solver.Options(nsteps = 3000, ntraj = 250, num_cpus = 25))
    
    with open("Recoded_Update.txt", 'a') as file:
        file.write(f"Finished running simulation for w_d = {Delta_w_d + System.w_r} it took {(time.time()-start_time)/60} minutes. \n")

Create = open("Recoded_Drive_Sweep.pkl", 'x')
Create.close()

with open("Recoded_Drive_Sweep.pkl", 'ab') as file:
        pickle.dump(System,file)


    
    