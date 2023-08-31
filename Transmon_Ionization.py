r"""
Python module for examining ionization in the transmon using the branch analysis
"""
###############################################################
########################## Importing ##########################
###############################################################
import numpy as np
from numpy import linalg as LA
from numpy import pi as pi

import scipy as sp

import qutip as qp

import warnings

import copy

##################################################################
########################## Initializing ##########################
##################################################################
class TR():
    def __init__(self, params):
        self.params = copy.deepcopy(params)
        
        #Necessary parameters for transmon and resonator
        self.dim_t = self.params['dim_t']
        self.EC = self.params['EC']
        self.EJ = self.params['EJ']
        
        self.dim_r = self.params['dim_r']
        self.w_r = self.params['w_r']
        self.kappa = self.params['kappa']
        
        self.g = self.params['g']
        
        self.dim = self.dim_t*self.dim_r #Dimension of Hilbert space and in particular number of eigenvectors
        
        #Parameters that can be zero
        if self.params.get('n_g') == None:
            self.n_g = 0
        else:
            self.n_g = self.params['n_g']
        
        if self.params.get('phi_0') == None:
            self.phi_0 = 0
        else:
            self.phi_0 = self.params['phi_0']
        
        #Transmon functions
        self.E_t, self.Eigvecs_t, self.n_t, self.cos_t, self.sin_t = self.transmon_diag()
        self.n_t, self.cos_t, self.sin_t = qp.tensor(qp.Qobj(self.n_t[:self.dim_t,:self.dim_t]), qp.qeye(self.dim_r)), qp.tensor(qp.Qobj(self.cos_t[:self.dim_t,:self.dim_t]), qp.qeye(self.dim_r)), qp.tensor(qp.Qobj(self.sin_t[:self.dim_t,:self.dim_t]), qp.qeye(self.dim_r))
        
        self.t_projectors = [qp.tensor(qp.basis(self.dim_t, it)*qp.basis(self.dim_t, it).dag(), qp.qeye(self.dim_r)) for it in range(self.dim_t)] #List of projector on transmon states sum_{n_r}|i_t,n_r><n_r, i_t|
        self.N_t = sum([i_t*P for i_t, P in enumerate(self.t_projectors)]) #Average transmon population sum_{n_r, i_t} i_t |i_t,n_r><n_r, i_t|
        
        #Resonator functions
        self.a = qp.tensor(qp.qeye(self.dim_t), qp.destroy(self.dim_r))
        self.a_dag = self.a.dag()
        self.n_r = qp.tensor(qp.qeye(self.dim_t), qp.num(self.dim_r))
        
        self.r_projectors = [qp.tensor(qp.qeye(self.dim_t), qp.basis(self.dim_r, m)*qp.basis(self.dim_r, m).dag()) for m in range(self.dim_r)] #List of projector on resonator states sum_{i_t}|i_t,n_r><n_r, i_t|
        
        #Getting eigenvalues and eigenvectors of full system
        self.E_tr, self.Eigvecs_tr= self.H_tr().eigenstates()
        self.E_tr -= self.E_tr[0]
        
        #Dressed ground and excited state. Defined as state with largest overlap with bare ground and excited state
        self.dressed_quantities()
        
#############################################################################
########################## Transmon diagonalization #########################
#############################################################################
    #Diagonalizing the transmon
    def transmon_diag(self):
        max_charge = 200
        
        n_t = np.diag(np.arange(-max_charge, max_charge+1))
        cos_t = 1/2*(np.diag(np.ones(2*max_charge), -1) + np.diag(np.ones(2*max_charge), 1))
        sin_t = 1/(2*1j)*(np.diag(np.ones(2*max_charge), -1) - np.diag(np.ones(2*max_charge), 1))
        
        Kinetic = 4*self.EC*(n_t-self.n_g*np.eye(2*max_charge+1))**2
        
        Potential = - self.EJ*(cos_t*np.cos(self.phi_0) + sin_t*np.sin(self.phi_0))
        
        E, Eigvecs = sp.linalg.eigh(Kinetic+Potential)
        E -= E[0]
        
        return E, Eigvecs, np.conj(Eigvecs.T)@n_t@Eigvecs, np.conj(Eigvecs.T)@cos_t@Eigvecs, np.conj(Eigvecs.T)@sin_t@Eigvecs
    
###########################################################################
######################### Resonator + Hamiltonian #########################
###########################################################################
    #Function that spits back a tensor prodduct state
    def tr_state(self, exct_list):
        if len(exct_list) != 2:
            return ValueError('Argument must be a list with two elements')
        
        if exct_list[0] >= self.dim_t or exct_list[0] < 0 or type(exct_list[0]) != int:
            return ValueError('Transmon excitation number must be a positive integer smaller than dim_t =  ' + str(exct_list[0]))
        
        if exct_list[1] >= self.dim_r or exct_list[0] < 0  or type(exct_list[1]) != int:
            return ValueError('Resonator excitation number must be a positive integer smaller than dim_r =  ' + str(exct_list[1]))
        
        return qp.tensor(qp.basis(self.dim_t, exct_list[0]), qp.basis(self.dim_r, exct_list[1]))
    
    #Transmon resonator Hamiltonian
    def H_tr(self):
        return qp.tensor(qp.qdiags(self.E_t[:self.dim_t],0),qp.qeye(self.dim_r))+self.w_r*self.n_r - 1j*self.g*self.n_t*(self.a-self.a_dag)
    
    #Overlaps with bare state |i_t, n_r> or uncondition probabilities \sum_{i_t} <i_t,n_r| . |n_r, i_t>, \sum_{n_t} <i_t,n_r| . |n_r, i_t>
    def bare_basis_overlap(self, state):
        overlap_mat = np.zeros([self.dim_t, self.dim_r])
        
        for it in range(self.dim_t):
            for nr in range(self.dim_r):
                overlap_mat[it, nr] = np.abs(state.overlap(self.tr_state([it, nr])))**2
        
        return overlap_mat
            
    def t_overlaps(self, state):
        return [np.abs(state.overlap(P*state)) for P in self.t_projectors]
    
    def r_overlaps(self, state):
        return [np.abs(state.overlap(P*state)) for P in self.r_projectors]
    
    def dressed_quantities(self):
        dressed_g0_idx = np.argmax([np.abs(state.overlap(self.tr_state([0,0])))**2 for state in self.Eigvecs_tr])
        self.dressed_g0 = copy.deepcopy(self.Eigvecs_tr[dressed_g0_idx])
        
        dressed_e0_idx = np.argmax([np.abs(state.overlap(self.tr_state([1,0])))**2 for state in self.Eigvecs_tr])
        self.dressed_e0 = copy.deepcopy(self.Eigvecs_tr[dressed_e0_idx])
        self.dressed_w_q = copy.deepcopy(self.E_tr[dressed_e0_idx])
        
        dressed_g1_idx = np.argmax([np.abs(state.overlap(self.tr_state([0,1])))**2 for state in self.Eigvecs_tr])
        dressed_e1_idx = np.argmax([np.abs(state.overlap(self.tr_state([1,1])))**2 for state in self.Eigvecs_tr])
        
        self.dressed_w_r = (self.E_tr[dressed_e1_idx]+self.E_tr[dressed_g1_idx] -self.dressed_w_q)/2
        self.chi = (self.E_tr[dressed_e1_idx]-self.E_tr[dressed_g1_idx] -self.dressed_w_q)/2

        self.dressed_alpha = copy.deepcopy(self.E_tr[np.argmax([np.abs(state.overlap(self.tr_state([2,0])))**2 for state in self.Eigvecs_tr])] - 2*self.dressed_w_q)

###########################################################################
############################# Branch analysis #############################
###########################################################################
    def branch_analysis(self, **maxes):
        if maxes.get('max_t') == None or maxes.get('max_r') == None:
            raise ValueError('Need to specify the max number of transmon and resonator states max_t, max_r you want to do the branch analysis with')
        elif type(maxes['max_t']) != int or type(maxes['max_r']) != int:
            raise ValueError('max_t and max_r should be integers')
        else:
            max_t, max_r = copy.deepcopy(maxes['max_t'], maxes['max_r'])
        
        #Builindg the matrices which we will use to determine the different branches
        branch_criteria_matrix = self.a_dag.transform(self.Eigvecs_tr).full()
        
        for lambda_idx in range(self.dim):
            for n_idx in range(self.dim):
                branch_criteria_matrix[lambda_idx, n_idx] = np.abs(branch_criteria_matrix[lambda_idx, n_idx])
        
        #Initializing branches and branch index
        self.branches_idx = [[] for l in range(max_t)]
        self.E_branches = [[] for l in range(max_t)]
        self.branches = [[] for l in range(max_t)]
        
        for i_t in range(max_t):
            #First state in branch has largest overlap with bare |i_t, 0> 
            self.branches_idx[i_t].append(np.argmax([np.abs(state[i_t*self.dim_r])**2 for state in self.Eigvecs_tr]))
            #Have to set the right row in the branch_criteria_matrix to zero to ensure that it is never picked again.
            branch_criteria_matrix[self.branches_idx[i_t][0]] = 0
            
            n_r = 0
            
            while n_r <= max_r:
                #Recursive definition of the branch
                self.branches_idx[i_t].append(np.argmax(branch_criteria_matrix[:, self.branches_idx[i_t][n_r]]))
                #Same idea, can't have two states selected be the same
                branch_criteria_matrix[self.branches_idx[i_t][n_r+1]] = 0
                n_r += 1
            
            i_t += 1
        
        #Now with the branch_idx we can assign the eigenvectors and energies correctly.
        for i_t in range(max_t):
            for n in range(max_r):
                self.E_branches[i_t].append(self.E_tr[self.branches_idx[i_t][n]])
                self.branches[i_t].append(self.Eigvecs_tr[self.branches_idx[i_t][n]])
        
        #Now compute <n_r> and <N_t>
        self.branches_n_r_avg = [[] for l in range(max_t)]
        self.branches_N_t_avg = [[] for l in range(max_t)]
        
        for i_t in range(max_t):
            for state in self.branches[i_t]:
                self.branches_n_r_avg[i_t].append(np.real(state.overlap(self.n_r*state)))
                self.branches_N_t_avg[i_t].append(np.real(state.overlap(self.N_t*state)))

##########################################################################################
############################# Time-dependent displaced frame #############################
##########################################################################################
    def alpha_r(self,t, drive_params): #Classical response to the drive. Assumes the drive is of the form epsilon*sin(w_d t)
        w_d = drive_params['w_d']
        epsilon = drive_params['epsilon']
        
        res = (np.exp(-1j*w_d*t)-np.exp((-1j*self.w_r-self.kappa/2)*t))/((w_d-self.w_r)+1j*self.kappa/2)
        
        counter = (np.exp(1j*w_d*t)-np.exp((-1j*self.w_r-self.kappa/2)*t))/((-w_d-self.w_r)+1j*self.kappa/2)
       
        return -epsilon/2*(res-counter)
    
    def envelope_minus(self,t, drive_params): #Envelope of term exp(-i omega_d t) that multiplies n_t after disaplcement
        w_d = drive_params['w_d']
        epsilon = drive_params['epsilon']
        
        return 1j*self.g*epsilon/2*(1-np.exp(1j*(w_d-self.w_r)*t-self.kappa/2*t))/(w_d-self.w_r+1j*self.kappa/2)
    
    def n_t_drive(self,t, drive_params): #Drive that the transmon sees after displacement, which should be equivalent to exp(-i omega_d t) envelope_minus + c.c.
        return 2*self.g*np.imag(self.alpha_r(t,drive_params))
    
####################################################################
############################# Dynamics #############################
####################################################################
    def time_evolve_g(self, t, drive_params, options = None):
        #If this is the first time running this function, create a dictionary that will be used to store results for different w_d
        if hasattr(self, 'start_g'):
            pass
        else:
            self.start_g = {}
            
            self.t_dist_g = {}
            self.t_pop_g = {}
            self.charge_avg_g = {}
            
            self.r_pop_g = {}
            self.a_r_avg_g = {}
            self.n_r_avg_g = {}
            
            self.n_r_avg_lab_g = {}
        
        self.g_times = copy.deepcopy(times)
        
        Measured_Ops = copy.deepcopy(self.t_projectors)
        Measured_Ops += copy.deepcopy([self.n_t, self.N_t])
        
        Measured_Ops += copy.deepcopy(self.r_projectors)
        Measured_Ops += copy.deepcopy([self.a, self.n_r])
        
        key = f"({drive_params['w_d']},{drive_params['epsilon']})"
        
        self.start_g[key] = qp.mcsolve([self.H_tr(),
                                          [self.n_t, self.n_t_drive]
                                         ],
                                        self.dressed_g0,
                                        t,
                                        np.sqrt(self.kappa)*self.a,
                                        Measured_Ops,
                                        progress_bar = True,
                                        options = options,
                                        args = drive_params)
        
        #Average of transmon operators
        self.t_dist_g[key] = self.start_g[key].expect[:self.dim_t]
        self.charge_avg_g[key] = self.start_g[key].expect[self.dim_t]
        self.t_pop_g[key] = self.start_g[key].expect[self.dim_t+1]
        
        #Average of resonator operators
        self.r_pop_g[key] = self.start_g[key].expect[self.dim_t +2: self.dim_t+2+self.dim_r]
        self.a_r_avg_g[key] = self.start_g[key].expect[self.dim_t+2+self.dim_r]
        self.n_r_avg_g[key] = self.start_g[key].expect[self.dim_t+2+self.dim_r+1]
        
        self.n_r_avg_lab_g[key] = self.n_r_avg_g[key] +self.a_r_avg_g[key]*np.conj(self.alpha_r(t, drive_params))+np.conj(self.a_r_avg_g[key])*self.alpha_r(t, drive_params) + np.abs(self.alpha_r(t,drive_params))**2
        
    def time_evolve_e(self, t,drive_params, options = None):
        #If this is the first time running this function, create a dictionary that will be used to store results for different w_d
        if hasattr(self, 'start_e'):
            pass
        else:
            self.start_e = {}
            
            self.t_dist_e = {}
            self.t_pop_e = {}
            self.charge_avg_e = {}
            
            self.r_pop_e = {}
            self.a_r_avg_e = {}
            self.n_r_avg_e = {}
            
            self.n_r_avg_lab_e = {}
        
        self.e_times = copy.deepcopy(t)
        
        key = f"({drive_params['w_d']},{drive_params['epsilon']})"
        
        Measured_Ops = copy.deepcopy(self.t_projectors)
        Measured_Ops += copy.deepcopy([self.n_t, self.N_t])
        
        Measured_Ops += copy.deepcopy(self.r_projectors)
        Measured_Ops += copy.deepcopy([self.a, self.n_r])
        
        self.start_e[key] = qp.mcsolve([self.H_tr(),
                                          [self.n_t, self.n_t_drive]
                                         ],
                                        self.dressed_e0,
                                        t,
                                        np.sqrt(self.kappa)*self.a,
                                        Measured_Ops,
                                        progress_bar = True,
                                        options = options,
                                        args = drive_params)
        
        #Average of transmon operators
        self.t_dist_e[key] = copy.deepcopy(self.start_e[key].expect[:self.dim_t])
        self.charge_avg_e[key] = copy.deepcopy(self.start_e[key].expect[self.dim_t])
        self.t_pop_e[key] = copy.deepcopy(self.start_e[key].expect[self.dim_t+1])
        
        #Average of resonator operators
        self.r_pop_e[key] = self.start_e[key].expect[self.dim_t +2: self.dim_t+2+self.dim_r]
        self.a_r_avg_e[key] = self.start_e[key].expect[self.dim_t+2+self.dim_r]
        self.n_r_avg_e[key] = self.start_e[key].expect[self.dim_t+2+self.dim_r+1]
        
        self.n_r_avg_lab_e[key] = self.n_r_avg_e[key] +self.a_r_avg_e[key]*np.conj(self.alpha_r(t, drive_params))+np.conj(self.a_r_avg_e[key])*self.alpha_r(t, drive_params) + np.abs(self.alpha_r(t,drive_params))**2
    
###################################################################
############################# Floquet #############################
###################################################################
    def Quantum_Floquet_Analysis(self, drive_params, branches, times, options = None):
        #Make sure that branches is a list of tuples
        if type(branches) != list or not(all(type(branch) == tuple for branch in branches)):
            raise ValueError("Branches must be a list of tuples of states you wish to track")
        
        #Make sure that the indices of the tuples are smaller than dim_t or dim_r 
        if not(all(len(branch) == 2 for branch in branches)) or not(all(branch[0] < self.dim_t for branch in branches)) or not(all(branch[1] < self.dim_r for branch in branches)):
            raise ValueError("Branches must have length two, and the arguments must be smaller than (dim_t, dim_r)")
        
        #If this is the first time running this function, create a dictionary that will be used to store results for different w_d
        if hasattr(self, 'quantum_floquet_branches'):
            pass
        else:
            self.quantum_floquet_branches = {}
            self.quantum_floquet_quasi_energies = {}
            self.quantum_floquet_a_r_avg = {}
            self.quantum_floquet_t_pop_avg = {}
            
            self.floquet_times = {} #Times at which the drive is evaluated
        
        key = f"({drive_params['w_d']},{drive_params['epsilon']})"
        
        self.quantum_floquet_branches[key] = {}
        self.quantum_floquet_quasi_energies[key] = {}
        self.quantum_floquet_a_r_avg[key] = {}
        self.quantum_floquet_t_pop_avg[key] = {}
        
        #Each of these dictionaries will have a dictionary for each branch
        for branches in branches:
            self.quantum_floquet_branches[key][f"{branch}"] = []
            self.quantum_floquet_quasi_energies[key][f"{branch}"] = []
            self.quantum_floquet_a_r_avg[key][f"{branch}"] = []
            self.quantum_floquet_t_pop_avg[key][f"{branch}"] = []
        
        self.floquet_times[key] = times
        
        #We want to assign branches to the Floquet modes we generate. Our first branch is thus the bare branches in our list
        for branch in branches:
            self.quantum_floquet_branches[key][f"{branch}"].append(self.tr_state([branch[0], branch[1]]))
        
        #The envelope of the exp(-i w_d t) in this displaced frame
        minus_amplitudes = copy.deepcopy(self.envelope_minus(self.floquet_times[f'{w_d}'], drive_params))
        
        #Period of drive. Remember we work in units where time is scaled by 2pi
        T = copy.deepcopy(1/drive_params['w_d'])
        
        #We can then run over all amplitudes
        for amp in minus_amplitudes:
            H_total = [self.H_tr(), [self.n_t*amp, 'exp(-1j*w_d*t)'], [self.n_t*np.conj(amp), 'exp(1j*w_d*t)']]
            
            floquet_modes, quasi_energies = qp.floquet.floquet_modes(H_total, T, args = {'w_d' : w_d})
        
            for branch in branches:
                overlap_vec = self.quantum_floquet_branches[key][f"{branch}"][-1].transform(floquet_modes) #overlpa_vec[j] = <floquet_modes[j]|previous_branch>
                floquet_eig_idx = np.argmax([np.abs(overlap_vec)**2 for ovrlap in overlap_vec.full().flatten()]) #Might be able to do something faster, but finding the max is not the bottleneck here
       
            
        for branch in branches:
            overlap_vec = self.tr_state([branch[0], branch[1]]).transform(floquet_modes) #overlpa_vec[j] = <floquet_modes[j]|i_t, n_r> 
            
            
            
            self.quantum_floquet_branches[key][f"{branch}"] = []
            self.quantum_floquet_branches[key][f"{branch}"].append(floquet_modes[floquet_eig_idx])
            
            self.quantum_floquet_quasi_energies[key][f"{branch}"] = []
            self.quantum_floquet_quasi_energies[key][f"{branch}"].append(quasi_energies[floquet_eig_idx])
            
            self.quantum_floquet_a_r_avg = {}
            
            #while overlap_vec[floquet_idx]
            
        
   