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
        self.N_t = self.params['N_t']
        self.EC = self.params['EC']
        self.EJ = self.params['EJ']
        
        self.N_r = self.params['N_r']
        self.w_r = self.params['w_r']
        self.kappa = self.params['kappa']
        
        self.g = self.params['g']
        
        self.dim = self.N_t*self.N_r #Dimension of Hilbert space and in particular number of eigenvectors
        
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
        self.n_t, self.cos_t, self.sin_t = qp.tensor(qp.Qobj(self.n_t[:self.N_t,:self.N_t]), qp.qeye(self.N_r)), qp.tensor(qp.Qobj(self.cos_t[:self.N_t,:self.N_t]), qp.qeye(self.N_r)), qp.tensor(qp.Qobj(self.sin_t[:self.N_t,:self.N_t]), qp.qeye(self.N_r))
        
        self.t_projectors = [qp.tensor(qp.basis(self.N_t, it)*qp.basis(self.N_t, it).dag(), qp.qeye(self.N_r)) for it in range(self.N_t)]
        self.N_t_operator = sum([i_t*P for i_t, P in enumerate(self.t_projectors)])
        
        #Resonator functions
        self.a = qp.tensor(qp.qeye(self.N_t), qp.destroy(self.N_r))
        self.a_dag = self.a.dag()
        self.n_r = qp.tensor(qp.qeye(self.N_t), qp.num(self.N_r))
        
        self.r_projectors = [qp.tensor(qp.qeye(self.N_t), qp.basis(self.N_r, m)*qp.basis(self.N_r, m).dag()) for m in range(self.N_r)]
        
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
        
        if exct_list[0] >= self.N_t or exct_list[0] < 0 or type(exct_list[0]) != int:
            return ValueError('Transmon excitation number must be a positive integer smaller than N_t =  ' + str(exct_list[0]))
        
        if exct_list[1] >= self.N_r or exct_list[0] < 0  or type(exct_list[1]) != int:
            return ValueError('Resonator excitation number must be a positive integer smaller than N_r =  ' + str(exct_list[1]))
        
        return qp.tensor(qp.basis(self.N_t, exct_list[0]), qp.basis(self.N_r, exct_list[1]))
    
    #Transmon resonator Hamiltonian
    def H_tr(self):
        return qp.tensor(qp.qdiags(self.E_t[:self.N_t],0),qp.qeye(self.N_r))+self.w_r*self.n_r - 1j*self.g*self.n_t*(self.a-self.a_dag)
    
    #Overlaps with bare state |i_t, n_r> or uncondition probabilities \sum_{i_t} <i_t,n_r| . |n_r, i_t>, \sum_{n_t} <i_t,n_r| . |n_r, i_t>
    def bare_basis_overlap(self, state):
        overlap_mat = np.zeros([self.N_t, self.N_r])
        
        for it in range(self.N_t):
            for nr in range(self.N_r):
                overlap_mat[it, nr] = np.abs(state.overlap(self.tr_state([it, nr])))**2
        
        return overlap_mat
            
    def t_overlaps(self, state):
        return [np.abs(state.overlap(P*state)) for P in self.t_projectors]
    
    def r_overlaps(self, state):
        return [np.abs(state.overlap(P*state)) for P in self.r_projectors]
    
    def dressed_quantities(self):
        dressed_g0_idx = np.argmax([np.abs(state.overlap(self.tr_state([0,0])))**2 for state in self.Eigvecs_tr])
        self.dressed_g0 = self.Eigvecs_tr[dressed_g0_idx]
        
        dressed_e0_idx = np.argmax([np.abs(state.overlap(self.tr_state([1,0])))**2 for state in self.Eigvecs_tr])
        self.dressed_e0 = self.Eigvecs_tr[dressed_e0_idx]
        self.dressed_w_q = self.E_tr[dressed_e0_idx]
        
        dressed_g1_idx = np.argmax([np.abs(state.overlap(self.tr_state([0,1])))**2 for state in self.Eigvecs_tr])
        dressed_e1_idx = np.argmax([np.abs(state.overlap(self.tr_state([1,1])))**2 for state in self.Eigvecs_tr])
        
        self.dressed_w_r = (self.E_tr[dressed_e1_idx]+self.E_tr[dressed_g1_idx] -self.dressed_w_q)/2
        self.chi = (self.E_tr[dressed_e1_idx]-self.E_tr[dressed_g1_idx] -self.dressed_w_q)/2

        self.dressed_alpha = self.E_tr[np.argmax([np.abs(state.overlap(self.tr_state([2,0])))**2 for state in self.Eigvecs_tr])] - 2*self.dressed_w_q

###########################################################################
############################# Branch analysis #############################
###########################################################################
    def branch_analysis(self, **maxes):
        if maxes.get('max_t') == None or maxes.get('max_r') == None:
            raise ValueError('Need to specify the max number of transmon and resonator states max_t,, max_r you want to do the branch analysis with')
        elif type(maxes['max_t']) != int or type(maxes['max_r']) != int:
            raise ValueError('max_t and max_r should be integers')
        else:
            max_t, max_r = maxes['max_t'], maxes['max_r']
        
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
            self.branches_idx[i_t].append(np.argmax([np.abs(state[i_t*self.N_r])**2 for state in self.Eigvecs_tr]))
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
                self.branches_N_t_avg[i_t].append(np.real(state.overlap(self.N_t_operator*state)))

##########################################################################################
############################# Time-dependent displaced frame #############################
##########################################################################################
    def alpha_r(self,t, args): #Classical response to the drive. Assumes the drive is of the form epsilon*sin(w_d t)
        w_d = args['w_d']
        epsilon = args['epsilon']
        
        counter = (np.exp(1j*w_d*t)-np.exp((-1j*w_d-self.kappa/2)*t))/(1j*(self.w_r+w_d)+self.kappa/2)
        
        res = (np.exp(-1j*w_d*t)-np.exp((-1j*w_d-self.kappa/2)*t))/(1j*(self.w_r-w_d)+self.kappa/2)
        
        return epsilon/(2*1j)*(counter-res)
    
    def n_t_drive(self,t, args): #Drive that the transmon sees after displacement

        return 2*self.g*np.imag(self.alpha_r(t,args))

####################################################################
############################# Dynamics #############################
####################################################################
    def time_evolve_g(self, t, w_d, epsilon, options = None):
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
        
        args = {'w_d' : w_d, 'epsilon' : epsilon}
        
        Measured_Ops = self.t_projectors
        Measured_Ops += [self.n_t, self.N_t_operator]
        
        Measured_Ops += self.r_projectors
        Measured_Ops += [self.a, self.n_r]
        
        self.start_g[f"{w_d}"] = qp.mcsolve([self.H_tr(),
                                          [self.n_t, self.n_t_drive]
                                         ],
                                        self.dressed_g0,
                                        t,
                                        np.sqrt(self.kappa)*self.a,
                                        Measured_Ops,
                                        progress_bar = True,
                                        options = options,
                                        args = args)
        
        #Average of transmon operators
        self.t_dist_g[f"{w_d}"] = self.start_g[f"{w_d}"].expect[:self.N_t]
        self.charge_avg_g[f"{w_d}"] = self.start_g[f"{w_d}"].expect[self.N_t]
        self.t_pop_g[f"{w_d}"] = self.start_g[f"{w_d}"].expect[self.N_t+1]
        
        #Average of resonator operators
        self.r_pop_g[f"{w_d}"] = self.start_g[f"{w_d}"].expect[self.N_t +2: self.N_t+2+self.N_r]
        self.a_r_avg_g[f"{w_d}"] = self.start_g[f"{w_d}"].expect[self.N_t+2+self.N_r]
        self.n_r_avg_g[f"{w_d}"] = self.start_g[f"{w_d}"].expect[self.N_t+2+self.N_r+1]
        
        self.n_r_avg_lab_g[f"{w_d}"] = self.n_r_avg_g[f"{w_d}"] + np.conj(self.a_r_avg_g[f"{w_d}"])*self.alpha_r(t,args) + self.a_r_avg_g[f"{w_d}"]*np.conj(self.alpha_r(t, args)) + self.alpha_r(t,args)*np.conj(self.alpha_r(t ,args))
        
    def time_evolve_e(self, t, w_d, epsilon, options = None):
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
        
        args = {'w_d' : w_d, 'epsilon' : epsilon}
        
        Measured_Ops = self.t_projectors
        Measured_Ops += [self.n_t, self.N_t_operator]
        
        Measured_Ops += self.r_projectors
        Measured_Ops += [self.a, self.n_r]
        
        self.start_e[f"{w_d}"] = qp.mcsolve([self.H_tr(),
                                          [self.n_t, self.n_t_drive]
                                         ],
                                        self.dressed_e0,
                                        t,
                                        np.sqrt(self.kappa)*self.a,
                                        Measured_Ops,
                                        progress_bar = True,
                                        options = options,
                                        args = args)
        
        #Average of transmon operators
        self.t_dist_e[f"{w_d}"] = self.start_e[f"{w_d}"].expect[:self.N_t]
        self.charge_avg_e[f"{w_d}"] = self.start_e[f"{w_d}"].expect[self.N_t]
        self.t_pop_e[f"{w_d}"] = self.start_e[f"{w_d}"].expect[self.N_t+1]
        
        #Average of resonator operators
        self.r_pop_e[f"{w_d}"] = self.start_e[f"{w_d}"].expect[self.N_t +2: self.N_t+2+self.N_r]
        self.a_r_avg_e[f"{w_d}"] = self.start_e[f"{w_d}"].expect[self.N_t+2+self.N_r]
        self.n_r_avg_e[f"{w_d}"] = self.start_e[f"{w_d}"].expect[self.N_t+2+self.N_r+1]
        
        self.n_r_avg_lab_e[f"{w_d}"] = self.n_r_avg_e[f"{w_d}"] + np.conj(self.a_r_avg_e[f"{w_d}"])*self.alpha_r(t,args) + self.a_r_avg_e[f"{w_d}"]*np.conj(self.alpha_r(t, args)) + self.alpha_r(t,args)*np.conj(self.alpha_r(t ,args))
    
###################################################################
############################# Floquet #############################
###################################################################
    #Does the Floquet analysis for the transmon only
    def Floquet_Analysis_Transmon(self, w_d, epsilon, options = None):
        #First define the slow amplitude multiplying the cos and sin envelope of the drive
        detuning = w_d-self.w_r
        t_eval = 1/(2*w_d) #The envolope is assumed to be constant during a single evolution of the period. We thus evaluate it at one specific point t_eval, here halfway through evolution
        cos_env = epsilon*self.g/(detuning**2+(self.kappa/2)**2)*(
                                                                  self.kappa/2*(1-np.cos(detuning*t_eval)*np.exp(-self.kappa/2*t_eval))
                                                                 + detuning*np.sin(detuning*t_eval)*np.exp(-self.kappa/2*t_eval)
                                                                 )
        sin_env = epsilon*self.g/(detuning**2+(self.kappa/2)**2)*(
                                                                  detuning*(1-np.cos(detuning*t_eval)*np.exp(-self.kappa/2*t_eval))
                                                                 -self.kappa/2*np.sin(detuning*t_eval)*np.exp(-self.kappa/2*t_eval)
                                                                 )
        
   