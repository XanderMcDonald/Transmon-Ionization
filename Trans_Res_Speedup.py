r"""
Python module for diagonalizing transmon + resonator system
"""

################ Importing ################
import numpy as np
from numpy import linalg as LA
from numpy import pi as pi

import scipy as sp

import qutip as qp

import warnings

import copy

############################################################################
############################################################################
############################################################################
############################################################################
########################### Original Hamiltonian ###########################
############################################################################
############################################################################
############################################################################
############################################################################

class TR_Orig:   
    def __init__(self, params):
        self.params = copy.deepcopy(params)
        
        self.EC = self.params['EC']
        self.EJ = self.params['EJ']
        self.N_t = self.params['N_t']
        
        self.N_r = self.params['N_r']
        self.w_r = self.params['w_r']
        
        self.g = self.params['g']
        
        #Number of charge states used to diagonalize the transmon
        if self.params.get('max_charge') == None:
            self.max_charge = 200
        elif type(self.params['max_charge']) != int :
            raise TypeError('Max number of charge states used to diagonalize the transmon should be an integer')
        else: 
            self.max_charge = self.params['max_charge']
             
        #Making sure there is no offset phase or gate charge
        if self.params.get('phi_0') != None and self.params['phi_0'] != 0:
            return ValueError('Offset phase must be zero for this routine to work')

        if self.params.get('n_g') != None and self.params['n_g'] != 0:
            return ValueError('Gate charge must be zero for this routine to work')
        
        self.E_t, self.Eigvecs_t = self.Eigh_t()
        
        #Building the usual charge, cos and sin operators
        self.n_t = np.conj(self.Eigvecs_t.T)@(np.diag(np.arange(2*self.max_charge+1)-self.max_charge))@self.Eigvecs_t
    
        self.cos_t = np.conj(self.Eigvecs_t.T)@(0.5*(np.diag(np.ones(2*self.max_charge), 1) + np.diag(np.ones(2*self.max_charge), -1)))@self.Eigvecs_t
    
        self.sin_t = np.conj(self.Eigvecs_t.T)@(1j*0.5*(np.diag(np.ones(2*self.max_charge), 1) - np.diag(np.ones(2*self.max_charge), -1)))@self.Eigvecs_t
        
        #Getting eigenvalues and eigenvectors two different ways
        self.E, self.Eigvecs= self.H_tr().eigenstates()
        self.E -= self.E[0]
        
#############################################################################
########################## Transmon-only functions ##########################
#############################################################################

################### Eigenvalues and Eigenvectors ###################
    def Eigh_t(self):
        ###### Getting eigenvectors whose wavefunction is odd under n -> - n ######
        Kinetic_odd = 4*self.EC*np.diag(np.arange(1,self.max_charge+1))**2
        Potential_odd = -self.EJ/2*(np.diag(np.ones(self.max_charge-1),1)+ np.diag(np.ones(self.max_charge-1),-1))
    
        [E_odd, Eigvecs_half_odd] = sp.linalg.eigh(Kinetic_odd + Potential_odd)
    
        #Creating full eigenvector of size 2*max_charge + 1. 
        Eigvecs_odd = np.zeros([2*self.max_charge+1, self.max_charge])
    
        for j in range(self.max_charge):
            Eigvecs_odd[self.max_charge+1: 2*self.max_charge+1, j] = Eigvecs_half_odd[:,j]
            Eigvecs_odd[0:self.max_charge,j] = -np.flip(Eigvecs_odd[self.max_charge+1: 2*self.max_charge+1, j])
        
        ###### Getting eigenvectors whose wavefunction is even under n -> - n ######
        Kinetic_even = 4*self.EC*np.diag(np.arange(0,self.max_charge+1))**2
        Potential_even = -self.EJ/2*(np.diag(np.ones(self.max_charge),1)+ np.diag(np.ones(self.max_charge),-1))
        
        #Without using symmetry, eigenvalue equation is -E \psi_0 -EJ/2(psi_1+psi_-1). Employing symmetry means making following substitution to Hamiltonian
        Kinetic_even[0,1] += -self.EJ/2
   
        [E_even, Eigvecs_half_even] = sp.linalg.eig(Kinetic_even + Potential_even)

        #We are digaonlizing a non-Hermitian matrix, we might get non-real eigenvalues
        if np.max(np.imag(E_even)) != 0:
            warnings.warn("Imaginary part of the energy was found and takes a value " + str(np.max(np.imag(E_even))))
    
        #We must sort the eigenvalues and eigenvectors ourself, since scipy doesn't do it for us for a non-Hermitian Hamiltonian
        order_list = np.argsort(np.real(E_even))
        
        E_even = np.real(E_even)
        
        #Creating a temporary set of eigenvectors which will use to reorganize our list
        temp_Eigvecs = copy.deepcopy(Eigvecs_half_even)
        
        for j in range(self.max_charge+1):
            Eigvecs_half_even[:,j] = temp_Eigvecs[:,order_list[j]]
        
        #Creating full eigenvector of size 2*max_charge + 1. 
        Eigvecs_even = np.zeros([2*self.max_charge+1, self.max_charge+1])
    
        for j in range(self.max_charge+1):
            Eigvecs_even[self.max_charge: 2*self.max_charge+1, j] = Eigvecs_half_even[:,j]
            Eigvecs_even[0:self.max_charge,j] = np.flip(Eigvecs_half_even[1:, j])
        
        
        ###### Sorting even and odd eigenvectors ######
        Eigvecs = np.zeros([2*self.max_charge +1, 2*self.max_charge+1])
        E = np.zeros([2*self.max_charge+1])
        
        for j in range(self.max_charge): #Here we explicitely make use of the fact that ordering goes even, odd, even, odd, etc...
            Eigvecs[:,2*j] = Eigvecs_even[:,j]
            E[2*j] = E_even[j]
            
            Eigvecs[:,2*j+1] = Eigvecs_odd[:,j]
            E[2*j+1] = E_odd[j]
            
        Eigvecs[:,-1] = Eigvecs_even[:,-1]
        E[-1] = E_even[-1]
        
        E -= E[0] #Making sure ground state is always zero

        for j in range(2*self.max_charge+1):
            Eigvecs[:,j] /= sp.linalg.norm(Eigvecs[:,j])
        
        return E, Eigvecs

################### Sorting transmon excitation number ###################
    def Transmon_Exct_Number(self):
        Exct_list = np.zeros(2*self.max_charge+1)

        #Ground and excited are obviously in well
        Exct_list[0] = 0
        Exct_list[1] = 1

        exct_count = 2

        #Assigning excitation numbers to states within well
        while self.E_t[exct_count] < 2*self.EJ:
            Exct_list[exct_count] = exct_count
            exct_count += 1

        #Remaining states must then by definition by outside well and come in pairs
        for j in range(0, (2*self.max_charge+1-exct_count-1)//2):

            Exct_list[exct_count+2*j] = exct_count+j
            Exct_list[exct_count+2*j+1] = exct_count+j

        return Exct_list
    
###########################################################################
######################### Resonator + Hamiltonian #########################
###########################################################################
    def tr_state(self, exct_list):
        if len(exct_list) != 2:
            return ValueError('Argument must be a list with two elements')
        
        if exct_list[0] >= self.N_t or exct_list[0] < 0 or type(exct_list[0]) != int:
            return ValueError('Transmon excitation number must be a positive integer smaller than N_t =  ' + str(exct_list[0]))
        
        if exct_list[1] >= self.N_r or exct_list[0] < 0  or type(exct_list[1]) != int:
            return ValueError('Resonator excitation number must be a positive integer smaller than N_r =  ' + str(exct_list[0]))
        
        return qp.tensor(qp.basis(self.N_t, exct_list[0]), qp.basis(self.N_r, exct_list[1]))
        
    def H_tr(self):
        Exct_t = self.Transmon_Exct_Number()
        
        #Don't necessarily want to keep all 2*max_charge+1 eigenstates, keep only N_t states.
        #Following code makes sure that N_t is chosen such that states above well are kept in pairs as they should be.
        #If you pick wrong N_t, then this adds N_t+1 
        if Exct_t[self.params['N_t']+1] != Exct_t[self.params['N_t']]:
            self.params['N_t'] += 1
            self.N_t += 1 
            
        E_t, Eigvecs_t, n_t, cos_t, sin_t = self.E_t[:self.N_t], self.Eigvecs_t[:self.N_t, :self.N_t], self.n_t[:self.N_t, :self.N_t], self.cos_t[:self.N_t, :self.N_t], self.sin_t[:self.N_t, :self.N_t]

        n_t = qp.tensor(qp.Qobj(n_t), qp.qeye(self.N_r))
        
        a = qp.tensor(qp.qeye(self.N_t), qp.destroy(self.N_r))
        a_dag = a.dag()
        n_r = qp.tensor(qp.qeye(self.N_t), qp.num(self.N_r))
        
        return qp.tensor(qp.qdiags(E_t,0),qp.qeye(self.N_r))+self.w_r*n_r + 1j*self.g*n_t*(a-a_dag)
    

    