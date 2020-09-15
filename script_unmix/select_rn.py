"""
Sparse Spectal unmixing:
    Select active radionuclides by favouring sparse solution of the activity.
    References
    ------------------------------------
    Xu, J., Bobin, J., de Vismes Ott, A., and Bobin, C. (2020). Sparse spectral unmixing for activity 
    estimation in γ-ray spectrometry applied to environmental measurements. Applied Radiation and Isotopes, 156:108903.
"""
import numpy as np
from scipy.stats import chi2

class ompPoisson(object):
    '''
    Poisson Orthogonal Matching Pursuit (OMP) algorithm
    - Lpoisson: compute the loglikelihood poisson
    - algo_selection: select straightforward active radionuclides 
    '''
    
    def __init__(self,x,Phi,bdf,alpha,algo_unmix):
        '''
        
        sequentially selects the radionuclide that maximizes the likelihood
        Input
        ------------------------------------
        x: measured spectrum
        Phi: spectral library
        bdf: background radiation spectrum
        algoUnmixing: spectal unmixing algorithm 
        '''
        self.x = x
        self.Phi = Phi
        self.bdf = bdf
        self.alpha= alpha
        self.algo_unmix = algo_unmix
        

    def Lpoisson(self,mod):
        index_pos = np.where(self.x>0)[0]
        return np.sum(mod[index_pos] - self.x[index_pos]*np.log(mod[index_pos]))
    
    def algo_selection(self):
        
        t = np.shape(self.Phi)[1] # num of RN
        Ic = np.ones((t,1)) # ind to check
        I = np.zeros((t,1)) # ind chosen
        l0 = self.Lpoisson(self.bdf)
        
        for r in range(t):        
            # Check the non-selected radionuclides (check all in the begining)
            ind_Ic = np.where(Ic.squeeze() == 1)[0]        
            L = [] # -loglikelihood of each test
            for q in ind_Ic:
                
                test_I = np.copy(I.squeeze()) 
                test_I[q] = 1 # test the qth RN
                Itest = np.where(test_I == 1)[0]  
                aout = self.algo_unmix(self.Phi[:,Itest].reshape((-1,len(Itest)))) # estimated value
                mod = self.Phi[:,Itest]@aout + self.bdf # model phi@A + b
                L.append(self.Lpoisson(mod))

            # Calculate deviance between M0 (the model with selected radionuclides) and M1 (the model with an extra active radionuclide)
            l1 = np.min(L)
            D = 2*(l0-l1)
            # Chi2 statistical deviance 
            p = 1.- chi2.cdf(D,1)
           
            if p>self.alpha/(t-r): # Bonferroni
                Idone = np.where(I == 1)[0]
                break
            else:
                i_star = ind_Ic[np.argmin(L)] #select the radionuclide that minimizes the neg-log-likelihood
                print("Deviance : ", l0,'/',l1,'- selected this turn : ',i_star)                
                I[i_star] = 1
                Ic[i_star] = 0
                l0 = np.copy(l1)
                Idone = np.where(I == 1)[0]
        return Idone
    
