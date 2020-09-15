"""
Calculation of characteristic limits
"""
from scipy.stats import norm
import numpy as np
def sd_calcul(x,a_estim,b,phi,alpha=0.025):
    nP = phi.shape
    
    sd = {}
    sd['w1'] = np.zeros(nP[1])
    sd['w2'] = np.zeros(nP[1])
    for r in range(nP[1]):
        Af = np.copy(a_estim)
        Af[r] = 0
    
        # test with pseudo inverse solution w1
        psi = np.zeros((nP[0],2))
        psi[:,0] = phi[:,r]
        psi[:,1] = phi@Af+b # background without the RN to test
    
        w1 = ((np.linalg.inv(psi.T@psi))@psi.T)[0]

        mu = np.sum(w1*psi[:,1]) # mean 
        phi_sum = np.sum(w1*phi[:,r])
        var = np.sum(w1**2*psi[:,1]) # variance 
        l = norm.ppf(1-(alpha/2),mu,np.sqrt(var))
        sd['w1'][r] = (l-mu)/phi_sum
        
        psi_tn = psi.T@np.diag(1/(phi@a_estim+b))
        w2 = (np.linalg.inv(psi_tn@psi)@psi_tn)[0]
        mu = np.sum(w2*psi[:,1])
        phi_sum = np.sum(w2*phi[:,r])
        var = np.sum(w2**2*psi[:,1]) # variance 
        l = norm.ppf(1-(alpha/2),mu,np.sqrt(var))
        sd['w2'][r]  = (l-mu)/phi_sum
    
    return sd

def rn_fisher(x,a,b,phi):
    ind = np.where((phi@a+b)!=0)[0]
    I = (phi[ind,:]).T@(np.diag(x[ind]/(((phi@a+b)**2)[ind])))@phi[ind,:]
    return np.sqrt(np.linalg.inv(I).diagonal())
