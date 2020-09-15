import numpy as np

# proximal operator indicator function
def prox_Nonnegative(a,step):
    ind_negative = a < 0
    a[ind_negative] = 0
    return a

# proximal operator negative log likelihood of Poission (P(x|u))     
def Lpoisson(u,step):
    return lambda x: (u-step+np.sqrt((step-u)**2+4*step*x))/2

# Dual proximal operator by moreau decomposition
def dual_prox(prox):
    return lambda u, sigma: u - sigma * prox(u / sigma, 1 / sigma)

# gradient least square
def grad_LS(A):
    return lambda phi,b,x: np.dot(phi.T,(np.dot(phi,A)+b-x))

