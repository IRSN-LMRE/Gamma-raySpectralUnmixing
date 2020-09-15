'''
Spectral unmixing algorithms 
'''


import numpy as np
from numpy import  linalg as LA


def forward_backward(prox_g, grad_f, a_initial,L,n_item=0,tol=1e-12,verb=True,error_list=True,option='FISTA'):
    '''
     Minimize the sum of two functions using the Forward-backward splitting.
     This algorithm minimizes:    F(a) + G(a)
         G is "proximable"
         F is convexe and differentialble
         
     Input(required)
     ----------------------------
     1. prox_g : Compute proximal operators. callable.
     two arguments : 
        The vector on which we want to compute the proximal operator
        The step parameter       
        
     2. grad_f : Compute gradient. callable.
     one argument    
         The vector on which we want to compute the gradient
         
     3. a_initial : Initial vetor of a
     
     4. L : Lipschitz constant
     
     5. params: stopCriteria parameters 
     
     References
     ------------------------------------
     Forward- backward splitting (P. L. Combettes and V. Wajs, 2005)
     FISTA (A. Beck and M. Teboulle, 2009)
    '''
        
    # option FBS or FISTA
    
    gamma = .9/L
    
    # FBS
    a = np.copy(a_initial)
    
    # Fista
    t = 1.
    
    def FBS(a):
        a = prox_g(a - gamma * grad_f(a), gamma)
        return [a]
    
    def FISTA(a,t):
        ay = np.copy(a)
        anew = prox_g(ay - gamma * grad_f(ay), gamma)
        tp = (1 + np.sqrt(1 + 4. * t ** 2)) / 2
        ay = anew + (t - 1) / tp * (anew - a)
        t = np.copy(tp)
        a = np.copy(anew)
        return a,t
    
    if option == 'FBS':
        return stopCriteria(iterStep=FBS,n_item=n_item,tol=tol,values=[a],verb=verb,error_list=error_list)
    
    elif option == 'FISTA':
        return stopCriteria(iterStep=FISTA,n_item=n_item,tol=tol,values=[a,t],verb=verb,error_list=error_list)
    else:
        raise Exception('NO_OPTION')
            
def chambolle_pock(prox_fC, prox_g, a_initial, K, KT, sigma,tau,n_item=0,tol=1e-12,verb=True,error_list=True):
        
        
    '''
     Minimize the sum of two functions using the primal-dual algorithm.
     This algorithm minimizes: F(K(A))+G(A)
     F and G are convex functions and K(x) is a linear operator
     
     Input(required)
     ----------------------------
     1. prox_fC (conjugate f): Compute proximal operators. callable.
     two arguments : 
        The vector on which we want to compute the proximal operator
        The step parameter       
        
     2. prox_g: Compute proximal operators. callable.
     two arguments : 
        The vector on which we want to compute the proximal operator
        The step parameter  
        
     3. a_initial : Initial vector of a
     
     4. K : matrix (Phi in the spectral unmixing formulation)
     
     5. KT : matrix transpose (Phi transpose in the spectral unmixing formulation)
     
     6. sigma, tau: step parameters
     
     7. params: stopCriteria parameters 
     
    
    References
    ------------------------------------
    A. Chambolle and T. Pock, “A first-order primal-dual algorithm for convex problems with applications to imaging,”
    Journal of Mathematical Imaging and Vision, vol. 40, pp. 120 – 145, May 2011.
    '''
    a = np.copy(a_initial)
    abar = np.copy(a_initial)
    u = K(abar)
    theta = 1
        
    def iterate(a,abar,u):
        u = prox_fC(u+sigma * K(abar),sigma)
        anew = prox_g(a - tau * KT(u), tau)
        abar = anew + theta * (anew - a)   
        a = np.copy(anew)
        return a,abar,u
    
    return stopCriteria(iterStep=iterate,n_item=n_item,tol=tol,values=[a,abar,u],verb=verb,error_list=error_list)

def stopCriteria(iterStep,n_item=0,tol=1e-12,values=0,verb=True,error_list=True):
    '''
    The stop criteria for spectral unmixing algorithms
    Input(required)
    ----------------------------
    iterStep : Iteration step defined in the algorithm
    params:
    n_item : the max number of iterations (if n_item = 0, stops when error smaller than tol)
    tol : tolerance for stopping criteria
        
    Input(optional)
    ---------------------------
    verb: boolean -print error in each iteration default : True
    error_list: boolean -save error list defalut : False
    '''
    
    item = 0
    step_error = 1
    Valueslist = []
        
    if n_item == 0:
            
        while step_error > tol:
                
            oldValues = np.copy(values)
            values = iterStep(*values)
                
            if error_list:
                Valueslist.append(values[0])
            
            if item%1000 == 0:
                step_error = LA.norm((values[0]-oldValues[0]),ord=1)/LA.norm((oldValues[0]),ord=1)
                
                if verb:
                    print('iteration:',item,'--error:',step_error)
                
            item+=1
    else:
        
        for i in range(n_item):
            if step_error < tol:
                break
            
            oldValues = np.copy(values)
            values = iterStep(*values)
                
            if error_list:
                Valueslist.append(values[0])
                
            if i%1000 == 0:
                step_error = LA.norm((values[0]-oldValues[0]),ord=1)/LA.norm((oldValues[0]),ord=1)
                
                if verb:
                    print('iteration:',i,'--error:',step_error)

    return values[0],Valueslist




        
        


    
