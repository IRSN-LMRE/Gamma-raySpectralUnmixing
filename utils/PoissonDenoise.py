import numpy as np
from copy import deepcopy as dp
import scipy.linalg as lng
import copy as cp

def mad(x):

    v=np.abs(x-np.median(x))
    y=np.median(v)/0.6735
    return y

def VSTStarlet(m,J=8):
    
    T = np.size(m)
    c = np.zeros((T,J))
    
    for r in range(J):
        w=forward1d(m,J=r+1)
        c[:,r] = w[:,r+1]
    
    w=forward1d(m,J=J+1)
    f = w[:,0:J]/np.sqrt(1e-16+c)
        
    return c,f

def VSTDenoise(m,J=8,kmad=3.):
    
    c,f = VSTStarlet(m,J=J)
    
    w=forward1d(m,J=J+1)
    
    for r in range(J):
        f[:,r] = f[:,r]*(abs(f[:,r]) > kmad*mad(f[:,r]))
        
    w[:,0:J] = f*np.sqrt(1e-16+c)
    
    return np.sum(w,axis=1)


############################################################
################# STARLET TRANSFORM
############################################################

def length(x=0):

    l = np.max(np.shape(x))
    return l

################# 1D convolution

def filter_1d(xin=0,h=0,boption=3):

    import numpy as np
    import scipy.linalg as lng
    import copy as cp

    x = np.squeeze(cp.copy(xin));
    n = length(x);
    m = length(h);
    y = cp.copy(x);

    z = np.zeros(1,m);

    m2 = np.int(np.floor(m/2))

    for r in range(m2):

        if boption == 1: # --- zero padding

            z = np.concatenate([np.zeros(m-r-m2-1),x[0:r+m2+1]],axis=0)

        if boption == 2: # --- periodicity

            z = np.concatenate([x[n-(m-(r+m2))+1:n],x[0:r+m2+1]],axis=0)

        if boption == 3: # --- mirror

            u = x[0:m-(r+m2)-1];
            u = u[::-1]
            z = np.concatenate([u,x[0:r+m2+1]],axis=0)

        y[r] = np.sum(z*h)

    a = np.arange(np.int(m2),np.int(n-m+m2),1)

    for r in a:

        y[r] = np.sum(h*x[r-m2:m+r-m2])


    a = np.arange(np.int(n-m+m2+1)-1,n,1)

    for r in a:

        if boption == 1: # --- zero padding

            z = np.concatenate([x[r-m2:n],np.zeros(m - (n-r) - m2)],axis=0)

        if boption == 2: # --- periodicity

            z = np.concatenate([x[r-m2:n],x[0:m - (n-r) - m2]],axis=0)

        if boption == 3: # --- mirror

            u = x[n - (m - (n-r) - m2 -1)-1:n]
            u = u[::-1]
            z = np.concatenate([x[r-m2:n],u],axis=0)

        y[r] = np.sum(z*h)

    return y

################# 1D convolution with the "a trous" algorithm

def Apply_H1(x=0,h=0,scale=1,boption=3):

	m = length(h)

	if scale > 1:
		p = (m-1)*np.power(2,(scale-1)) + 1
		g = np.zeros( p)
		z = np.linspace(0,m-1,m)*np.power(2,(scale-1))
		g[z.astype(int)] = h

	else:
		g = h

	y = filter_1d(x,g,boption)

	return y

def forward1d(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):
    
    c = np.zeros((len(x),))
    w = np.zeros((len(x),J+1))
    c = cp.copy(x)
    cnew = cp.copy(x)
    
    for scale in range(J):
        cnew = Apply_H1(c,h,scale,boption)
        w[:,scale] = c - cnew
        c = np.copy(cnew)
    w[:,scale+1] = c
    return w
    