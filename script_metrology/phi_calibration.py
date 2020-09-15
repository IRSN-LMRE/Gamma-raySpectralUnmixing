
"""
The calibration of spectral signatures for real data analysis
"""
from scipy import interpolate
import numpy as np


# Cubic interpolation of spectrum
def interp(xp,yp,xnew,lp,lnew):
    '''
    xp: energy bins of the spectrum
    yp: counts of the spectrum
    xm: energy bins at which to evaluate the interpolated values.
    lp: energy bins' width of the spectrum
    lnew: energy bins' width at which to evaluate the interpolated values
    '''
    f = interpolate.interp1d(xp,yp,bounds_error=False,kind='cubic',fill_value=0) 
    ynew = f(xnew)
    ynew[ynew<min(i for i in yp if i > 0)]= 0 
    ynew = (lnew/lp)*ynew
    return ynew

# shift a spectrum with zeros padding
def offset(x,step):
    y = np.zeros(len(x))
    if step<0: 
        y[:step]=x[-step::]
    elif step>0:
        y[step::] = x[:-step]
    else:
        y=x
    return y

# determine the shift correction which minimizes least squares residual in ROI
def shift(ind_dic,E,ind_r,phi_hr,energy_hr,l_hr,energy,Ac,x,b):  
    '''
    Iuputs:
    --------peak regions inputs-----------------
    ind_dic: region of interst of each peak, [energy_min,energy_max] at each row 
    E: peak energy of each peak
    ind_r: index of radionuclides for each peak
    --------high resoultion inputs--------------
    phi_hr: High resolution spectral signatures 
    energy_hr: High resoultion energy bins
    l_hr: high resoluation energy bins width
    --------measurement inputs------------------
    energy: energy bins of the measured spectrum
    Ac: preliminary estimation of mixing weights
    x: measured spectrum
    b: background spectrum
    
    Outputs:
    delta_e: number of shift bins of phi_hr
    u: shift correction polynomial fit parameters
    '''
    y0 = phi_hr@Ac # hight resoultion estimation model

    sigma2=[] # weights of residuals 
    delta_e = []
    
    for [min_ind,max_ind],r in zip(ind_dic,ind_r):
    
        indx = np.where((energy>min_ind)&(energy<max_ind)) # determine peak region index 
    
        Af = np.copy(Ac)
        Af[r] = 0
        ym = [] # equivalent background other RN
        
        for i in indx[0]:  
            i_left = np.argmin(abs(energy_hr-energy[i] + (energy[i]-energy[i-1])/2))
            i_right = np.argmin(abs(energy_hr-energy[i] - (energy[i+1]-energy[i])/2))            
            ym.append(np.mean((phi_hr@Af)[i_left:i_right])) 
            
        sigma2.append(1/np.sum(np.asarray(ym)+b[indx])) 
        
        s = (x-b)[indx] 
        dex = []
        step = np.arange(-15,15)
        
        for n in step:

            yc = offset(y0,n) # shift n steps
    
            y_test = []
    
            for i in indx[0]:
                i_left = np.argmin(abs(energy_hr-energy[i] + (energy[i]-energy[i-1])/2))
                i_right = np.argmin(abs(energy_hr-energy[i] - (energy[i+1]-energy[i])/2)) 
                y_test.append(np.mean(yc[i_left:i_right]))
    
            y_test = np.asarray(y_test)
            res = np.sum((s-y_test)**2)
            dex.append(res)
        
        delta_e.append(step[np.argmin(dex)]) 
    
    delta_e = np.asarray(delta_e)  
    sigma2 = np.asarray(sigma2)
    
    V = np.zeros((len(ind_r),4))

    for i in range(len(ind_r)):
        V[i,0] = 1
        V[i,1] = E[i]
        V[i,2] = E[i]**2
        V[i,3] = E[i]**3
    
    e_ecart = E+l_hr*delta_e
    u = np.linalg.inv(V.T@np.diag(sigma2)@V)@V.T@np.diag(sigma2)@e_ecart  
    
    return delta_e,u
def spectral_analysis(phi,x,b,b_measure,e0,e_phi,unmix_algo,ind_dic,E,ind_r):
    '''
    Inputs: 
    -----------------------------------------------------------
    phi: simulated spectral signatures 
    x: measured spectrum
    b_measure: measured background spectrum
    b: background spectrum (smoothed and interpolated to measurement energy bins)
    -----------------------------------------------------------
    e0: start energy (use the energy range > e0)
    e_phi: energy bins of simulations  
    -----------------------------------------------------------
    unmix_algo: unmix_algo(phi,x,b),algorithm to unmix the spectrum
    
    Peak informations to determine energy shift
    -----------------------------------------------------------
    ind_dic: region of interst of each peak, [energy_min,energy_max] at each row 
    E: peak energy of each peak
    ind_r: index of radionuclides for each peak
    '''
    nP = np.shape(phi)
    l_hr = 0.01 # high resoluation energy bins width (can be changed)
    ind_x = np.where(x['energy']>e0)
    
    xs = x['energy'][ind_x] # energy bins of the measurement
    ys = x['counts'][ind_x]
    t1 = b_measure['live_time'] # counting time of background (can be changed)
    t2 = x['live_time']
    
    b = t2/t1*b[ind_x] # to counting time of the measurment
    
    energy_max = np.max(xs)
    energy_hr = np.arange(10,energy_max+10,l_hr)
    phi_r = np.zeros((len(xs),nP[1])) 
    phi_hr = np.zeros((len(energy_hr),nP[1]))
    
    # Step 1: Interpolation of simulated spectral signatures
    for r in range(nP[1]):
        yp = phi[:,r]
        lp = e_phi[2]-e_phi[1] # simulation energy bins width (which are same for all energy range)
        phi_iterp = interp(e_phi,yp,energy_hr,lp,l_hr) # interpolation high resolution
        phi_hr[:,r] = phi_iterp
        phi_r[:,r] = interp(energy_hr,phi_iterp,xs,l_hr,x['A'][1]) # interpolation measurement
    
    # Step 2: First activity estimation 
    Ac = unmix_algo(phi_r,ys,b)
    
    
    # Step 3, 4: Correction of the energy shift for peaks of known energies and Energy correction function fitting
    delta_e,u = shift(ind_dic,E,ind_r,phi_hr,energy_hr,l_hr,xs,Ac,ys,b)
    energy_hr = u[0]+u[1]*energy_hr+u[2]*energy_hr**2+u[3]*energy_hr**3 # 
    
    phi_shift = np.zeros(phi_r.shape)
    nP = np.shape(phi_r)
    for r in range(nP[1]):
        phi_shift[:,r] = interp(energy_hr,phi_hr[:,r],xs,l_hr,x['A'][1])

    phi_shift = np.nan_to_num(phi_shift)
    Ac_shift = unmix_algo(phi_shift,ys,b)
    
    rr = {}
    rr['A'] = Ac_shift
    rr['phi'] = phi_shift
    rr['b'] = b
    rr['t'] = t2
    rr['x'] = ys
    rr['energy'] = xs
    
    return rr

    
        
    
