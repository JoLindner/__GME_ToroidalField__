import numpy as np

def gamma(l):
    return np.sqrt((2*l+1)/(4*np.pi))
    

def c_const(l,N):
    return np.sqrt((l+N)*(l-N+1))

