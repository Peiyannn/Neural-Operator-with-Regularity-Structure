# adapted from https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures.git
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time


# In[11]:


class Noise(object):
    
    def partition(self, a,b, dx): #makes a partition of [a,b] of equal sizes dx
        return np.linspace(a, b, int((b - a) / dx) + 1)
    
    # Create l dimensional Brownian motion with time step = dt
    
    def BM(self, start, stop, dt, l):
        T = self.partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = np.random.normal(scale=np.sqrt(dt), size=(len(T), l))
        BM[0] = 0 #set the initial value to 0
        BM = np.cumsum(BM, axis  = 0) # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM

    # Create space time noise. White in time and with some correlation in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    # X here is points in space as in SPDE1 function.
    def WN_space_time_single(self, s, t, dt, a, b, dx, correlation = None, numpy = True):
        
        # If correlation function is not given, use space-time white noise correlation fucntion
        if correlation is None:
            correlation = self.WN_corr
        
        T, X = self.partition(s,t, dt), self.partition(a, b, dx) #time points, space points,
        N = len(X)
        # Create correlation Matrix in space
        space_corr = np.array([[correlation(x, j, dx * (N-1)) for j in range(N)] for x in X])
        B = self.BM(s, t, dt, N)
        
        if numpy:
            return np.dot(B, space_corr.T)
        
        return pd.DataFrame(np.dot(B, space_corr), index=T, columns=X)
    
    def WN_space_time_many(self, s, t, dt, a, b, dx, num, correlation = None):
        
        return np.array([self.WN_space_time_single(s, t, dt, a, b, dx, correlation = correlation) for _ in range(num)])
    
    # Funciton for creating N random initial conditions of the form 
    # \sum_{i = -p}^{i = p} a_k sin(k*\pi*x/scale)/ (1+|k|^decay) where a_k are i.i.d standard normal.
    def initial(self, N, X, p = 10, decay = 2, scaling = 1):
        scale = max(X)/scaling
        IC = []
        SIN = np.array([[np.sin(k*np.pi*(x-0.5)/scale)/((np.abs(k)+1)**decay) for k in range(-p,p+1)] for x in X])
        for i in range(N):
            sins = np.random.normal(size = 2*p+1)
            extra = np.random.normal(size = 1)
            IC.append(np.dot(SIN, sins)+extra)
            # enforcing periodic boundary condition without error
            IC[-1][-1] = IC[-1][0]
        
        return np.array(IC) 
    
    
    #Correlation function that approximates WN in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    def WN_corr(self, x, j, a):
        return np.sqrt(2 / a) * np.sin(j * np.pi * x / a)
    
    # save list of noises as a multilevel dataframe csv file
    def save(self, W, name):
        W.to_csv(name)
        
    def upload(self, name):
        Data = pd.read_csv(name, index_col = 0, header = [0,1])
        Data.columns = pd.MultiIndex.from_product([['W'+str(i+1) for i in range(Data.columns.levshape[0])], np.asarray(Data['W1'].columns, dtype = np.float16)])
        
        return Data


# In[ ]:




