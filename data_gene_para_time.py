from SPDEs import *
from Rule import *
from Model import *
from Noise import *
from full_visualization import *


from IPython import embed
from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  configurations
##############################################################
ntrain = 100
ntest = 20
N = 120 # Number of realizations
k = 1 #batch_size
# sub = 2**3 #subsampling rate
# h = 2**13 // sub #total grid size divided by the subsampling rate
# s = h
t_tradition = 0
t_RS = 0

for i in range(int(N/k)):
    print(i)

    ################################################################
    # generate data
    ################################################################

    # Data is of the shape (number of samples, grid size)

    dx, dt = 1/128, 0.001 #space-time increments 
    a, b, s, t = 0, 1, 0, 0.05 # s

    # space-time boundaries

    X, T = Noise().partition(a,b,dx), Noise().partition(s,t,dt) # space grid O_X and time grid O_T

    W = Noise().WN_space_time_many(s, t, dt, a, b, dx, k) # Create realizations of space time white noise

    ic = lambda x: x*(1-x) # initial condition
    IC_1 = 0.1 * Noise().initial(k, X, scaling = 2) # 2 cycle
    IC_2 = np.array([[ic(dx*i) for i in range(len(X))] for n in range(k)])
    IC = IC_2
    # IC = IC_1+IC_2 / IC = IC_2

    mu = lambda x: 3*x-x**3 # drift
    sigma1 = lambda x: x # multiplicative diffusive term
    sigma2 = lambda x: 1 # additive diffusive term

    # solutions to the multiplicative equation 
    # Soln_mult = SPDE(BC = 'P', IC = IC, mu = mu, sigma = sigma1).Parabolic(W, T, X)
    
    t0 = default_timer()
    # solutions to the additive equation 
    Soln_add = SPDE(BC = 'P', IC = IC, mu = mu, sigma = sigma2).Parabolic(0.1*W, T, X)
    t1 = default_timer()
    t_tradition = t_tradition + t1 - t0

    t2 = default_timer()
    # solutions to the linearized equation
    I_xi = SPDE(BC = 'P', IC = lambda x: 0, mu = lambda x: 0, sigma = sigma2).Parabolic(0.1*W, T, X)
    # (K, 51, 129)
    # Will be used as an input to the model in order to speed up the model computation. All I_xi are solved in paralel

    # rule for multiplicative equation
    # R = Rule(kernel_deg = 2, noise_deg = -1.5, free_num = 3) # create rule with additive width m = 3
    # R.add_component(1, {'xi':1}) # add multiplicative width l = 1

    R_add = Rule(kernel_deg = 2, noise_deg = -1.5, free_num = 3) # create rule with additive width 3. No multiplicative component.

    I = SPDE(BC = 'P', T = T, X = X).Integrate_Parabolic_trees # initialize integration map I

    M_add = Model(integration = I, rule = R_add, height = 2, deg = 7.5) # initialize model for additive equation
    # height = 2 / 3 / 4 / 5


    # model for multiplicative equation
    # M = Model(integration = I, rule = R, height = 4, deg = 5) # initialize model

    # Set time-space points at which functions of the model will be evaluated and stored in memory
    points = [(i,j) for j in range(len(X)) for i in range(int((t-s)//dt+1))]

    # create model
    # Features_for_points = M_add.create_model_points(W, lollipops = I_xi, diff = True, dt = dt, points = points)

    # No trees of the form I_c[u_0] are added so the model without initial conditions is created
    # In order to add I_c[u_0] to the model need to solve 
    W_0 = np.zeros((k, len(T), len(X)))
    I_c = SPDE(BC = 'D', T = T, X = X, IC = IC, mu = lambda x: 0, sigma = lambda x: 0).Parabolic(W_0, T, X)
    # (K, 51, 129)

    # # Then call
    Features_for_points = M_add.create_model_points(W, lollipops = I_xi, diff = True, dt = dt, extra_planted = I_c, extra_deg = 2, key = "I_c[u_0]",  points = points)
    t3 = default_timer()
    t_RS = t_RS + t3 - t2

    # setting: u0 -> u
    # tree_without_xi = []
    # tree_with_xi = []
    # for i in range(Features_for_points[(0,0)].shape[1]):
    #     if 'xi' not in Features_for_points[(0,0)].columns[i]:
    #         tree_without_xi.append(i)
        # else:
        #     tree_with_xi.append(i)

    # print(Features_for_points[(0,0)].columns)
    # for i in tree_without_xi:
        # print(Features_for_points[(0,0)].columns[i])
    # for i in tree_with_xi:
    #     print(Features_for_points[(0,0)].columns[i])

    if i == 0:
        #######################################################################################################################
        x_data = np.zeros((k, len(X), int((t-s)//dt+1), Features_for_points[(0,0)].shape[1]))
        for n in tqdm(range(k)):
            x_data[n] = np.array([[[Features_for_points[(i,j)].iat[n, t] for t in range(Features_for_points[(0,0)].shape[1])] for i in range(int((t-s)//dt+1))] for j in range(len(X))])

        # setting: u0 -> u
        # x_data = np.array([[[Features_for_points[p].iat[i, t] for t in tree_without_xi] for p in points] for i in range(k)])

        # x_data = np.array([[[Features_for_points[p].iat[i, t] for t in tree_with_xi] for p in points] for i in range(k)])
        
        print(x_data.shape)
        # [sample, x, T, tree]

        y_data = Soln_add.transpose(0,2,1)
        # [sample, X, T]
        #######################################################################################################################
        
    else:
        x_ = np.zeros((k, len(X), int((t-s)//dt+1), Features_for_points[(0,0)].shape[1]))
        for n in tqdm(range(k)):
            x_[n] = np.array([[[Features_for_points[(i,j)].iat[n, t] for t in range(Features_for_points[(0,0)].shape[1])] for i in range(int((t-s)//dt+1))] for j in range(len(X))])

        # setting: u0 -> u
        # x_data = np.array([[[Features_for_points[p].iat[i, t] for t in tree_without_xi] for p in points] for i in range(k)])

        # x_data = np.array([[[Features_for_points[p].iat[i, t] for t in tree_with_xi] for p in points] for i in range(k)])

        y_ = Soln_add.transpose(0,2,1)
        
        x_data = np.concatenate((x_, x_data), axis=0)
        print(x_data.shape)
        y_data = np.concatenate((y_,y_data), axis=0)

print('time:')
print('Traditional Solver:{}'. format(t_tradition/N))
print('Regularity Structure:{}'.format(t_RS/N))

# dataloader = MatReader('/home/v-peiyanhu/fourier_neural_operator/dataset/burgers_data_R10.mat')
# x_data = dataloader.read_field('a')[:,::sub]
# y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain]
y_train = y_data[:ntrain]

x_test = x_data[-ntest:]
y_test = y_data[-ntest:]

# x_train = x_train.reshape(ntrain,x_train.shape[1],x_train.shape[2],1)
# x_test = x_test.reshape(ntest,x_test.shape[1],x_test.shape[2],1)

# np.save("/home/v-peiyanhu/rs+fno/parabolic_data/x_train4_t_xi", x_train)
# np.save("/home/v-peiyanhu/rs+fno/parabolic_data/x_test4_t_xi",x_test)
# np.save("/home/v-peiyanhu/rs+fno/parabolic_data/y_train4_t_xi",y_train)
# np.save("/home/v-peiyanhu/rs+fno/parabolic_data/y_test4_t_xi",y_test)


# #generate original x

# xo_data = Soln_add[:,0,:]

# xo_train = xo_data[:ntrain,:]
# xo_test = xo_data[-ntest:,:]

# xo_train = xo_train.reshape(ntrain,xo_train.shape[1],1)
# xo_test = xo_test.reshape(ntest,xo_test.shape[1],1)

# np.save("/home/v-peiyanhu/rs+fno/xo_train2_xi", xo_train)
# np.save("/home/v-peiyanhu/rs+fno/xo_test2_xi",xo_test)