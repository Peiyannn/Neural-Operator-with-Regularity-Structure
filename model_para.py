# adapted from https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures.git

from Classes.SPDEs import *
from Classes.Rule import *
from Classes.Model import *
from Classes.Noise import *
from Classes.full_visualization import *

import numpy as np
from tqdm import tqdm
import torch
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  configurations
##############################################################
ntrain = 1000
ntest = 200
N = 1200 # Number of realizations
k = 100 #batch_size

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
    IC = IC_1 + IC_2
    # (u0,xi)->u: IC = IC_1+IC_2 / xi->u: IC = IC_2

    mu = lambda x: 3*x-x**3 # drift
    sigma1 = lambda x: x # multiplicative diffusive term
    sigma2 = lambda x: 1 # additive diffusive term
    
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

    R_add = Rule(kernel_deg = 2, noise_deg = -1.5, free_num = 3) # create rule with additive width 3. No multiplicative component.

    I = SPDE(BC = 'P', T = T, X = X).Integrate_Parabolic_trees # initialize integration map I

    M_add = Model(integration = I, rule = R_add, height = 2, deg = 7.5) # initialize model for additive equation
    # height = 2 / 3 

    # Set time-space points at which functions of the model will be evaluated and stored in memory
    points = [(i,j) for j in range(len(X)) for i in range(int((t-s)//dt+1))]

    # create model

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

    # print model feature vectors
    print(Features_for_points[(0,0)].columns)
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
        
        print(x_data.shape)
        # [sample, x, T, tree]

        u_data = Soln_add.transpose(0,2,1)
        # [sample, X, T]
        #######################################################################################################################
        
    else:
        x_ = np.zeros((k, len(X), int((t-s)//dt+1), Features_for_points[(0,0)].shape[1]))
        for n in tqdm(range(k)):
            x_[n] = np.array([[[Features_for_points[(i,j)].iat[n, t] for t in range(Features_for_points[(0,0)].shape[1])] for i in range(int((t-s)//dt+1))] for j in range(len(X))])

        # setting: u0 -> u
        # x_data = np.array([[[Features_for_points[p].iat[i, t] for t in tree_without_xi] for p in points] for i in range(k)])

        u_ = Soln_add.transpose(0,2,1)
        
        x_data = np.concatenate((x_, x_data), axis=0)
        print(x_data.shape)
        u_data = np.concatenate((u_,u_data), axis=0)

# print('time:')
# print('Traditional Solver:{}'. format(t_tradition/N))
# print('Regularity Structure:{}'.format(t_RS/N))

x_train = x_data[:ntrain]
u_train = u_data[:ntrain]

x_test = x_data[-ntest:]
u_test = u_data[-ntest:]

np.save("/.../parabolic_data/x_train", x_train)
np.save("/.../parabolic_data/x_test",x_test)
np.save("/.../parabolic_data/u_train",u_train)
np.save("/.../parabolic_data/u_test",u_test)
