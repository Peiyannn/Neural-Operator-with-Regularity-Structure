# adapted from https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures.git
# adapted from https://github.com/crispitagorico/Neural-SPDEs.git

from Classes.SPDEs import *
from Classes.Rule import *
from Classes.Model import *
from Classes.Noise import *

from Data.generator_sns import navier_stokes_2d
from Data.random_forcing import GaussianRF

import numpy as np
import torch
from timeit import default_timer
import math

torch.manual_seed(0)
np.random.seed(0)

################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 200
N = 1200
k = 100

sub_x = 4
sub_t = 1

t_tradition = 0
t_RS = 0

################################################################
# generate data
################################################################
for i in range(int(N/k)):
    print(i)

    device = torch.device('cuda')

    # Viscosity parameter
    nu = 1e-4

    # Spatial Resolution
    s = 64

    # domain where we are solving
    a = [1,1]

    # Temporal Resolution   
    T = 5e-2
    delta_t = 1e-3

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=3, tau=3, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s+1, device=device)
    t = t[0:-1]

    X,Y = torch.meshgrid(t, t)
    dx = X[1,0] - X[0,0]
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

    # Stochastic forcing function: sigma*dW/dt 
    stochastic_forcing = {'alpha':0.005, 'kappa':10, 'sigma':0.05}

    # Number of snapshots from solution
    record_steps = int(T/(delta_t))

    # Solve equations in batches (order of magnitude speed-up)

    # Batch size
    bsize = 10

    c = 0

    t0 = default_timer()

    #Sample random fields
    # w0 = GRF.sample(1).repeat(bsize,1,1)

    for j in range(k//bsize):

        w0 = GRF.sample(bsize) # (u0,xi)->u
        # w0 = torch.zeros((bsize, X.shape[0], X.shape[1]), device = device) # xi->u

        sol, sol_t, force = navier_stokes_2d(a, w0, f, nu, T, delta_t, record_steps, stochastic_forcing)  

        # add time 0
        time = torch.zeros(record_steps+1)
        time[1:] = sol_t.cpu()
        sol = torch.cat([w0[...,None],sol],dim=-1)
        force = torch.cat([torch.zeros_like(w0)[...,None],force],dim=-1)

        if j == 0:
            Soln = sol
            forcing = force
            IC = w0
            Soln_t = sol_t
        else:
            Soln = torch.cat([Soln, sol], dim=0)
            forcing = torch.cat([forcing, force], dim=0)
            IC = torch.cat([IC, w0],dim=0)

        c += bsize
        t1 = default_timer()
        # print(j, c, t1-t0)
        t_tradition = t_tradition + t1 - t0

    # Soln: [sample, x, y, step]
    # Soln_t: [t=step*delta_t]

    Soln = Soln[...,:-1].cpu()
    Soln_t = Soln_t.cpu()
    forcing = forcing[...,:-1].cpu()
    X, Y = X.cpu(), Y.cpu()
    IC = IC.cpu()

    Soln = Soln.transpose(2,3).transpose(1,2).numpy()
    Soln_t = Soln_t.numpy()
    forcing = forcing.transpose(2,3).transpose(1,2).numpy()
    X, Y = X.numpy(), Y.numpy()
    IC = IC.numpy()

    Soln = Soln[:,::sub_t,::sub_x,::sub_x]
    Soln_t = Soln_t[::sub_t]
    forcing = forcing[:,::sub_t,::sub_x,::sub_x]
    X, Y = X[::sub_x,::sub_x], Y[::sub_x,::sub_x]
    IC = IC[:,::sub_x,::sub_x]

    t2 = default_timer()

    sigma2 = lambda x: 1 # additive diffusive term

    f_0 = np.zeros((k, X.shape[0], X.shape[1]))
    IC_0 = np.zeros((k, X.shape[0], X.shape[1]))

    # solutions to the linearized equation
    I_xi = SPDE(BC = 'P', IC = IC_0, mu = lambda x: 0, sigma = sigma2, eps = nu).Parabolic_2d(forcing, Soln_t, X)
    # Will be used as an input to the model in order to speed up the model computation. All I_xi are solved in paralel

    R = Rule(kernel_deg = 2, noise_deg = -2, free_num = 2) # create rule with additive width 2. No multiplicative component.

    I = SPDE(BC = 'P', T = Soln_t, X = X, eps = nu).Integrate_Parabolic_trees_2d # initialize integration map I

    M = Model(integration = I, rule = R, height = 2, deg = 7.5, derivative = True) # initialize model for additive equation
    # height = 2 / 3 

    # Set time-space points at which functions of the model will be evaluated and stored in memory
    points = [(int(T/delta_t/sub_t)-1,i, j) for i in range(X.shape[0]) for j in range(X.shape[1])]

    # In order to add I_c[u_0] to the model need to solve 
    W_0 = np.zeros(forcing.shape)
    I_c = SPDE(BC = 'D', T = Soln_t, X = X, IC = IC, mu = lambda x: 0, sigma = lambda x: 0, eps = nu).Parabolic_2d(W_0, Soln_t, X)

    # # Then call
    Features_for_points = M.create_model_points_2d(forcing, lollipops = I_xi, diff = True, X = X, dt = sub_t*delta_t, batch_size = 10, extra_planted = I_c, extra_deg = 2, key = "I_c[u_0]",  points = points)
    t3 = default_timer()

    if i == 0:
        # print(Features_for_points[(0,0,0)].columns)
        #######################################################################################################################
        x_data = np.array([[[[Features_for_points[(int(T/delta_t/sub_t)-1,i,j)].iat[n, t] for t in range(Features_for_points[(int(T/delta_t/sub_t)-1,0,0)].shape[1])] for j in range(X.shape[1])] for i in range(X.shape[0])] for n in range(k)])
        # [sample, x, y, tree]

        # t3 = default_timer()
        t_RS = t_RS + t3 - t2
        print(x_data.shape)

        u_data = Soln[:,int(T/delta_t/sub_t)-1,:,:] 
        # [sample, x, y]
        #######################################################################################################################
    else:
        x_ = np.array([[[[Features_for_points[(int(T/delta_t/sub_t)-1,i,j)].iat[n, t] for t in range(Features_for_points[(int(T/delta_t/sub_t)-1,0,0)].shape[1])] for j in range(X.shape[1])] for i in range(X.shape[0])] for n in range(k)])
        # [sample, x, y, tree]

        # t3 = default_timer()
        t_RS = t_RS + t3 - t2
        print(x_data.shape)

        u_ = Soln[:,int(T/delta_t/sub_t)-1,:,:] 
        # [sample, x, y]

        x_data = np.concatenate((x_, x_data), axis=0)
        print(x_data.shape)
        u_data = np.concatenate((u_,u_data), axis=0)

print('time:')
print('Traditional Solver:{}'. format(t_tradition/N))
print('Regularity Structure:{}'.format(t_RS/N))

x_train = x_data[:ntrain,:]
print(x_train.shape)
u_train = u_data[:ntrain,:]

x_test = x_data[-ntest:,:]
u_test = u_data[-ntest:,:]

np.save("/..../NS_data/NS_x_train", x_train)
np.save("/.../NS_data/NS_x_test",x_test)
np.save("/.../NS_data/NS_u_train",u_train)
np.save("/.../NS_data/NS_u_test",u_test)
