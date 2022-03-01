from NORS_1d_time import NORS_space1D_time, train_nors_1d
from utilities1 import count_params, LpLoss

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# **Get the dataloaders**

x_test = torch.from_numpy(np.load('/.../parabolic_data/x_test.npy'))
x_train = torch.from_numpy(np.load('/.../parabolic_data/x_train.npy'))
# [sample, x, T, tree]
u_test = torch.from_numpy(np.load('/.../parabolic_data/u_test.npy'))
u_train = torch.from_numpy(np.load('/.../parabolic_data/u_train.npy'))
# [sample, X, T]

ntrain = 1000
ntest = 200
num_tree = x_test.shape[-1]
batch_size = 20

x_train = x_train[:ntrain, :]
u_train = u_train[:ntrain, :]
x_test = x_test[-ntest:, :]
u_test = u_test[-ntest:, :]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size, shuffle=False)

# **Define a model**

model = NORS_space1D_time(modes1=32, modes2=24, width=32, L=4, num_tree=num_tree).cuda()

print('The model has {} parameters'. format(count_params(model)))

# **Train the model**

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss = LpLoss(size_average=False)

model, losses_train, losses_test = train_nors_1d(model, train_loader, test_loader, 
                                                device, loss, batch_size=batch_size, epochs=500, 
                                                learning_rate=0.001, scheduler_step=100, 
                                                scheduler_gamma=0.5, print_every=1)
