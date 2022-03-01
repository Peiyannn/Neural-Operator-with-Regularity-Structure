# adapted from https://github.com/zongyi-li/fourier_neural_operator

from NORS_2d import NORS_space2D, train_nors_2d
from utilities1 import count_params, LpLoss, UnitGaussianNormalizer

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.01

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

x_test = torch.from_numpy(np.load('/.../NS_data/NS_x_test.npy'))
x_train = torch.from_numpy(np.load('/.../NS_data/NS_x_train.npy'))
u_test = torch.from_numpy(np.load('/.../NS_data/NS_u_test.npy'))
u_train = torch.from_numpy(np.load('/.../NS_data/NS_u_train.npy'))

x_test = x_test[:ntest,:]
x_train = x_train[:ntrain,:]
u_test = u_test[:ntest,:]
u_train = u_train[:ntrain,:]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

u_normalizer = UnitGaussianNormalizer(u_train)
u_train = u_normalizer.encode(u_train)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size, shuffle=False)

# **Define a model**

model = NORS_space2D(modes, modes, width, x_train.shape).cuda()

print('The model has {} parameters'. format(count_params(model)))

# **Train the model**

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss = LpLoss(size_average=False)

model, losses_train, losses_test = train_nors_2d(model, train_loader, test_loader, u_normalizer, u_train.shape,
                                                device, loss, batch_size=batch_size, epochs=500, 
                                                learning_rate=0.001, scheduler_step=100, 
                                                scheduler_gamma=0.5, print_every=1)

