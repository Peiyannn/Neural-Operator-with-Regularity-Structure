# adapted from https://github.com/zongyi-li/fourier_neural_operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

#===========================================================================
# 2d fourier layers
#===========================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,t), (in_channel, out_channel, x,t) -> (batch, out_channel, x,t)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)


    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
            
        return x


class NORS_space1D_time(nn.Module):
    def __init__(self, modes1, modes2, width, L, num_tree):
        super(NORS_space1D_time, self).__init__()

        """
        The overall network. It contains L layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. L layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input shape: (batchsize, x=64, t=T, c=num_tree+2)
        output: the solution at T timesteps
        output shape: (batchsize, x=64, t=T, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_tree = num_tree
        self.L = L
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.num_tree+2, self.width) # input channel : (I[a], I_c[a],..., x, t)

        # self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm2d(self.width)
        # self.bn1 = torch.nn.BatchNorm2d(self.width)
        # self.bn2 = torch.nn.BatchNorm2d(self.width)
        # self.bn3 = torch.nn.BatchNorm2d(self.width)
        self.net = [ FNO_layer(modes1, modes2, width) for i in range(self.L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """ - x: (batch, dim_x, T, num_tree+2)
        """
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.to(torch.float32)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        # x1 = self.conv0(x)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        x = self.net(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_t = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, size_t, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridt), dim=-1).to(device)


#===========================================================================
# Training functionalities
#===========================================================================

def train_nors_1d(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    try:
            
        for ep in range(epochs):

            model.train()
            
            train_loss = 0.
            for x_, u_ in train_loader: 

                loss = 0.
                x_ = x_.to(device)
                u_ = u_.to(device)

                u_pred = model(x_)
                u_pred = u_pred[..., 0]
                loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            test_loss = 0.
            t_inf = 0.
            with torch.no_grad():
                for x_, u_ in test_loader:
                    
                    loss = 0.
                    
                    x_ = x_.to(device)
                    u_ = u_.to(device)
                    
                    t1 = default_timer()
                    u_pred = model(x_)
                    t2 = default_timer()
                    u_pred = u_pred[..., 0]
                    loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                    test_loss += loss.item()
                    t_inf = t_inf + t2 - t1
            # print('Inference time:{}'. format(t_inf/ntest))

            scheduler.step()
            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test
    
    except KeyboardInterrupt:

        return model, losses_train, losses_test

