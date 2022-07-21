import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

#######################################################
#  Logic Functions

def AND(x):
    y = 0
    if x[0]==1 and x[1]==1:
        y = 1
    return torch.tensor(y).type(torch.float32)

def OR(x):
    y = 0
    if x[0]==1 or x[1]==1:
        y = 1
    return torch.tensor(y).type(torch.float32)

def NOR1(x):
    y = 0
    if x[0]==1 and x[1]==0:
        y = 1
    return torch.tensor(y).type(torch.float32)

def NOR2(x):
    y = 0
    if x[1]==0 and x[1]==1:
        y = 1
    return torch.tensor(y).type(torch.float32)

def XOR(x):
    y = 0
    if (x[0]==1 or x[1]==1) and x[0]!=x[1]:
        y = 1
    return torch.tensor(y).type(torch.float32)

def generate_samples(func):
    x = torch.randint(high=2,size=(1,2))[0]
    y = func(x)
    return x,y

#######################################################
# Basic Plotting Functions

def plot_activity(time,epoch=-1,data=[]):
    all_patterns, output_history, fbi_history, loss_history, weight_history = data
    output_size = 2
    input_size = 2
    fbi_size = 1

    # Activity heatmaps
    fig, ax = plt.subplots(1,3,figsize=(15,4.5))
    axis = 0
    im = ax[axis].imshow(all_patterns.T,aspect='equal',cmap='gray_r')
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('pattern')
    ax[axis].set_ylabel('input unit')

    axis = 1
    im = ax[axis].imshow(output_history[:,time,:,epoch],aspect='equal',vmin=0,cmap='gray_r')
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('pattern')
    ax[axis].set_ylabel('output unit')

    axis = 2
    im = ax[axis].imshow(fbi_history[:,time,:,epoch],aspect='equal',vmin=0,cmap='gray_r')
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('pattern')
    ax[axis].set_ylabel('fbi unit')

    plt.suptitle("Unit activities",fontsize=20)
    plt.tight_layout()

    # Activity dynamics
    fix, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    for i in range(output_size):
        ax[0].plot(torch.mean(output_history[i, :, :, epoch], axis=1), color='gray', alpha=0.9)
    ax[0].set_xlabel('timestep')
    ax[0].set_ylabel('unit activity')
    ax[0].set_title('Output units')

    for i in range(fbi_size):
        ax[1].plot(torch.mean(fbi_history[i, :, :, epoch], axis=1), color='gray', alpha=0.9)
    ax[1].set_xlabel('timestep')
    ax[1].set_title('FBI units')

    plt.suptitle('Unit dynamics over RNN steps', fontsize=20)

    sns.despine()
    plt.tight_layout()

    # Weights
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))

    axis = 0
    im = ax[axis].imshow(weight_history['in2out'][:, :, epoch], aspect='equal', cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('input unit')
    ax[axis].set_ylabel('output unit')
    ax[axis].set_title('in2out', fontsize=15)

    axis = 1
    im = ax[axis].imshow(weight_history['out2fbi'][:, :, epoch].T, aspect='equal', cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_ylabel('output unit')
    ax[axis].set_xlabel('fbi unit')
    ax[axis].set_title('out2fbi', fontsize=15)

    axis = 2
    im = ax[axis].imshow(weight_history['fbi2out'][:, :, epoch], aspect='equal', cmap='Blues_r', vmax=0)
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('fbi unit')
    ax[axis].set_ylabel('output unit')
    ax[axis].set_title('fbi2out', fontsize=15)

    axis = 3
    im = ax[axis].imshow(weight_history['fbi2fbi'][:, :, epoch], aspect='equal', cmap='Blues_r', vmax=0)
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('fbi unit')
    ax[axis].set_ylabel('fbi unit')
    ax[axis].set_title('fbi2fbi', fontsize=15)

    plt.suptitle("Weights", fontsize=20)
    plt.tight_layout()
    plt.show()

#######################################################
# Loss Function Plotting

def loss_function(model, w, b, target_func):
    all_patterns = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    model.in2out.weight.data.fill_(w)
    model.in2out.bias.detach()[1] = b

    loss = 0
    for x in all_patterns:
        y_predicted = model.forward_rnn(x)
        y_target = target_func(x)
        loss += torch.mean((y_target - y_predicted)**2)
    mse_loss = loss / len(x)
    return mse_loss


def plot_loss_landscape(model, target_func):
    model.fbi2out.weight.data = torch.tensor([[-1.1],[0]])
    model.out2fbi.weight.data = torch.tensor([[0., 1.8]])
    model.in2out.bias.detach()[0] = 0

    max_weight = 2
    max_bias = 2
    w_range = torch.linspace(0, max_weight, 50)
    b_range = torch.linspace(-max_bias, max_bias, 50)

    w_mesh, b_mesh = torch.meshgrid(w_range, b_range)

    loss_landscape = torch.zeros_like(w_mesh)
    for i in np.ndindex(w_mesh.shape):
        w = w_mesh[i]
        b = b_mesh[i]
        loss_landscape[i] = loss_function(model,w,b,target_func)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(b_mesh, w_mesh,
                   loss_landscape,
                   shading='nearest',
                   cmap='Reds')
    plt.xlabel('b')
    plt.ylabel('w')
    cbar = plt.colorbar()
    cbar.set_label('mse loss', rotation=-90.)

    plt.title('loss landscape')
    plt.show()

#######################################################
# Network definition

class FBI_RNN(nn.Module):
    def __init__(self, input_size, output_size, fbi_size,bias):
        super().__init__()
        self.out_size = output_size
        self.fbi_size = fbi_size
        self.in_size = input_size

        self.in2out = nn.Linear(self.in_size, self.out_size, bias=bias)
        self.fbi2out = nn.Linear(self.fbi_size, self.out_size, bias=False)
        self.out2fbi = nn.Linear(self.out_size, self.fbi_size, bias=False)
        self.fbi2fbi = nn.Linear(self.fbi_size, self.fbi_size, bias=False)

        # initialize close to optimal weights
        self.in2out.weight.data = torch.tensor([[ 1.,  1.],
                                                 [0.5, 0.5]])
        self.out2fbi.weight.data = torch.tensor([[1.,0.],
                                                 [0.,1.]])
        self.fbi2out.weight.data = torch.tensor([[-0.,-5.],
                                                 [-0.5,-0.]])
        self.fbi2fbi.weight.data = torch.tensor([[-0.,-1.],
                                                 [-1.,-0.]])

        # self.in2out.weight.data.uniform_(0.1,0.5)
        # self.out2fbi.weight.data.uniform_(0.1,0.5)
        # self.fbi2out.weight.data.uniform_(-0.5,-0.1)
        # self.fbi2out.weight.data.uniform_(-0.5,-0.1)

    def forward(self, input_pattern, out0, fbi0, act_sharpness=4):
        out = self.in2out(input_pattern) + self.fbi2out(fbi0)
        # out = F.softplus(out, beta=act_sharpness)
        out = F.relu(out)

        fbi = self.out2fbi(out)
        # fbi = F.softplus(fbi, beta=act_sharpness)
        fbi = F.relu(fbi)
        return out, fbi


    def forward_tau(self, input_pattern, out_preact, fbi_preact, out0, fbi0, act_sharpness=4, tau=1.5):
        out = out_preact + (-out_preact + self.in2out(input_pattern) + self.fbi2out(fbi0)) / tau
        # out = F.softplus(out, beta=act_sharpness)
        out = F.relu(out)

        fbi = fbi_preact + (-fbi_preact + self.out2fbi(out)) / tau
        # fbi = F.softplus(fbi, beta=act_sharpness)
        fbi = F.relu(fbi)
        return out, fbi, out_preact, fbi_preact


    def forward_tau_inh(self, input_pattern, out_preact, fbi_preact, out0, fbi0, act_sharpness=4, tau=2):
        out = out_preact + (-out_preact + self.in2out(input_pattern) + self.fbi2out(fbi0)) / tau
        # out = F.softplus(out, beta=act_sharpness)
        out = F.relu(out)

        fbi = fbi_preact + (-fbi_preact + self.out2fbi(out0) + self.fbi2fbi(fbi0)) / tau
        # fbi = F.softplus(fbi, beta=act_sharpness)
        fbi = F.relu(fbi)
        return out, fbi, out_preact, fbi_preact


    def forward_rnn(self, pattern):
        output = torch.zeros(self.out_size)
        fbi = torch.zeros(self.fbi_size)
        for t in range(self.num_timesteps):  # iterate through all timepoints of the RNN
            with torch.no_grad():
                output, fbi = self.forward(pattern, output, fbi)
        return output


    def train(self, num_epochs, num_timesteps, num_BPTT_steps, eval_step, all_patterns, learning_rate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        num_patterns = all_patterns.shape[0]
        self.num_timesteps = num_timesteps

        output_history = torch.zeros(self.out_size, num_timesteps, num_patterns, num_epochs)
        fbi_history = torch.zeros(self.fbi_size, num_timesteps, num_patterns, num_epochs)
        loss_history = torch.zeros(num_epochs)
        weight_history = {'in2out': torch.zeros(self.out_size, self.in_size, num_epochs),
                          'out2fbi': torch.zeros(self.fbi_size, self.out_size, num_epochs),
                          'fbi2out': torch.zeros(self.out_size, self.fbi_size, num_epochs),
                          'fbi2fbi': torch.zeros(self.fbi_size, self.fbi_size, num_epochs)}

        for epoch in tqdm(range(num_epochs)):
            for pattern_idx in torch.randperm(num_patterns):
                pattern = all_patterns[pattern_idx]

                output = torch.zeros(self.out_size)
                fbi = torch.zeros(self.fbi_size)
                out_preact = torch.zeros(self.out_size)
                fbi_preact = torch.zeros(self.fbi_size)

                target = torch.zeros(2)
                target[0] = XOR(pattern)
                target[1] = AND(pattern)

                loss = 0
                for t in range(num_timesteps):  # iterate through all timepoints of the RNN
                    if t >= (
                            eval_step - num_BPTT_steps) and t <= eval_step:  # truncate BPTT to only evaluate n steps from the end
                        track_grad = True
                    else:
                        track_grad = False

                    with torch.set_grad_enabled(track_grad):
                        # output, fbi = self.forward(pattern, output, fbi)
                        # output, fbi, out_preact, fbi_preact = self.forward_tau(pattern, out_preact, fbi_preact, output, fbi)
                        output, fbi, out_preact, fbi_preact = self.forward_tau_inh(pattern, out_preact, fbi_preact, output, fbi)

                    output_history[:, t, pattern_idx, epoch] = output.detach()
                    fbi_history[:, t, pattern_idx, epoch] = fbi.detach()

                    if t == eval_step:
                        loss += criterion(output, target)

                weight_history['in2out'][:,:,epoch] = self.in2out.weight.detach()
                weight_history['out2fbi'][:,:,epoch] = self.out2fbi.weight.detach()
                weight_history['fbi2out'][:,:,epoch] = self.fbi2out.weight.detach()
                weight_history['fbi2fbi'][:,:,epoch] = self.fbi2fbi.weight.detach()

                loss_history[epoch] = loss.detach()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.in2out.weight.data = self.in2out.weight.data.clamp(min=0, max=None)
                self.out2fbi.weight.data = self.out2fbi.weight.data.clamp(min=0, max=None)
                self.fbi2out.weight.data = self.fbi2out.weight.data.clamp(min=None, max=0)
                self.fbi2fbi.weight.data = self.fbi2fbi.weight.data.clamp(min=None, max=0)

        return output_history, fbi_history, loss_history, weight_history