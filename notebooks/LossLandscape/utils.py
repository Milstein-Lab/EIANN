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

def plot_summary(time,epoch=-1,data=[]):
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

    # Initial Weights
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    axis = 0
    im = ax[axis].imshow(weight_history['in2out'][:, :, 0], aspect='equal', cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('input unit')
    ax[axis].set_ylabel('output unit')
    ax[axis].set_title('in2out', fontsize=15)

    axis = 1
    im = ax[axis].imshow(weight_history['out2fbi'][:, :, 0].T, aspect='equal', cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_ylabel('output unit')
    ax[axis].set_xlabel('fbi unit')
    ax[axis].set_title('out2fbi', fontsize=15)

    axis = 2
    im = ax[axis].imshow(weight_history['fbi2out'][:, :, 0], aspect='equal', cmap='Blues_r', vmax=0)
    plt.colorbar(im, ax=ax[axis])
    ax[axis].set_xlabel('fbi unit')
    ax[axis].set_ylabel('output unit')
    ax[axis].set_title('fbi2out', fontsize=15)

    plt.suptitle("Initial Weights", fontsize=20)
    plt.tight_layout()
    plt.show()

    # Final Weights
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
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

    plt.suptitle("Final Weights", fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_activity(time,data):
    all_patterns, output_history, fbi_history, loss_history, weight_history = data
    output_size = 2
    input_size = 2
    fbi_size = 1
    epoch = -1

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
    plt.show()

#######################################################
# Loss Function Plotting

def loss_function(model, w, b, target_func):
    all_patterns = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    model.in2out.weight.data.fill_(w)
    model.in2out.bias.detach()[1] = b

    loss = 0
    for x in all_patterns:
        y_predicted = model.forward_rnn(x,num_timesteps=9)
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
    def __init__(self, input_size, output_size, fbi_size, bias, rule='bp'):
        super().__init__()
        self.out_size = output_size
        self.fbi_size = fbi_size
        self.in_size = input_size

        self.in2out = nn.Linear(self.in_size, self.out_size, bias=bias)
        self.out2fbi = nn.Linear(self.out_size, self.fbi_size, bias=False)
        self.fbi2out = nn.Linear(self.fbi_size, self.out_size, bias=False)

        if rule=='btsp':
            self.in2out.weight.requires_grad = False
            self.in2out.bias.requires_grad = False
            self.out2fbi.weight.requires_grad = False
            self.fbi2out.weight.requires_grad = False

        # # initialize weights
        self.in2out.weight.data.uniform_(0.1,0.5)
        self.out2fbi.weight.data.uniform_(0.1,0.5)
        self.fbi2out.weight.data.uniform_(-0.5,-0.1)

        # initialize close to optimal weights
        self.in2out.weight.data.fill_(0.1)
        self.out2fbi.weight.data = torch.tensor([[0., 1.5]])
        self.fbi2out.weight.data = torch.tensor([[-1.5],[-0.]])


    def forward(self, input_pattern, out_preact, fbi_preact, out0, fbi0, act_sharpness=4, tau=1.):
        out = out_preact + (-out_preact + self.in2out(input_pattern) + self.fbi2out(fbi0)) / tau
        # out = F.softplus(out, beta=act_sharpness)
        out = F.relu(out)

        fbi = fbi_preact + (-fbi_preact + self.out2fbi(out)) / tau
        # fbi = F.softplus(fbi, beta=act_sharpness)
        fbi = F.relu(fbi)
        return out, fbi, out_preact, fbi_preact


    def forward_rnn(self, pattern, num_timesteps, tau=1.):
        output = torch.zeros(self.out_size)
        out_preact = torch.zeros(self.out_size)
        fbi_preact = torch.zeros(self.fbi_size)
        fbi = torch.zeros(self.fbi_size)

        with torch.no_grad():
            for t in range(num_timesteps):  # iterate through all timepoints of the RNN
                output, fbi, out_preact, fbi_preact = self.forward(pattern, out_preact, fbi_preact, output, fbi, tau)

        return output


    def train(self, learning_rate, num_epochs, num_timesteps, num_BPTT_steps, eval_step, all_patterns, all_targets, tau=1.):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        num_patterns = all_patterns.shape[0]

        output_history = torch.zeros(self.out_size, num_timesteps, num_patterns, num_epochs)
        fbi_history = torch.zeros(self.fbi_size, num_timesteps, num_patterns, num_epochs)
        loss_history = torch.zeros(num_epochs)
        weight_history = {'in2out': torch.zeros(self.out_size, self.in_size, num_epochs+1),
                          'out2fbi': torch.zeros(self.fbi_size, self.out_size, num_epochs+1),
                          'fbi2out': torch.zeros(self.out_size, self.fbi_size, num_epochs+1)}

        weight_history['in2out'][:,:,0] = self.in2out.weight.detach()
        weight_history['out2fbi'][:,:,0] = self.out2fbi.weight.detach()
        weight_history['fbi2out'][:,:,0] = self.fbi2out.weight.detach()

        for epoch in tqdm(range(num_epochs)):
            for pattern_idx in torch.randperm(num_patterns):
                pattern = all_patterns[pattern_idx]
                target = all_targets[pattern_idx]

                output = torch.zeros(self.out_size)
                fbi = torch.zeros(self.fbi_size)
                out_preact = torch.zeros(self.out_size)
                fbi_preact = torch.zeros(self.fbi_size)

                loss = 0
                for t in range(num_timesteps):  # iterate through all timepoints of the RNN
                    if t >= (
                            eval_step - num_BPTT_steps) and t <= eval_step:  # truncate BPTT to only evaluate n steps from the end
                        track_grad = True
                    else:
                        track_grad = False

                    with torch.set_grad_enabled(track_grad):
                        output, fbi, out_preact, fbi_preact = self.forward(pattern, out_preact, fbi_preact, output, fbi, tau)

                    output_history[:, t, pattern_idx, epoch] = output.detach()
                    fbi_history[:, t, pattern_idx, epoch] = fbi.detach()

                    if t == eval_step:
                        loss += criterion(output, target)

                weight_history['in2out'][:,:,epoch+1] = self.in2out.weight.detach()
                weight_history['out2fbi'][:,:,epoch+1] = self.out2fbi.weight.detach()
                weight_history['fbi2out'][:,:,epoch+1] = self.fbi2out.weight.detach()

                loss_history[epoch] = loss.detach()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.in2out.weight.data = self.in2out.weight.data.clamp(min=0, max=None)
                self.out2fbi.weight.data = self.out2fbi.weight.data.clamp(min=0, max=None)
                self.fbi2out.weight.data = self.fbi2out.weight.data.clamp(min=None, max=0)

        data_list = [all_patterns, output_history, fbi_history, loss_history, weight_history]
        return data_list


    def train_btsp(self, lr, num_epochs, num_timesteps, eval_step, all_patterns, all_targets, tau=1.):
        criterion = nn.MSELoss()
        num_patterns = all_patterns.shape[0]
        output_history = torch.zeros(self.out_size, num_timesteps, num_patterns, num_epochs)
        fbi_history = torch.zeros(self.fbi_size, num_timesteps, num_patterns, num_epochs)
        loss_history = torch.zeros(num_epochs)
        weight_history = {'in2out': torch.zeros(self.out_size, self.in_size, num_epochs+1),
                          'out2fbi': torch.zeros(self.fbi_size, self.out_size, num_epochs+1),
                          'fbi2out': torch.zeros(self.out_size, self.fbi_size, num_epochs+1)}
        weight_history['in2out'][:,:,0] = self.in2out.weight
        weight_history['out2fbi'][:,:,0] = self.out2fbi.weight
        weight_history['fbi2out'][:,:,0] = self.fbi2out.weight

        for epoch in tqdm(range(num_epochs)):
            for pattern_idx in torch.randperm(num_patterns):
                pattern = all_patterns[pattern_idx]
                target = all_targets[pattern_idx]

                output = torch.zeros(self.out_size)
                fbi = torch.zeros(self.fbi_size)
                out_preact = torch.zeros(self.out_size)
                fbi_preact = torch.zeros(self.fbi_size)

                loss = 0
                for t in range(num_timesteps):  # iterate through all timepoints of the RNN
                    output, fbi, out_preact, fbi_preact = self.forward(pattern, out_preact, fbi_preact, output, fbi, tau)
                    output_history[:, t, pattern_idx, epoch] = output
                    fbi_history[:, t, pattern_idx, epoch] = fbi

                    if t == eval_step:
                        loss += criterion(output, target)

                # update weights
                d_weight, d_bias = btsp_step(self.in2out.weight, pattern, output, target)
                self.in2out.weight.data += d_weight * lr
                self.in2out.bias.data += d_bias * lr

                weight_history['in2out'][:,:,epoch+1] = self.in2out.weight
                weight_history['out2fbi'][:,:,epoch+1] = self.out2fbi.weight
                weight_history['fbi2out'][:,:,epoch+1] = self.fbi2out.weight

                loss_history[epoch] = loss

                self.in2out.weight.data = self.in2out.weight.data.clamp(min=0, max=None)
                self.out2fbi.weight.data = self.out2fbi.weight.data.clamp(min=0, max=None)
                self.fbi2out.weight.data = self.fbi2out.weight.data.clamp(min=None, max=0)

        data_list = [all_patterns, output_history, fbi_history, loss_history, weight_history]
        return data_list

#######################################################
# Learning rules

def scaled_single_sigmoid(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError(
            'scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return lambda xi: (target_amp / amp) * (1. / (1. + np.exp(-slope * (xi - th))) - start_val) + ylim[0]


def btsp_single(W, pre, output, target, Wmax=2):
    error = target - output
    if error > 0:
        alpha = 1.
    elif error <= 0:
        alpha = 0.2

    M = error**2
    el_trace = alpha * np.minimum(1.,pre.numpy())
    sigm_dep = np.vectorize(scaled_single_sigmoid(0.01,0.02), )

    d_weight = M * ( (Wmax-W)*el_trace - W*sigm_dep(el_trace) )
    d_bias = error
    return d_weight, d_bias


def btsp_step(W, pre, output, target, Wmax=2):
    error = target - output
    alpha = (error > 0).type(torch.float32)
    alpha[alpha == 0] = 0.2  # when error<0 set alpha to 0.2

    M = error ** 2
    el_trace = np.outer(alpha, np.minimum(1., pre)).T
    sigm_dep = np.vectorize(scaled_single_sigmoid(0.01, 0.02))

    d_weight = M * ((Wmax - W) * el_trace - W * sigm_dep(el_trace))
    d_bias = error
    return d_weight, d_bias


def plot_btsp():
    input_peak_rate = 2.0
    max_weight_scale = 2.

    pre = torch.linspace(0., input_peak_rate, 100)
    w0 = torch.linspace(0., max_weight_scale, 100)

    pre_mesh, w0_mesh = torch.meshgrid(pre, w0)
    plt.figure(figsize=(10,6))
    plt.pcolormesh(pre_mesh, w0_mesh,
                   btsp_single(W=w0_mesh, pre=pre_mesh, output=0., target=1., Wmax=max_weight_scale)[0],
                   cmap='RdBu_r', shading='nearest',
                   vmin=-max_weight_scale, vmax=max_weight_scale)
    plt.xlabel('Pre activity')
    plt.ylabel('Initial weight')
    cbar = plt.colorbar()
    cbar.set_label('Delta weight', rotation=-90.,labelpad=10)
    plt.title('BTSP Fixed Points')
    plt.show()