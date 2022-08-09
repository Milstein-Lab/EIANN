import torch
import torch.nn as nn
from tqdm import tqdm
import itertools

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'axes.linewidth':1.2,
                     'xtick.major.size': 6,
                     'xtick.major.width': 1.2,
                     'ytick.major.size': 6,
                     'ytick.major.width': 1.2,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     'figure.figsize': [14.0, 4.0],})
'''
params_dict = {'layer_0':
                   {'E':
                        {'n': 7,
                         'projections': []}},
               'layer_1':
                   {'E':
                        {'n': 100,
                         'bias': False,
                         'activation': 'relu',
                         'projections': ['layer_0']}},
               'layer_2':
                   {'E':
                        {'n': 10,
                         'bias': False,
                         'activation': 'relu',
                         'projections': ['layer_1']}}
               }

model = universalNet(params_dict)
'''


activation_dict = {'linear': nn.Identity(),
                   'relu': nn.ReLU(),
                   'sigmoid': nn.Sigmoid(),
                   'softplus': nn.Softplus(beta=4)}

def Hebb(pre, post, *args):
    delta_W = torch.outer(post, pre)
    return delta_W

def Oja(pre, post, W, *args):
    delta_W = torch.outer(post, pre) - W * (post ** 2).unsqueeze(1)
    return delta_W

def BCM(pre, post, W, theta):
    delta_W = torch.outer(post, pre) * (post - theta).unsqueeze(1)
    return delta_W

class universalNet(nn.Module):
    def __init__(self, input_size, tau=1, seed=42, dales_law=True):
        super().__init__()
        self.tau = tau
        self.seed = seed
        self.dales_law = dales_law
        self.theta_tau = 1 # for BCM learning
        self.theta_k = 1 # for BCM learning

        self.nn_modules = nn.ModuleDict()
        self.nn_parameters = nn.ParameterDict()

        self.input_layer = Layer('input_layer', populations={'E':input_size})

        self.layer1 = Layer('layer1', populations={'E':5, 'I':1})
        self.layer1.E.update(self, activation='softplus')
        self.layer1.E.add_projections(self, [self.input_layer.E])

        self.layer2 = Layer('layer2', populations={'E':21})
        self.layer2.E.update(self, activation='softplus', learn_bias=True)
        self.layer2.E.add_projections(self, [self.input_layer.E, self.layer1.E], learning_rule='Oja')

        self.init_weights()

    def forward(self, input_pattern):
        for i, layer in enumerate(self):
            for j, population in enumerate(layer):
                if i==0 and j==0: # set activity for the first population of the input layer
                    population.activity = input_pattern
                else:
                    delta_state = -population.state # decay previous activity
                    delta_theta = -population.theta_BCM
                    for projection in population:
                        delta_state += projection(projection.pre.activity)
                        delta_theta += population.activity**2 / self.theta_k
                    population.state += population.bias + delta_state / self.tau
                    population.theta_BCM += delta_theta / self.theta_tau

                    population.activity = population.activation(population.state)

                population.activity_history_ls.append(population.activity.detach())
                population.bias_history_ls.append(population.bias.detach())

    def train(self, num_epochs, all_patterns, all_targets, lr, num_timesteps=1, num_BPTT_steps=1, plot=False):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        loss_history = []
        weight_history = {}
        for param_name, tensor in self.named_parameters():
            if 'weight' in param_name:
                name = param_name.split('.')[1]
                weight_history[name] = [tensor.detach()]

        num_patterns = all_patterns.shape[0]

        for epoch in tqdm(range(num_epochs)):
            for pattern_idx in torch.randperm(num_patterns):
                input_pattern = all_patterns[pattern_idx]
                target = all_targets[pattern_idx]

                # Reset activities
                for layer in self:
                    for population in layer:
                        population.state = torch.zeros(population.size)
                        population.activity = torch.zeros(population.size)

                for t in range(num_timesteps):
                    if t >= (num_timesteps - num_BPTT_steps):  # truncate BPTT to only evaluate n steps from the end
                        track_grad = True
                    else:
                        track_grad = False

                    with torch.set_grad_enabled(track_grad):
                        self.forward(input_pattern)

                output = self.layer2.E.activity
                loss = self.criterion(output,target)
                loss_history.append(loss.detach())

                self.update_params(loss)

                if self.dales_law:
                    self.rectify_weights()

                # Save weights
                for param_name, tensor in self.named_parameters():
                    if 'weight' in param_name:
                        name = param_name.split('.')[1]
                        weight_history[name].append(tensor.detach())

        self.loss_history = torch.tensor(loss_history)
        self.weight_history = {proj_name: torch.stack(weights) for proj_name, weights in weight_history.items()}

        if plot:
            self.plot_summary()

    def init_weights(self):
        torch.manual_seed(self.seed)
        for layer in self:
            for population in layer:
                for projection in population:
                    distribution = projection.weight_distribution + '_'
                    getattr(projection.weight.data, distribution)(*projection.init_weight_bounds) # convert distribution name into pytorch callable

    def rectify_weights(self):
        for layer in self:
            for population in layer:
                for projection in population:
                    if projection.pre.name == 'E':
                        projection.weight.data = projection.weight.data.clamp(min=0, max=None)
                    elif projection.pre.name == 'I':
                        projection.weight.data = projection.weight.data.clamp(min=None, max=0)
                    else:
                        raise RuntimeWarning("Population name must be 'E' or 'I' when using Dale's law. Weights not rectified")

    def update_params(self, loss):
        if any([param.requires_grad for param in self.parameters()]):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for layer in self:
            for population in layer:
                population.step_bias()
                for projection in population:
                    if projection.learning_rule != 'backprop':
                        delta_W = projection.learning_rule(pre = projection.pre.activity, post = population.activity,
                                                           w = projection.weight, theta = population.theta_BCM)
                        projection.weight.data += projection.lr * delta_W

    def plot_summary(self):
        fig, ax = plt.subplots(2, 3, figsize=(14,8))
        axes = (0,0)
        ax[axes].plot(self.loss_history)
        ax[axes].set_xlabel('Training steps (epochs*patterns)')
        ax[axes].set_ylabel('Loss')
        ax[axes].set_ylim(bottom=0)

        for name,weights in self.weight_history.items():
            axes = (0,1)
            avg_weight = torch.mean(weights,dim=(1,2))
            ax[axes].plot(avg_weight,label=name)
            ax[axes].set_xlabel('Training steps (epochs*patterns)')
            ax[axes].set_ylabel('Average weight')
            ax[axes].legend()

        w = self.weight_history['layer1E_to_layer2E'].flatten(1)
        axes = (0,2)
        ax[axes].plot(w)
        ax[axes].set_xlabel('Training steps (epochs*patterns)')
        ax[axes].set_ylabel('weight')
        ax[axes].set_title('layer1E_to_layer2E')

        axes = (1, 0)
        for layer in self:
            for population in layer:
                avg_bias = torch.mean(population.bias_history, dim=1)
                ax[axes].plot(avg_bias)
                ax[axes].set_xlabel('Training steps (epochs*patterns)')
                ax[axes].set_ylabel('bias')

        ax[1,1].axis('off')
        ax[1,2].axis('off')

        plt.tight_layout()
        plt.show()

    def __iter__(self):
        for key,value in self._modules.items():
            if isinstance(value, Layer):
                yield value


class Layer(nn.Module):
    def __init__(self, name, populations):
        super().__init__()
        self.name = name
        self.populations = populations
        for pop,size in populations.items():
            self.__dict__[pop] = Population(layer=name, name=pop, size=size)

    def __iter__(self):
        for key,value in self.__dict__.items():
            if callable(value): #only iterate over Populations
                yield value

    def __repr__(self):
        ls = []
        for name,value in self.populations.items():
            if name[0] != '_':
                ls.append(f'{name}: {value}')
        items = ", ".join(ls)
        return f'{type(self)} :\n\t({items})'


class Population(nn.Module):
    def __init__(self, layer, name, size):
        super().__init__()
        # Hyperparameters
        self.layer = layer
        self.name = name
        self.fullname = self.layer + self.name
        self.size = size
        self.activation = 'linear'
        self.bias_rule = 'backprop'
        self.learn_bias = False
        self.inputs = []

        # State variables
        self.state = torch.zeros(self.size)
        self.activity = torch.zeros(self.size)
        self.bias = torch.zeros(size)
        self.theta_BCM = torch.zeros(size)
        self.activity_history_ls = []
        self.bias_history_ls = []

    def update(self, network, activation='linear', learn_bias=False, bias_rule='backprop'):
        self.activation = activation
        self.learn_bias = learn_bias
        self.bias_rule = bias_rule

        if self.learn_bias == True:
            name = 'bias_'+self.fullname
            self.bias = nn.Parameter(self.bias)
            network.nn_parameters[name] = self.bias

    def step_bias(self):
        return

    def add_projections(self, network, pre_populations, learning_rule='backprop'):
        for pre_pop in pre_populations:
            name = pre_pop.layer + pre_pop.name
            self.inputs.append(name)
            self.__dict__[name] = nn.Linear(pre_pop.size, self.size, bias=False)

            # Set projection attributes
            projection = self.__dict__[name]
            projection.name = f'{name}_to_{self.fullname}'
            projection.pre = pre_pop
            projection.weight_distribution = 'uniform'
            projection.init_weight_bounds = (0, 1)
            projection.lr = 0.01
            projection.learning_rule = learning_rule
            if learning_rule != 'backprop':
                projection.weight.requires_grad = False
                projection.learning_rule = lambda pre, post, w, theta: globals()[learning_rule](pre, post, w, theta)

            # Add to ModuleList to make projection a trainable parameter
            network.nn_modules[projection.name] = projection

    def __iter__(self):
        for key,value in self.__dict__.items():
            if isinstance(value, nn.Linear): #only iterate over projections
                yield value

    def __repr__(self):
        ls = []
        for name in self.__dict__:
            if name[0] != '_':
                ls.append(name)
        items = ", ".join(ls)
        return f'{type(self)} :\n\t({items})'

    @property
    def activation(self):
        # Automatically turn activation string into a callable function
        return activation_dict[self._act_name]
    @activation.setter
    def activation(self, act_name):
        self._act_name = act_name

    @property
    def activity_history(self):
        return torch.stack(self.activity_history_ls)

    @property
    def bias_history(self):
        return torch.stack(self.bias_history_ls)


def n_hot_patterns(n,length):
    all_permutations = torch.tensor(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = torch.sum(all_permutations,axis=1)
    idx = torch.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns