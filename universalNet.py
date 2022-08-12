import torch
import torch.nn as nn
from tqdm import tqdm
from universalNet_utils import plot_summary

# hparams = {'seed': 42,
#            'dales_law': True}
#
# params_dict = {'layer0':
#                    {'E': {'n': 7}},
#                'layer1':
#                    {'E': {'n': 5,
#                          'activation': 'softplus',
#                          'inputs': ['layer0.E']},
#                     'I': {'n': 1}},
#                'layer2':
#                    {'E': {'n': 21,
#                          'bias': True,
#                          'inputs': ['layer0.E', 'layer1.E'],
#                          'learning_rule': 'Oja'}}}
#
# model = universalNet(params_dict, **hparams)

activation_dict = {'linear': nn.Identity(),
                   'relu': nn.ReLU(),
                   'sigmoid': nn.Sigmoid(),
                   'softplus': nn.Softplus(beta=4)}

# TODO: store order of presented patterns
# TODO: add kwargs as dict to population activation params eg softmax, {beta=4}
# TODO: store list/set of rules for every projection in a population (different rule for each proj)
# TODO: add weight bounds

def Hebb(pre, post):
    delta_W = torch.outer(post, pre)
    return delta_W

def Oja(pre, post, W):
    delta_W = torch.outer(post, pre) - W * (post**2).unsqueeze(1)
    return delta_W

def BCM(pre, post, theta):
    delta_W = torch.outer(post, pre) * (post - theta).unsqueeze(1)
    return delta_W


class universalNet(nn.Module):
    def __init__(self, params_dict, seed=42, dales_law=True):
        super().__init__()
        self.seed = seed
        self.dales_law = dales_law

        self.nn_modules = nn.ModuleDict()
        self.nn_parameters = nn.ParameterDict()

        # Create layer & population objects from params dict
        for layer_name in params_dict:
            populations = {}
            for population in params_dict[layer_name]:
                populations[population] = params_dict[layer_name][population].pop('n')
            self._modules[layer_name] = self._modules[layer_name] = Layer(layer_name, populations=populations)

            # Add projections and apply optional parameters to the population
            for population in params_dict[layer_name]:
                if params_dict[layer_name][population]:
                    self._modules[layer_name].__dict__[population].update(self, **params_dict[layer_name][population])

        self.output_pop = self._modules[layer_name].E # E pop of final layer

        # self.input_layer = Layer('input_layer', populations={'E':7})
        # self.layer1 = Layer('layer1', populations={'E':5, 'I':1})
        # self.layer1.E.update(self, activation='softplus',
        #                      inputs=['input_layer.E'])
        # self.layer2 = Layer('layer2', populations={'E':21})
        # self.layer2.E.update(self, activation='softplus', bias=True,
        #                      inputs=['input_layer.E', 'layer1.E'], learning_rule = 'Oja')

        self.init_weights()

    def forward(self, input_pattern, training=True):
        for i, layer in enumerate(self):
            for j, population in enumerate(layer):
                if i==0 and j==0: # pass pattern to first population of input layer
                    population.activity = input_pattern
                else:
                    population.prev_activity = population.activity
                    population.state = population.state - population.state/population.tau
                    for projection in population:
                        if projection.direction == 'FF':
                            population.state = population.state + projection(projection.pre.activity)/population.tau
                        elif projection.direction == 'FB':
                            population.state = population.state + projection(projection.pre.prev_activity)/population.tau
                    population.activity = population.activation(population.state + population.bias)

                if training:
                    population.activity_history_ls.append(population.activity.detach())
                    population.bias_history_ls.append(population.bias.detach())

    def train(self, num_epochs, all_patterns, all_targets, lr, num_timesteps=1, num_BPTT_steps=1, plot=False):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.all_patterns = all_patterns
        self.num_timesteps = num_timesteps

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

                # Reset activities & store history
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

                output = self.output_pop.activity
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
        self.weight_history = {projection_name: torch.stack(weights) for projection_name, weights in weight_history.items()}

        if plot:
            plot_summary(self)

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
                # TODO: add theta update conditional on population projections
                # delta_theta = (-population.theta_BCM + population.activity ** 2 / population.theta_k) / population.theta_tau
                # population.theta_BCM += delta_theta
                for projection in population:
                    if projection.learning_rule == 'Hebb':
                        delta_W = projection.delta_W(pre = projection.pre.activity,
                                                     post = population.activity)
                        projection.weight.data += projection.lr * delta_W

                    elif projection.learning_rule == 'Oja':
                        delta_W = projection.delta_W(pre = projection.pre.activity,
                                                     post = population.activity,
                                                     W = projection.weight)
                        projection.weight.data += projection.lr * delta_W

                    elif projection.learning_rule == 'BCM':
                        delta_W = projection.delta_W(pre = projection.pre.activity,
                                                     post = population.activity,
                                                     theta = population.theta_BCM)
                        delta_theta = (-population.theta_BCM + population.activity**2/population.theta_k) / population.theta_tau
                        population.theta_BCM += delta_theta
                        projection.weight.data += projection.lr * delta_W

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
        self.tau = 1
        self.activation = 'linear'
        self.bias_rule = 'backprop'
        self.learn_bias = False
        self.inputs = []
        self.theta_tau = 1 # for BCM learning
        self.theta_k = 1 # for BCM learning

        # State variables
        self.state = torch.zeros(self.size)
        self.activity = torch.zeros(self.size)
        self.prev_activity = torch.zeros(self.size)
        self.bias = torch.zeros(size)
        self.theta_BCM = torch.zeros(size)
        self.activity_history_ls = []
        self.bias_history_ls = []

    def update(self, network, activation='linear', bias=False, bias_rule='backprop', inputs=None, learning_rule='backprop'):
        self.activation = activation
        self.learn_bias = bias
        self.bias_rule = bias_rule

        if self.learn_bias == True:
            name = 'bias_'+self.fullname
            self.bias = nn.Parameter(self.bias)
            network.nn_parameters[name] = self.bias

        if inputs:
            self.add_projections(network, inputs, learning_rule)

    def step_bias(self):
        # TODO: add bias update (as nn.Parameter? or through nn.Linear?)
        return

    def add_projections(self, network, inputs, learning_rule='backprop'):
        for object_name in inputs:
            pre_layer, pre_pop = object_name.split('.')
            pre_name = pre_layer + pre_pop
            pre_pop = network._modules[pre_layer].__dict__[pre_pop]
            self.inputs.append(pre_name)
            self.__dict__[pre_name] = nn.Linear(pre_pop.size, self.size, bias=False)

            # TODO: add theta during projection init for BCM layers
            # TODO: create set of learning rules and separate backward passes for each
            # Set projection attributes
            projection = self.__dict__[pre_name]
            projection.pre = pre_pop
            projection.post = self
            projection.name = f'{pre_name}_to_{projection.post.fullname}'
            projection.direction = 'FF'
            projection.weight_distribution = 'uniform'
            projection.init_weight_bounds = (0, 1)
            projection.lr = 0.01
            projection.learning_rule = learning_rule
            if learning_rule != 'backprop':
                projection.weight.requires_grad = False
                projection.delta_W = lambda **args: globals()[learning_rule](**args)

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