import line_profiler

import torch
import torchvision
import torchvision.transforms as T
from EIANN import Network
import EIANN.utils as ut

device = 'cpu'

# Load dataset
tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
MNIST_train_dataset = torchvision.datasets.MNIST(root='../datasets/MNIST_data/', train=True, download=False,
                                         transform=tensor_flatten)
MNIST_test_dataset = torchvision.datasets.MNIST(root='../datasets/MNIST_data/',
                                        train=False, download=False,
                                        transform=tensor_flatten)
# Add index to train & test data
MNIST_train = []
for idx,(data,target) in enumerate(MNIST_train_dataset):
    target = torch.eye(len(MNIST_train_dataset.classes))[target]
    MNIST_train.append((idx, data, target))

MNIST_test = []
for idx,(data,target) in enumerate(MNIST_test_dataset):
    target = torch.eye(len(MNIST_test_dataset.classes))[target]
    MNIST_test.append((idx, data, target))

# Put data in dataloader
data_generator = torch.Generator()
train_dataloader = torch.utils.data.DataLoader(MNIST_train, shuffle=True, generator=data_generator)
train_sub_dataloader = torch.utils.data.DataLoader(MNIST_train[0:10000], shuffle=True, generator=data_generator)
test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10000, shuffle=False)

network_seed = 42

# Create network
# network_config = ut.read_from_yaml('../config/MNIST/EIANN_1_hidden_mnist_backprop_Dale_relu_SGD_config.yaml')
network_config = ut.read_from_yaml('../config/MNIST/EIANN_1_hidden_mnist_backprop_relu_SGD_config.yaml')
# network_config = ut.read_from_yaml('../config/MNIST/EIANN_2_hidden_mnist_backprop_Dale_relu_SGD_config.yaml')

layer_config = network_config['layer_config']
projection_config = network_config['projection_config']
training_kwargs = network_config['training_kwargs']

bp_network = Network(layer_config, projection_config, seed=network_seed, device=device, **training_kwargs)

@profile
def forward(self, dataloader, store_history=False, store_dynamics=False, no_grad=True):
    for idx, data, target in dataloader:
        sample = data.squeeze(0).to(device)
        if len(sample.shape) > 1:
            batch_size = sample.shape[0]
        else:
            batch_size = 1
        for i, layer in enumerate(self):
            if i == 0:
                input_pop = next(iter(layer))
            for pop in layer:
                pop.reinit(self.device, batch_size=batch_size)
        input_pop.activity = torch.squeeze(sample)

        for t in range(self.forward_steps):
            if (t >= self.forward_steps - self.backward_steps) and not no_grad:
                track_grad = True
            else:
                track_grad = False
            with torch.set_grad_enabled(track_grad):
                for post_layer in self:
                    for post_pop in post_layer:
                        post_pop.prev_activity = post_pop.activity
                for i, post_layer in enumerate(self):
                    for post_pop in post_layer:
                        if i > 0:
                            for projection in post_pop:
                                pre_pop = projection.pre
                                if projection.update_phase in ['forward', 'all', 'F', 'A']:
                                    if projection.direction in ['forward', 'F']:
                                        delta_state = projection(pre_pop.activity)
                                    elif projection.direction in ['recurrent', 'R']:
                                        delta_state = projection(pre_pop.prev_activity)
                            post_pop.state = post_pop.state + (-post_pop.state + post_pop.bias + delta_state) / post_pop.tau
                            post_pop.activity = post_pop.activation(post_pop.state)
                        if store_dynamics:
                            post_pop.forward_steps_activity.append(post_pop.activity.detach().clone())

        if store_history:
            for layer in self:
                for pop in layer:
                    pop.activity_history_list.append(pop.forward_steps_activity)


forward(bp_network, train_dataloader)