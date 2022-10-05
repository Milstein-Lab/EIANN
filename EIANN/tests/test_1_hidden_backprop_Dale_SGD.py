import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pprint
from EIANN import Network
from EIANN.utils import read_from_yaml, test_EIANN_config

epochs=300
data_seed=0
network_seed=42
shuffle=True

input_size = 21
dataset = torch.eye(input_size)
target = torch.eye(dataset.shape[0])

sample_indexes = torch.arange(len(dataset))
data_generator = torch.Generator()
data_generator.manual_seed(data_seed)
dataloader = DataLoader(list(zip(sample_indexes, dataset, target)), shuffle=shuffle, generator=data_generator)

network_config = read_from_yaml('../optimize/data/20220915_EIANN_1_hidden_backprop_softplus_SGD_config.yaml')
pprint.pprint(network_config)

layer_config = network_config['layer_config']
projection_config = network_config['projection_config']
training_kwargs = network_config['training_kwargs']

network = Network(layer_config, projection_config, seed=network_seed, **training_kwargs)
test_EIANN_config(network, dataloader, epochs, supervised=False)

