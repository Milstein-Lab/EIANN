import torch
from torch.utils.data import DataLoader
import click

import matplotlib.pyplot as plt
pt.update_plot_defaults()

from EIANN import Network
import EIANN.utils as ut
import EIANN.plot as pt


@click.command()
@click.option("--show", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--network_config1", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option("--network_config2", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)

def main(show, save, network_config1, network_config2):
    pass

if __name__ == '__main__':
    main(standalone_mode=False)


# Setup
input_size = 21
dataset = torch.eye(input_size)  # each row is a different pattern
target = torch.eye(dataset.shape[0])

data_seed = 0
data_generator = torch.Generator()
sample_indexes = torch.arange(len(dataset))
dataloader = DataLoader(list(zip(sample_indexes, dataset, target)),
                        shuffle=True,
                        generator=data_generator)

test_dataloader = DataLoader(list(zip(sample_indexes, dataset, target)),
                             batch_size=21)
epochs = 30
seed = 42


# Gjorgjieva network
network_config = ut.read_from_yaml('../optimize/data/20220902_EIANN_1_hidden_Gjorgjieva_Hebb_config_A.yaml')
layer_config = network_config['layer_config']
projection_config = network_config['projection_config']
training_kwargs = network_config['training_kwargs']

gj_network = Network(layer_config, projection_config, seed=seed, **training_kwargs)

data_generator.manual_seed(data_seed)
gj_network.train(dataloader, epochs, store_history=True, store_params=True, status_bar=True)
pt.plot_test_loss_history(gj_network, test_dataloader)
plt.savefig('figures/gjNet_raw_loss_history.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/gjNet_raw_loss_history.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)

min_loss_sorting = ut.get_optimal_sorting(gj_network, test_dataloader)
ut.recompute_history(gj_network, min_loss_sorting)
pt.plot_test_loss_history(gj_network, test_dataloader)
plt.savefig('figures/gjNet_sorted_loss_history.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/gjNet_sorted_loss_history.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)

flat_param_history_gj,_ = pt.get_flat_param_history(gj_network)
pt.plot_param_history_PCs(flat_param_history_gj)
plt.savefig('figures/gjNet_pca.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/gjNet_pca.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)

pt.plot_loss_landscape(test_dataloader, gj_network, num_points=20)
plt.savefig('figures/gjNet_loss_landscape.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/gjNet_loss_landscape.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)

# Backprop network
network_config = ut.read_from_yaml('../config/EIANN_1_hidden_backprop_softplus_SGD_matched_config.yaml')
layer_config = network_config['layer_config']
projection_config = network_config['projection_config']
training_kwargs = network_config['training_kwargs']

bp_network = Network(layer_config, projection_config, seed=seed, **training_kwargs)

gj_initial_state = gj_network.param_history[0]
bp_network.load_state_dict(gj_initial_state) # initialize backprop net with same weights as Gjorg. init

data_generator.manual_seed(data_seed)
bp_network.train(dataloader, epochs, store_history=True, store_params=True, status_bar=True)

for layer in bp_network:  # swap to ReLU activation to make loss comparable across networks
    for population in layer:
        population.activation = torch.nn.ReLU()

pt.plot_test_loss_history(bp_network, test_dataloader)
plt.savefig('figures/bpNet_loss_history.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/bpNet_loss_history.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)

flat_param_history_bp,_ = pt.get_flat_param_history(bp_network)
pt.plot_param_history_PCs(flat_param_history_bp)
plt.savefig('figures/bpNet_pca.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/bpNet_pca.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)

pt.plot_loss_landscape(test_dataloader, bp_network, num_points=20)
plt.savefig('figures/bpNet_loss_landscape.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/bpNet_loss_landscape.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)

# Combined loss landscape
flat_param_history1,_ = pt.get_flat_param_history(gj_network)
flat_param_history2,_ = pt.get_flat_param_history(bp_network)
combined_param_history = torch.cat([flat_param_history1,flat_param_history2])
pt.plot_param_history_PCs(combined_param_history)

gj_network.name = 'Gjorgjieva'
bp_network.name = 'Backprop'
pt.plot_loss_landscape(test_dataloader, gj_network, bp_network, num_points=20, extension=0.5)
plt.savefig('figures/bp_gj_combined_loss_landscape.png',edgecolor='white',dpi=300,facecolor='white',transparent=True)
plt.savefig('figures/bp_gj_combined_loss_landscape.svg',edgecolor='white',dpi=300,facecolor='white',transparent=True)
