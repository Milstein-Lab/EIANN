import torch
import torchvision
import torchvision.transforms as T
from tqdm.notebook import tqdm
import numpy as np
import click
import os
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt

from EIANN import Network
import EIANN.utils as ut
import EIANN.plot as pt


def load_data():
    # Load dataset
    tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root='../datasets/MNIST_data/', train=True, download=False,
                                                     transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root='../datasets/MNIST_data/',
                                                    train=False, download=False,
                                                    transform=tensor_flatten)

    # Add index to train & test data
    MNIST_train = []
    for idx, (data, target) in enumerate(MNIST_train_dataset):
        target = torch.eye(len(MNIST_train_dataset.classes))[target]
        MNIST_train.append((idx, data, target))

    MNIST_test = []
    for idx, (data, target) in enumerate(MNIST_test_dataset):
        target = torch.eye(len(MNIST_test_dataset.classes))[target]
        MNIST_test.append((idx, data, target))

    # Put data in dataloader
    data_generator = torch.Generator()
    train_dataloader = torch.utils.data.DataLoader(MNIST_train, shuffle=True, generator=data_generator)
    train_sub_dataloader = torch.utils.data.DataLoader(MNIST_train[0:10000], shuffle=True, generator=data_generator)
    val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10000:], batch_size=10000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10000, shuffle=False)

    return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator


#  python train_mnist_model.py --config_path=../optimize/data/mnist/20230220_1_hidden_mnist_BTSP_Clone_Dend_I_4.yaml
#  python train_mnist_model.py --config_path=../config/MNIST/EIANN_1_hidden_mnist_backprop_relu_SGD_config.yaml
#  python train_mnist_model.py --config_path=../optimize/data/mnist/20230102_EIANN_1_hidden_mnist_bpDale_softplus_config.yaml
#  python train_mnist_model.py --config_path=../optimize/data/mnist/20230214_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_C.yaml

@click.command()
@click.option("--config_path")
@click.option("--name", default=None)
@click.option("--export_path", default='../saved_networks/model_metrics.hdf5')
@click.option("--plot", is_flag=True, default=False)
@click.option("--retrain", is_flag=True, default=False)
@click.option("--epochs", default=1)
@click.option("--data_seed", default=0)
@click.option("--network_seed", default=42)
def main(config_path, name, export_path, plot, retrain, epochs, data_seed, network_seed):
    pt.update_plot_defaults()

    network_config = ut.read_from_yaml(config_path)
    layer_config = network_config['layer_config']
    projection_config = network_config['projection_config']
    training_kwargs = network_config['training_kwargs']
    network = Network(layer_config, projection_config, seed=network_seed, **training_kwargs)

    if name == None:
        name = config_path.split('/')[-1]
    network.name = name

    # Load data
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = load_data()

    # Load pretrained network if it exists
    saved_network_path = f"../saved_networks/{network.name}.pkl"
    if os.path.exists(saved_network_path) and retrain==False:
        network.load(saved_network_path)
    else:
        # Train network
        data_generator.manual_seed(data_seed)
        network.train_and_validate(train_sub_dataloader,
                                    test_dataloader,
                                    epochs=epochs,
                                    val_interval=(0, -1, 1000),
                                    store_history=True,
                                    store_weights=False,
                                    status_bar=True)
        network.save(file_name_base=network.name, dir='../saved_networks')

    # Compute receptive fields
    population = network.H1.E
    receptive_fields, _ = ut.compute_maxact_receptive_fields(population, test_dataloader, sigmoid=False)

    # Export metrics data to hdf5 file
    metrics_dict = ut.compute_representation_metrics(network.H1.E, test_dataloader, receptive_fields, plot=plot)
    metrics_dict['val_loss'] = network.val_loss_history
    metrics_dict['val_loss_steps'] = network.val_history_train_steps
    ut.export_metrics_data(metrics_dict, network.name, export_path)

    if plot:
        pt.plot_batch_accuracy(network, test_dataloader, population=network.H1.E)
        pt.plot_train_loss_history(network)
        pt.plot_rsm(network, test_dataloader)
        pt.plot_hidden_weights(network.module_dict['H1E_InputE'].weight, sort=True)
        if 'BTSP' in network.name:
            pt.plot_plateaus(population=btsp_network.H1.E, start=0, end=10000)
            sorted_plateaus, unit_ids = pt.plot_sorted_plateaus(btsp_network.Output.E, test_dataloader)
            sorted_plateaus, unit_ids = pt.plot_sorted_plateaus(btsp_network.H1.E, test_dataloader)
        pt.plot_correlations(network, test_dataloader)
        pt.plot_total_input(network.H1.E, test_dataloader, sorting='EI_balance', act_threshold=0)
    #     # Plot H1.E receptive fields
    #     _, activity_preferred_inputs = ut.compute_act_weighted_avg(population, test_dataloader)
    #     # _, activity_preferred_inputs = ut.compute_maxact_receptive_fields(population, test_dataloader, sigmoid=True)
    #     pt.plot_receptive_fields(receptive_fields, activity_preferred_inputs)
    #
    #     # Plot Output.E receptive fields
    #     population = network.Output.E
    #     _, activity_preferred_inputs = ut.compute_act_weighted_avg(population, test_dataloader)
    #     # _, activity_preferred_inputs = ut.compute_maxact_receptive_fields(population, test_dataloader, sigmoid=True)
    #     receptive_fields, _ = ut.compute_maxact_receptive_fields(population, test_dataloader, sigmoid=False)
    #     pt.plot_receptive_fields(receptive_fields, activity_preferred_inputs)


if __name__ == '__main__':
    main(standalone_mode=False)