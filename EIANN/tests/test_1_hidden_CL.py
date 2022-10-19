import torch
from torch.utils.data import DataLoader
from EIANN import Network
from EIANN.utils import test_EIANN_CL_config, read_from_yaml, write_to_yaml
import numpy as np
import matplotlib.pyplot as plt
import pprint
import click


@click.command()
@click.option("--epochs", type=int, default=300)
@click.option("--seed", type=int, default=42)
@click.option("--data-seed", type=int, default=0)
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
def main(epochs, seed, data_seed, config_file_path):
    """

    :param config_file_path: str path to .yaml
    """
    input_size = 21
    dataset = torch.eye(input_size)
    target = torch.eye(dataset.shape[0])

    data_generator = torch.Generator()
    sample_indexes = torch.arange(len(dataset))
    dataloader = DataLoader(list(zip(sample_indexes, dataset, target)), shuffle=True, generator=data_generator)

    network_config = read_from_yaml(config_file_path)
    pprint.pprint(network_config)

    layer_config = network_config['layer_config']
    projection_config = network_config['projection_config']
    training_kwargs = network_config['training_kwargs']

    network = Network(layer_config, projection_config, seed=seed, **training_kwargs)
    data_generator.manual_seed(data_seed)
    test_EIANN_CL_config(network, dataloader, epochs, split=0.75, generator=data_generator)

    plt.show()

if __name__ == '__main__':
    main(standalone_mode=False)