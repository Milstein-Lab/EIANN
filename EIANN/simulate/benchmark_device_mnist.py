import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import click

from EIANN import Network
from EIANN.utils import read_from_yaml


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../config/MNIST/20230628_EIANN_1_hidden_mnist_bpDale_softplus_SGD_config_F.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--train-steps", type=int, default=3000)
@click.option("--device", type=click.Choice(['cpu', 'cuda']), default='cpu')
@click.option("--status-bar", is_flag=True)
def main(config_file_path, output_dir, train_steps, device, status_bar):
    """

    :param config_file_path:
    :param output_dir:
    :param train_steps:
    :param device:
    :param status_bar:
    """
    seed_start = 1
    num_instances = 1
    network_id = 1
    task_id = 2
    data_seed_start = 1
    epochs = 1
    val_interval = (-2001, -1, 100)

    seed = int.from_bytes((network_id, task_id, seed_start), byteorder='big')
    data_seed = int.from_bytes((network_id, data_seed_start), byteorder='big')
    
    network_config = read_from_yaml(config_file_path)
    layer_config = network_config['layer_config']
    projection_config = network_config['projection_config']
    training_kwargs = network_config['training_kwargs']
    
    # Load dataset
    tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=output_dir + '/datasets/MNIST_data/', train=True,
                                                     download=True, transform=tensor_flatten)
    
    # Add index to train & test data
    MNIST_train = []
    for idx, (data, target) in enumerate(MNIST_train_dataset):
        target = torch.eye(len(MNIST_train_dataset.classes))[target]
        MNIST_train.append((idx, data, target))
    
    # Put data in dataloader
    data_generator = torch.Generator()
    train_sub_dataloader = \
        torch.utils.data.DataLoader(MNIST_train[0:train_steps], shuffle=True, generator=data_generator)
    val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10000:], batch_size=10000, shuffle=False)

    network = Network(layer_config, projection_config, seed=seed, device=device, **training_kwargs)

    data_generator.manual_seed(data_seed)
    network.train_and_validate(train_sub_dataloader,
                               val_dataloader,
                               epochs=epochs,
                               val_interval=val_interval,
                               status_bar=status_bar)


if __name__ == '__main__':
    main(standalone_mode=False)