import torch
import torchvision
import torchvision.transforms as T
from EIANN import Network
import EIANN.utils as ut
import click
import time, os


def get_network(config_file_path, network_seed):
    network_config = ut.read_from_yaml(config_file_path)
    layer_config = network_config['layer_config']
    projection_config = network_config['projection_config']
    training_kwargs = network_config['training_kwargs']

    return Network(layer_config, projection_config, seed=network_seed, **training_kwargs)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", required=True, type=click.Path(exists=True, file_okay=True,
                                                                   dir_okay=False))
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data/mnist/datasets/MNIST_data/')
def main(config_file_path, data_dir):
    # Load dataset
    tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=False,
                                                     transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False,
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
    train_dataloader = torch.utils.data.DataLoader(MNIST_train[0:50000], shuffle=True, generator=data_generator)
    val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10000:], batch_size=10000, shuffle=False)
    
    data_seed = 257
    network_seed = 66049
    
    data_generator.manual_seed(data_seed)
    network = get_network(config_file_path, network_seed)
    
    print(network.device)
    return
    
    current_time = time.time()
    network.train(train_dataloader, val_dataloader, samples_per_epoch=2000, val_interval=(-2001, -1, 100),
                  status_bar=True, debug=True)
    print(time.time() - current_time)
    

if __name__ == '__main__':
    main(standalone_mode=False)